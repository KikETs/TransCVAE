from pathlib import Path
import sys
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent
sys.path.append(str(PROJECT_ROOT))
from utils.dataloader import dataset
from utils.LSTM_util import *
import torch
from torch.nn.utils.parametrizations import weight_norm

class CVAE(nn.Module):
    def __init__(self, d_model=256, latent_dim = 64, hidden_dim = 128):
        super().__init__()
        mid=(d_model+latent_dim)//2
        self.to_means = nn.Sequential(
            nn.Linear(d_model, mid),
            nn.Dropout(0.1),
            nn.Linear(mid, latent_dim)
        )
        self.to_var = nn.Linear(d_model, latent_dim)
        self.to_decoder = nn.Linear(latent_dim, d_model * 2)
        self.to_prop_z = nn.Linear(latent_dim, 3)

        self.encoder = LSTM(input_size=d_model, hidden_size=d_model, num_layers=2, batch_first=True, dropout=0.2)
        self.decoder = LSTM(input_size=d_model, hidden_size=d_model, num_layers=2, batch_first=True, dropout=0.2)
        self.to_prop = nn.Linear(latent_dim, 3)

        self.predict = nn.Linear(d_model, dataset.vocab_size)

        self.input_embedding = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.GELU()
        )
        self.smiles_embbed = nn.Embedding(dataset.vocab_size, d_model, padding_idx=dataset.vocab['[PAD]'])
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var).to(device)
        eps = torch.randn_like(std).to(device)
        return mu + eps * std

    def forward(
        self,
        smiles_enc: torch.Tensor,   # [B, L_enc]  (<cond>+SMILES)
        smiles_tgt: torch.Tensor,   # [B, L_out]  (손실 정렬 길이)
        properties: torch.Tensor,   # [B, 3, 1] 또는 [B,3]
        mode: str = "gumbel",       # "gumbel" | "greedy"
        tau: float = 1.0            # Gumbel-Softmax 온도
    ):
        """
        GT 분기(teacher-forcing) 없이 오직 Pred(자기회귀)만 사용.
        mode="gumbel": Straight-Through Gumbel-Softmax (학습 권장, 미분 가능)
        mode="greedy": argmax 샘플링 (추론용, 비미분)
        """
        device = smiles_enc.device
        B, L_out = smiles_tgt.shape
        SOS = dataset.vocab['[SOS]']

        # properties 형상 보정: [B,3] -> [B,3,1]
        if properties.dim() == 2:
            properties = properties.unsqueeze(-1)

        # ── 1) Encoder ─────────────────────────────────────────────
        emb_src = self.smiles_embbed(smiles_enc)          # [B, L_enc, d_model]
        prop_emb = self.input_embedding(properties)       # [B, 3,     d_model]
        enc_in = torch.cat([emb_src, prop_emb], dim=1)    # [B, L_enc+3, d_model]

        # 마지막 hidden state로 μ, logσ² 계산
        h_enc = self.encoder(enc_in)[1][0][-1]            # [B, d_model]
        mu, logv = self.to_means(h_enc), self.to_var(h_enc)  # [B, z]
        z = self.reparameterize(mu, logv)                 # [B, z]

        # z → (h0, c0)
        h0, c0 = torch.tanh(self.to_decoder(z)).chunk(2, dim=-1)  # [B, d_model] x2
        h0 = h0.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)  # [num_layers, B, d_model]
        c0 = c0.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)

        # ── 2) 자기회귀 디코딩(오직 Pred) ─────────────────────────
        x_t = torch.full((B, 1), SOS, dtype=torch.long, device=device)  # [B,1]
        emb_t = self.smiles_embbed(x_t)                                  # [B,1,d_model]

        logits_list = []
        h, c = h0, c0
        emb_weight = self.smiles_embbed.weight                           # [V, d_model]

        for t in range(L_out):
            dec_out_t, (h, c) = self.decoder(emb_t, (h, c))              # [B,1,d_model]
            logit_t = self.predict(dec_out_t.squeeze(1))                 # [B,V]
            logits_list.append(logit_t.unsqueeze(1))                     # [B,1,V]

            if mode == "gumbel":
                # Straight-Through Gumbel-Softmax (미분 가능)
                y = F.gumbel_softmax(logit_t, tau=tau, hard=True)        # [B,V]
                emb_next = y @ emb_weight                                # [B,d_model]
                emb_t = emb_next.unsqueeze(1)                            # [B,1,d_model]
            elif mode == "greedy":
                with torch.no_grad():
                    idx = logit_t.argmax(-1)                             # [B]
                emb_t = self.smiles_embbed(idx).detach().unsqueeze(1)    # [B,1,d_model]
            else:
                raise ValueError("mode must be 'gumbel' or 'greedy'.")

        logits = torch.cat(logits_list, dim=1)                           # [B, L_out, V]

        # ── 3) property heads (기존 정의 그대로) ───────────────────
        tgt_mu = self.to_prop(mu)        # [B, 3]
        tgt_z  = self.to_prop_z(z)       # [B, 3]

        return logits, tgt_mu, mu, logv, tgt_z


class PriorNet(nn.Module):
    """
    Simple Prior Network that maps condition y to prior distribution parameters (mu_p, logvar_p).

    Args:
        y_dim (int): Dimensionality of condition vector y.
        latent_dim (int): Dimensionality of latent space.
        hidden_dim (int): Hidden size for MLP.
    """
    def __init__(self, y_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.len = dataset.max_len+3
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            weight_norm(nn.Linear(y_dim, hidden_dim)),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            weight_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.GELU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        nn.init.constant_(self.fc_logvar.bias, -3.0)

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mu_p and logvar_p given condition y.

        Args:
            y: Tensor of shape [batch_size, y_dim]

        Returns:
            mu_p: Tensor of shape [batch_size, latent_dim]
            logvar_p: Tensor of same shape
        """
        h = self.mlp(y)
        mu = self.fc_mu(h)
        lv = self.fc_logvar(h)
        lv= torch.log1p(torch.exp(lv))
        lv = torch.clamp(lv, 1e-4, 5.0)
        return mu, lv.log()