import sys
import random
from pathlib import Path
PROJECT_ROOT = Path.home()/"바탕화면"/"torch"/"Chem"
sys.path.insert(0, str(PROJECT_ROOT))
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
        self.smiles_embbed = nn.Embedding(dataset.vocab_size, d_model)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var).to(device)
        eps = torch.rand_like(std).to(device)
        return mu + eps * std

    def forward(self, smiles_enc, smiles_tgt, properties, tf_ratio: float = 1.0):
        """
        smiles_enc : [B, L]   - encoder 입력  (<cond> + SMILES)
        smiles_tgt : [B, L]   - teacher-forcing target  (<SOS>+SMILES[:-1])
        properties : [B, 3]
        tf_ratio   : 0.0~1.0  - teacher-forcing 비율
        """
        # -------------------------------------------------------
        # 2) Encoder  (μ, σ 계산용 마지막 hidden-state)
        # -------------------------------------------------------
        prop_emb  = self.input_embedding(properties)          # [B, d]
        emb_src   = self.smiles_embbed(smiles_enc)            # [B,L,d]
        enc_in    = torch.cat([emb_src,prop_emb], dim=1)  # [B,L+3,d]
        h_enc     = self.encoder(enc_in)[1][0][-1]            # [B,d_model]  (h_N)

        mu, logv  = self.to_means(h_enc), self.to_var(h_enc)  # [B,z]
        z         = self.reparameterize(mu, logv)             # [B,z]

        # -------------------------------------------------------
        # 3) z → 초기 hidden (h0,c0)
        # -------------------------------------------------------
        h0c0      = torch.tanh(self.to_decoder(z)).chunk(2, dim=-1)
        h0, c0    = h0c0[0].unsqueeze(0).repeat(2,1,1), h0c0[1].unsqueeze(0).repeat(2,1,1)

        # -------------------------------------------------------
        # 4) teacher-forcing mix
        # -------------------------------------------------------
        emb_gt    = self.smiles_embbed(smiles_tgt)            # ground-truth embedding
        if tf_ratio < 1.0:                                    # scheduled sampling
            with torch.no_grad():
                logits_tf, _ = self.decoder(emb_gt, (h0,c0))
            emb_pred =logits_tf.argmax(-1)   # model tokens
            emb_pred = emb_pred.clamp(max=dataset.vocab_size-1)
            emb_pred = self.smiles_embbed(emb_pred)
            take_pred= (torch.rand_like(smiles_tgt.float()) > tf_ratio).unsqueeze(-1)
            emb_in   = torch.where(take_pred, emb_pred, emb_gt)   # GT ↔ Pred 혼합
        else:
            emb_in   = emb_gt

        # -------------------------------------------------------
        # 5) Decoder   (진짜 forward)
        # -------------------------------------------------------
        out, _    = self.decoder(emb_in, (h0, c0))
        logits    = self.predict(out)                         # [B,L,V]

        # -------------------------------------------------------
        # 6) property-heads 그대로 유지
        # -------------------------------------------------------
        tgt_mu    = self.to_prop(mu)                         # [B,3]
        tgt_z     = self.to_prop_z(z)                         # [B,3]

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