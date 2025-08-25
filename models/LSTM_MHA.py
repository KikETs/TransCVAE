from utils.LSTM_util import *
from utils.Trans_util import MultiHeadAttention
from utils.dataloader import dataset
from torch.nn.utils.parametrizations import weight_norm

class CVAE(nn.Module):
    def __init__(self, d_model=256, latent_dim = 64, hidden_dim = 128):
        super().__init__()
        self.len = dataset.max_len+3
        self.latent_dim = latent_dim
        mid=(d_model+latent_dim)//2
        self.to_means = nn.Sequential(
            nn.Linear(d_model, mid),
            nn.Dropout(0.1),
            nn.Linear(mid, latent_dim)
        )
        self.to_var = nn.Linear(d_model, latent_dim)
        self.to_decoder = nn.Linear(latent_dim, d_model*2)

        self.encoder = LSTM(input_size=d_model, hidden_size=d_model, num_layers=2, batch_first=True, dropout=0.2)
        self.decoder = LSTM(input_size=d_model, hidden_size=d_model, num_layers=2, batch_first=True, dropout=0.2)
        self.to_prop = nn.Linear(self.len *latent_dim, 3)
        self.to_prop_z = nn.Linear(self.len *latent_dim, 3)

        self.predict = nn.Linear(d_model, dataset.vocab_size)

        self.input_embedding = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 4, d_model // 2),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.GELU()
        )
        self.smiles_embbed = nn.Embedding(dataset.vocab_size, d_model, padding_idx=dataset.vocab['[PAD]'])
        self.norm1 = nn.LayerNorm(latent_dim)
        self.crossattn = MultiHeadAttention(d_model=latent_dim)
        self.alpha = nn.Parameter(torch.ones(1))
        self.input_embedding_p = nn.Sequential(
            nn.Linear(1, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, latent_dim),
        )
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim//2),
            nn.GELU(),
            nn.Linear(latent_dim//2, latent_dim)
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var).to(device)
        eps = torch.randn_like(std).to(device)
        return mu + eps * std

    def forward(
        self,
        smiles_enc: torch.Tensor,   # [B,L_enc]  (<cond>+SMILES)
        smiles_tgt: torch.Tensor,   # [B,L_out]  (타깃 길이: 손실 정렬용)
        properties: torch.Tensor,   # [B,3,1]
        mode: str = "gumbel",       # "gumbel" | "greedy"
        tau: float = 1.0            # gumbel-softmax 온도
    ):
        """
        GT 분기 없이 오직 Pred(자기회귀)만 사용.
        mode="gumbel": Straight-Through Gumbel-Softmax로 미분 가능(권장, 학습용)
        mode="greedy": argmax 샘플링(추론용, 비미분)
        """
        device = smiles_enc.device
        B, L_out = smiles_tgt.shape
        SOS = dataset.vocab['[SOS]']

        # ─── 1) Encoder ───────────────────────────────────────────────
        emb_enc  = self.smiles_embbed(smiles_enc)               # [B,L_enc,E]
        prop_e   = self.input_embedding(properties)             # [B,3,E]
        enc_in   = torch.cat((emb_enc, prop_e), dim=1)          # [B,L_enc+3,E]
        enc_out  = self.encoder(enc_in)[0]                      # [B,L_enc+3,E]

        mu, lv   = self.to_means(enc_out), self.to_var(enc_out) # [B,L_enc+3,z]
        z_sample = self.reparameterize(mu, lv)                  # [B,L_enc+3,z]

        # ─── 2) Cross-Attention + FFN on z ───────────────────────────
        prop_p   = self.input_embedding_p(properties)           # [B,1,z]
        z_z      = self.alpha * self.crossattn.forward(z_sample, prop_p, prop_p) + z_sample
        z_ff     = self.norm1(self.ff(z_z) + z_z)

        # ─── 3) 초기 (h0,c0) 생성 ────────────────────────────────────
        z_mean   = z_ff.mean(1)                                 # [B,z]
        h0, c0   = torch.tanh(self.to_decoder(z_mean)).chunk(2, dim=-1)
        h0, c0   = h0.unsqueeze(0).repeat(2,1,1), c0.unsqueeze(0).repeat(2,1,1)

        # ─── 4) 자기회귀 디코딩(오직 Pred) ──────────────────────────
        # 시작 토큰: SOS
        x_t = torch.full((B, 1), SOS, dtype=torch.long, device=device)   # [B,1]
        emb_t = self.smiles_embbed(x_t)                                  # [B,1,E]

        logits_list = []
        h, c = h0, c0

        # Embedding weight (Gumbel-ST용)
        emb_weight = self.smiles_embbed.weight                           # [V,E]

        for t in range(L_out):
            dec_out_t, (h, c) = self.decoder(emb_t, (h, c))              # [B,1,E]
            logit_t = self.predict(dec_out_t.squeeze(1))                 # [B,V]
            logits_list.append(logit_t.unsqueeze(1))                     # 수집

            if mode == "gumbel":
                # Straight-Through Gumbel-Softmax (미분 가능)
                y = F.gumbel_softmax(logit_t, tau=tau, hard=True)        # [B,V]
                emb_next = torch.matmul(y, emb_weight)                   # [B,E]
                emb_t = emb_next.unsqueeze(1)                            # [B,1,E]
            elif mode == "greedy":
                with torch.no_grad():
                    idx = logit_t.argmax(-1)                             # [B]
                emb_t = self.smiles_embbed(idx).detach().unsqueeze(1)    # [B,1,E]
            else:
                raise ValueError("mode must be 'gumbel' or 'greedy'.")

        logits = torch.cat(logits_list, dim=1)                           # [B,L_out,V]

        # ─── 5) property-heads ───────────────────────────────────────
        B_, L_encp, Z = mu.shape
        tgt_mu = self.to_prop(mu.reshape(B_, L_encp * Z))
        tgt_z  = self.to_prop_z(z_sample.reshape(B_, L_encp * Z))

        return logits, tgt_mu, mu, lv, tgt_z

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
            weight_norm(nn.Linear(hidden_dim, hidden_dim*self.len)),
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
        mu = self.fc_mu(h.view(-1, self.len, self.hidden_dim))
        lv = self.fc_logvar(h.view(-1, self.len, self.hidden_dim))
        lv= torch.log1p(torch.exp(lv))
        lv = torch.clamp(lv, 1e-4, 5.0)
        return mu, lv.log()