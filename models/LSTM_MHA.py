from utils.LSTM_util import *
from utils.Trans_util import MultiHeadAttention
from torch.nn.utils.parametrizations import weight_norm

class CVAE(nn.Module):
    def __init__(self, d_model=256, latent_dim = 64, hidden_dim = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.to_means = nn.Linear(d_model*42, latent_dim * 42)
        self.to_var = nn.Linear(d_model*42, latent_dim * 42)
        self.to_decoder = nn.Linear(latent_dim, d_model*2)

        self.encoder = LSTM(input_size=d_model, hidden_size=d_model, num_layers=2, batch_first=True, dropout=0.2)
        self.decoder = LSTM(input_size=d_model, hidden_size=d_model, num_layers=2, batch_first=True, dropout=0.2)
        self.pdecoder = nn.Linear(42*64, 3)
        self.to_prop_z = nn.Linear(42*64, 3)

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
        self.smiles_embbed = nn.Embedding(dataset.vocab_size, d_model)

        self.crossattn = MultiHeadAttention(d_model=latent_dim)

        self.input_embedding_p = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4),
        )
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, latent_dim)
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var).to(device)
        eps = torch.rand_like(std).to(device)
        return mu + eps * std

    def forward(
        self,
        smiles_enc: torch.Tensor,   # [B,L]   (<cond>+SMILES)
        smiles_tgt: torch.Tensor,   # [B,L]   (<SOS>+SMILES[:-1])
        properties: torch.Tensor,   # [B,3,1]
        tf_ratio: float = 1.0       # 1.0=GT  0.0=self-sampling
    ):
        B, L = smiles_tgt.shape

        # ─── 1. Encoder ────────────────────────────────────────────
        emb_enc  = self.smiles_embbed(smiles_enc)                 # [B,L,E]
        prop_e   = self.input_embedding(properties)               # [B,3,E]
        enc_in   = torch.cat((emb_enc, prop_e), dim=1)           # [B,L+3,E]
        enc_out  = self.encoder(enc_in)[0]                        # [B,L+3,E]

        mu, lv   = self.to_means(enc_out.reshape(B, -1)), self.to_var(enc_out.reshape(B, -1))   # [B,L,z]
        z_sample = self.reparameterize(mu, lv)                    # [B,L,z]

        # ─── 2. Cross-Attention + FFN on z ─────────────────────────
        prop_p   = self.input_embedding_p(properties)             # [B,1,E′]
        z_attn   = self.crossattn(z_sample.view(-1, 42, 64), prop_p, prop_p)       # [B,L,z]
        z_ff     = self.ff(z_attn)                                # [B,L,z]

        # ─── 3. 초기 (h0,c0)  — z_mean → 2*E  ─────────────────────
        z_mean   = z_ff.mean(1)                                   # [B,z]
        h0c0     = torch.tanh(self.to_decoder(z_mean)).chunk(2, dim=-1)
        h0, c0   = h0c0[0].unsqueeze(0).repeat(2,1,1), h0c0[1].unsqueeze(0).repeat(2,1,1)

        # ─── 4. Teacher-forcing mix  (GT vs Pred) ─────────────────
        emb_gt   = self.smiles_embbed(smiles_tgt)                 # [B,L,E]
        if tf_ratio < 1.0:
            with torch.no_grad():
                logits_tf, _ = self.decoder(emb_gt, (h0,c0))      # 예비 pass
            tok_pred  = logits_tf.argmax(-1)                      # [B,L]
            tok_pred = tok_pred.clamp(max=dataset.vocab_size-1)
            emb_pred  = self.smiles_embbed(tok_pred)              # [B,L,E]

            mask      = (torch.rand_like(smiles_tgt.float()) > tf_ratio).unsqueeze(-1)
            emb_in    = torch.where(mask, emb_pred, emb_gt)       # 혼합 임베딩
        else:
            emb_in    = emb_gt

        # ─── 5. Decoder (진짜 forward) ────────────────────────────
        dec_out, _ = self.decoder(emb_in, (h0, c0))               # [B,L,E]
        logits     = self.predict(dec_out)                        # [B,L,V]

        # ─── 6. property-heads (그대로) ───────────────────────────
        tgt_mu = self.pdecoder(mu.reshape(mu.size(0), -1))        # [B,3]
        tgt_z  = self.to_prop_z(z_sample.reshape(z_sample.size(0), -1))

        return logits, tgt_mu, mu, lv, tgt_z

class PriorNet(nn.Module):
    """
    Simple Prior Network that maps condition y to prior distribution parameters (mu_p, logvar_p).

    Args:
        y_dim (int): Dimensionality of condition vector y.
        latent_dim (int): Dimensionality of latent space.
        hidden_dim (int): Hidden size for MLP.
    """
    def __init__(self, y_dim: int, latent_dim: int, hidden_dim: int = 256*42):
        super().__init__()
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
        self.fc_mu = nn.Linear(hidden_dim, latent_dim*42)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim*42)
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
        lv= torch.log1p(torch.exp(lv))-3.0
        return mu, lv