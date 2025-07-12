from utils.Trans_util import *
from torch.nn.utils.parametrizations import weight_norm
class CVAE(nn.Module):
    def __init__(self, d_model=256, latent_dim = 64):
        super().__init__()
        self.to_means = nn.Linear(d_model*42, latent_dim*42)
        self.to_var = nn.Linear(d_model*42, latent_dim*42)

        self.encoder = TFEncoder()
        self.decoder = TFDecoder()
        self.to_prop = nn.Linear(42*64, 3)
        self.to_prop_z = nn.Linear(42*64, 3)

        self.crossattn = MultiHeadAttention()

        self.predict = nn.Linear(d_model, 167)

        self.input_embedding_p = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4),
        )

        self.input_embedding = nn.Sequential(
            nn.Linear(1, d_model // 8),
            nn.GELU(),
            nn.Linear(d_model // 8, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, latent_dim)
        )


        self.pos_enc = PositionalEncoding(d_model // 4, dropout=0.2, max_len=5000)
        nn.init.constant_(self.to_var.bias, -3.0)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, smiles_enc, smiles_dec_input, properties):
        B = smiles_enc.size(0)
        properties_e = self.input_embedding(properties)
        properties_p = self.input_embedding_p(properties)

        encoded = self.encoder(smiles_enc, properties_e) # (batch_size, seq_len, d_model // 8) 

        means = self.to_means(encoded.view(B, -1))
        log_var = self.to_var(encoded.view(B, -1)).clamp_(-8, -0.5)

        z = self.reparameterize(means, log_var)

        properties_p = self.pos_enc(properties_p)
        self.properties = properties_p

        z_z = self.crossattn(z.view(B, 42, 64), properties_p, properties_p)
        z_z = self.ff(z_z)

        tgt = self.to_prop(means)
        tgt_z = self.to_prop_z(z)
        output = self.decoder(smiles_dec_input, z_z)
        

        output = self.predict(output)



        return output, tgt, means, log_var, tgt_z
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