from utils.Trans_util import *
from torch.nn.utils.parametrizations import weight_norm
class CVAE(nn.Module):
    def __init__(self, d_model=256, latent_dim = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_len = dataset.max_len+3
        self.to_means = nn.Linear(d_model*self.max_len, latent_dim*self.max_len )
        self.to_var = nn.Linear(d_model*self.max_len, latent_dim*self.max_len )
        

        self.encoder = TFEncoder()
        self.decoder = TFDecoder(latent_dim=latent_dim)
        self.to_prop = nn.Linear(self.max_len *latent_dim, 3)
        self.to_prop_z = nn.Linear(self.max_len *latent_dim, 3)

        self.crossattn = MultiHeadAttention(d_model=latent_dim)

        self.predict = nn.Linear(d_model, dataset.vocab_size)

        self.input_embedding_p = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.GELU()
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
        self.norm1 = nn.LayerNorm(latent_dim)


        self.pos_enc = PositionalEncoding(latent_dim, dropout=0.2, max_len=5000)
        nn.init.constant_(self.to_var.bias, -3.0)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, smiles_enc, smiles_dec_input, properties):
        B = smiles_enc.size(0)
        properties_e = self.input_embedding(properties)
        properties_p = self.input_embedding_p(properties)

        encoded = self.encoder(smiles_enc, properties_e)

        means = self.to_means(encoded.view(B, -1))
        log_var = self.to_var(encoded.view(B, -1)).clamp_(max=-1)

        z = self.reparameterize(means, log_var)

        properties_p = self.pos_enc(properties_p)
        self.properties = properties_p

        z_z = self.crossattn(z.view(B, self.max_len , self.latent_dim), properties_p, properties_p)
        z_z = self.norm1(self.ff(z_z))


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
    def __init__(self, y_dim: int, latent_dim: int, hidden_dim: int = 256*(dataset.max_len+3)):
        super().__init__()
        self.max_len = dataset.max_len+3
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
        self.fc_mu = nn.Linear(hidden_dim, latent_dim*self.max_len)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim*self.max_len)
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