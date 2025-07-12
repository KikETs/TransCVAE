import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import math
from psmiles import PolymerSmiles as PS
from torch.utils.data import DataLoader, Dataset, random_split
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
from scipy.sparse import csr_matrix, lil_matrix
import atomInSmiles
from collections import Counter
from IPython.display import clear_output, display, HTML
import ipywidgets as widgets
from tqdm.notebook import tqdm
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter, LogLocator, LogFormatterMathtext
from typing import List, Dict, Tuple
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device=torch.device('cpu')
print(device)
# os.chdir("C:\\Users\\user\\test.venv")
# print(os.getcwd())
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import umap.umap_ as umap

log_transformer = FunctionTransformer(np.log1p, validate=True)
log_minmax_pipeline = Pipeline(steps=[
    ('log', log_transformer),
    ('minmax', MinMaxScaler())
])
display(HTML('''
<style>
div.widget-html-output {
    background-color: black !important;
    color: white !important;
}
div.progress-bar {
    background-color: white !important;
}
</style>
'''))

mm_scaler = MinMaxScaler()
class load_data(Dataset):
    def __init__(self, path):
        #csv 읽기
        self.raw = pd.read_csv(path)

        #SMILES
        self.SMILES = self.raw.iloc[:, 1:2].values
        self.SMILES = np.squeeze(self.SMILES)

        #Li-ion Conductivity
        self.conductivity = self.raw.iloc[:, 6:7].values
        self.conductivity = mm_scaler.fit_transform(self.conductivity.reshape(-1, 1))
        self.conductivity = log_minmax_pipeline.fit_transform(self.conductivity.reshape(-1, 1))

        #Degree of Polymerization
        self.DP = self.raw.iloc[:, 4:5].values
        self.DP = mm_scaler.fit_transform(self.DP.reshape(-1, 1))

        #Transfer Number
        self.tn = self.raw.iloc[:, 10:11].values
        self.tn = mm_scaler.fit_transform(self.tn.reshape(-1, 1))

        self.properties = np.concat((self.conductivity, self.DP, self.tn), axis=-1)
        mean_vec = self.properties.mean(axis=0)
        std_vec  = self.properties.std(axis=0)
        self.properties = (self.properties - mean_vec) / std_vec  # 정규화된 3차원 조건 벡터

        print(self.properties.shape)

        print(max(self.properties[:,0]))
        print(min(self.properties[:,0]))
        print(self.properties[:,0].mean())

        #PSMILES 변환
        psmiles = []
        for smiles in self.SMILES:
            ps = PS(smiles)
            ps.canonicalize
            psmiles.append(ps.psmiles)

        #Atom-In-SMILES Encoding
        ais_encoding = []
        for smiles in psmiles:
            ais_encoding.append(atomInSmiles.encode(smiles))
        

        #Atom-In-SMILES Tokenization (Encoder)
        ais_tokens = []
        for smiles in ais_encoding:
            ais_tokens.append(atomInSmiles.smiles_tokenizer("[SOS] " + smiles +" [EOS]"))

        max_len = len(max(ais_encoding, key=len)) + 1
        self.max_len = max_len
        print("max sequence length : ", max_len)

        #vocab 구성
        corpus = []
        for frags in ais_tokens:
            corpus.extend(frags)
        corpus.append("[PAD]")
        token_count = Counter(corpus)
        vocab = { token:i for i, (token, count) in enumerate(sorted(token_count.items(), key=lambda x: x[1], reverse=True))}
        vocab_size = len(vocab)
        
        self.vocab = vocab

        num_data = len(ais_tokens)
        print(vocab)

        ais_tokens_enc = ais_tokens
        ais_tokens_enc = [[tok for tok in tokens if tok not in ['[SOS]', '[EOS]']] for tokens in ais_tokens_enc]

        ais_tokens_dec_input = ais_tokens
        ais_tokens_dec_input = [[tok for tok in tokens if tok not in ['[EOS]']] for tokens in ais_tokens_dec_input]

        ais_tokens_dec_output = ais_tokens
        ais_tokens_dec_output = [[tok for tok in tokens if tok not in ['[SOS]']] for tokens in ais_tokens_dec_output]

        #Tokens to number (encoder)
        ais_token_num_enc = torch.full((num_data, max_len),166, dtype=torch.long)
        i=0
        for tokens in ais_tokens_enc:
            for length in range((len(tokens))):
                ais_token_num_enc[i, length] = vocab[tokens[length]]
            i += 1

        #Tokens to number (Decoder Input)
        ais_token_num_dec_input = torch.full((num_data, max_len),166, dtype=torch.long)
        i=0
        for tokens in ais_tokens_dec_input:
            for length in range((len(tokens))):
                ais_token_num_dec_input[i, length] = vocab[tokens[length]]
            i += 1

        #Tokens to number (Decoder Output)
        ais_token_num_dec_output = torch.full((num_data, max_len),166, dtype=torch.long)
        i=0
        for tokens in ais_tokens_dec_output:
            for length in range((len(tokens))):
                ais_token_num_dec_output[i, length] = vocab[tokens[length]]
            i += 1

        dec_input_temp = torch.ones((6270, 268),dtype=torch.long) + 165
        dec_input_temp[:,:265]=ais_token_num_dec_input

        dec_output_temp = torch.ones((6270, 268),dtype=torch.long) + 165
        dec_output_temp[:,:265]=ais_token_num_dec_output


        self.SMILES_enc = ais_token_num_enc
        self.SMILES_dec_input = dec_input_temp
        self.SMILES_dec_output = dec_output_temp

        vocab_size, num_data
        print("vocab size : ", vocab_size,"\nnumber of data : ",num_data)
        self.num_data = num_data
        
        self.vocab = vocab
        self.vocab_size = vocab_size


        print(self.SMILES_enc.shape)
        self.properties = torch.tensor(self.properties, dtype=torch.float).unsqueeze(-1)
        print(self.properties.shape)

        self.test_data = self.SMILES_enc[50]

        print("PSMILES : ", psmiles[50])
        print("After AIS encoding : ", ais_encoding[50])
        print("After AIS Tokenization : ", ais_tokens_enc[50])
        print("After to number : ", ais_token_num_enc[50])
        print("Properties : ",self.properties)
        print(len(ais_token_num_enc[50]))
        
    
    def __getitem__(self, i):
        return self.SMILES_enc[i], self.SMILES_dec_input[i], self.SMILES_dec_output[i], self.properties[i]
    
    def __len__(self):
        return self.SMILES_enc.shape[0]
    
    def vocab_len(self):
        return self.vocab_size
    

    import math
#from torch_pca import PCA
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pos_embedding =  self.pe[:, :x.size(1), :]
        pos_embedding = torch.repeat_interleave(pos_embedding, x.shape[0], dim=0)
        x =  torch.cat([x, pos_embedding], dim=2)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, nhead=4):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, query, key, value, attn_mask=None):
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attn_mask)
        return attn_output
    

class TFEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=4, d_ff=64, enc_seq_len=5000, dropout=0.2):
        super().__init__()
        self.normLayer = LayerNorm(d_model=d_model)
        
        self.encoderLayer = TransformerEncoderLayer(batch_first=True,
                                               d_model=d_model,
                                               nhead=n_heads,
                                               dim_feedforward=d_ff,
                                               dropout=dropout,
                                               activation="gelu")
        self.encoder = TransformerEncoder(encoder_layer=self.encoderLayer, num_layers=2,
                                          norm=self.normLayer)
        self.input_embedding_smiles = nn.Embedding(167, d_model // 2)
        self.pos_encoding = PositionalEncoding(d_model // 2, dropout, max_len=enc_seq_len)


    def forward(self, smiles_enc, properties):
        smiles_enc = self.input_embedding_smiles(smiles_enc)
        cat = torch.cat((smiles_enc, properties), dim=1)
        enc_input_0 = self.pos_encoding(cat)

        encoded_0 = self.encoder(enc_input_0)

        return encoded_0

class TFDecoder(nn.Module):
    def __init__(self, d_model=256, n_heads=4, d_ff=64, enc_seq_len=5000, dropout=0.4):
        super().__init__()
        self.normLayer = LayerNorm(d_model=d_model)
        self.decoderLayer = TransformerDecoderLayer(batch_first=True,
                                               d_model=d_model,
                                               nhead=n_heads,
                                               dim_feedforward=d_ff,
                                               dropout=dropout,
                                               activation="gelu")
        self.decoder = TransformerDecoder(decoder_layer=self.decoderLayer,num_layers=2,
                                          norm=self.normLayer)

        self.input_embedding_smiles = nn.Embedding(167, d_model // 2)
        self.pos_encoding = PositionalEncoding(d_model // 2, dropout, max_len=enc_seq_len)
        self.to_memory = nn.Linear(64, d_model)


    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, dec_input, latent):
        dec_input = self.input_embedding_smiles(dec_input)
        dec_input_0 = self.pos_encoding(dec_input)
        latent = self.to_memory(latent)
        x_mask = self.generate_square_subsequent_mask(dec_input.shape[1]).to(device)
        memory_mask = self.generate_square_subsequent_mask(latent.shape[1]).to(device)

        decoded = self.decoder(dec_input_0, latent, tgt_mask=x_mask, memory_mask = memory_mask)
        return decoded
    

class CVAE(nn.Module):
    def __init__(self, d_model=256, latent_dim = 64):
        super().__init__()
        self.to_means = nn.Linear(d_model, latent_dim)
        self.to_var = nn.Linear(d_model, latent_dim)

        self.encoder = TFEncoder()
        self.decoder = TFDecoder()
        self.to_prop = nn.Linear(268*64, 3)
        self.to_prop_z = nn.Linear(268*64, 3)

        self.crossattn = MultiHeadAttention()

        self.predict = nn.Linear(d_model, 167)

        self.input_embedding_p = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 8),
        )

        self.input_embedding = nn.Sequential(
            nn.Linear(1, d_model // 8),
            nn.GELU(),
            nn.Linear(d_model // 8, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 2),
        )
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, latent_dim)
        )

        self.pos_enc = PositionalEncoding(d_model // 8, dropout=0.2, max_len=5000)
        nn.init.constant_(self.to_var.bias, -3.0)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, smiles_enc, smiles_dec_input, properties):
        properties_e = self.input_embedding(properties)
        properties_p = self.input_embedding_p(properties)

        encoded = self.encoder(smiles_enc, properties_e) # (batch_size, seq_len, d_model // 8) 

        means = self.to_means(encoded)
        log_var = self.to_var(encoded.detach())
        log_var = torch.clamp(log_var, min=-6., max=-2.2)

        z = self.reparameterize(means, log_var)

        properties_p = self.pos_enc(properties_p)
        self.properties = properties_p

        z_z = self.crossattn(z, properties_p, properties_p)
        z_z = self.ff(z_z)

        tgt = self.to_prop(means.view(-1, 268*64))
        tgt_z = self.to_prop_z(z.view(-1, 268*64))
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
    def __init__(self, y_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
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
        y = y.repeat(1, 90, 1)[:, :268, :]  # [B, L, y_dim]
        h = self.mlp(y)
        mu_p = self.fc_mu(h)
        logvar_p = self.fc_logvar(h.detach())
        logvar_p = torch.clamp(logvar_p, min=-6, max=-2.2)
        return mu_p, logvar_p


class ConditionalVAELoss(nn.Module):
    def __init__(
        self,
        vocab_size      : int,
        max_beta        : float = 1.0,     # β-VAE 상한
        anneal_steps    : int   = 1000,    # β 스케줄 길이
        free_bits       : float = 0.02,    # per-dim nats
        capacity_max    : float = 0.0,     # 0이면 β-VAE, >0이면 Burgess-C
        capacity_inc    : float = 0.002,   # C(t) 증가속도 (nats/step)
        gamma           : float = 5.0,     # Burgess 벌점 계수
        prop_w          : float = 1.0      # property loss 가중
    ):
        super().__init__()
        self.V  = vocab_size
        self.fb = free_bits
        self.max_beta = max_beta
        self.anneal   = anneal_steps
        self.C_max    = capacity_max
        self.C_inc    = capacity_inc
        self.gamma    = gamma
        self.prop_w   = prop_w

    # --------------------------------------------------------------
    def forward(
        self, logits, target_tokens,
        mu_q, lv_q, mu_p, lv_p,
        prop_pred_mu, true_prop, prop_pred_z,
        step:int
    ):
        B, L, _ = logits.size()
        D       = mu_q.size(-1)

        # 1) Reconstruction
        recon = F.cross_entropy(
            logits.view(-1, self.V),           # (B·L, V)
            target_tokens.view(-1),            # (B·L,)
            reduction='sum'
        ) / B

        # 2) KL(q‖p)   -------------------------------------------------
        q = Normal(mu_q, torch.exp(0.5 * lv_q))
        p = Normal(mu_p, torch.exp(0.5 * lv_p))
        kld_dim = kl_divergence(q, p)               # (B, L, D)

        # 2-a) free-bits (per-dim clamp)
        if self.fb > 0.0:
            kld_dim = torch.clamp(kld_dim, min=self.fb)

        kld_sample = kld_dim.sum(-1).sum(-1).mean()   # scalar – batch 평균
        kld_token = kld_dim.sum(-1)
        kld_per_token = kld_token.mean()
        kld_raw    = kld_dim.mean()                   # 모니터링용

        # 2-b) KL term
        beta = min(self.max_beta, self.max_beta * step / self.anneal)
        kl_term = beta * kld_per_token

        # if self.C_max > 0:          # Burgess Capacity-VAE 모드
        #     C_t     = min(self.C_max, self.C_inc * step)
        #     kl_term = self.gamma * F.relu(kld_sample - C_t)
        # else:                       # 순수 β-VAE
        #     kl_term = beta * kld_sample

        # 3) Property losses ------------------------------------------
        prop_loss_mu = F.mse_loss(prop_pred_mu, true_prop)
        prop_loss_z  = F.mse_loss(prop_pred_z , true_prop)

        

        # 4) 최종 손실 -------------------------------------------------
        loss = (
            recon
            + kl_term
            + self.prop_w * (4 * prop_loss_mu + 0.5 * prop_loss_z)
            + 1.0* ((kld_dim - kld_dim.mean())**2).mean() # imbalnce
            + 0.4 * torch.exp(lv_q).mean() # sigma_penalty
        )

        return loss, recon, kld_sample, kld_raw, prop_loss_mu
