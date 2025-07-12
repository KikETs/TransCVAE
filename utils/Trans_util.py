import math
#from torch_pca import PCA
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from utils.utils import *
d_model=256
smi_emb = nn.Embedding(167, d_model)
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=7000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)                # (1, max_len, d_model)  ← batch_first
        self.register_buffer("pe", pe)

    def forward(self, x):                   # x: (B, L, d_model)
        L = x.size(1)
        # pe도 같은 device에 두는 것이 안전
        x = x + self.pe[:, :L, :].to(x.device)
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
        self.smi_embed = smi_emb
        self.encoderLayer = TransformerEncoderLayer(batch_first=True,
                                               d_model=d_model,
                                               nhead=n_heads,
                                               dim_feedforward=d_ff,
                                               dropout=dropout,
                                               activation="gelu")
        self.encoder = TransformerEncoder(encoder_layer=self.encoderLayer, num_layers=2,
                                          norm=self.normLayer)
        self.pe = PositionalEncoding(d_model, dropout=0.2, max_len=5000)
    def forward(self, smi, prop):
        smi = self.smi_embed(smi)
        enc_in = self.pe(torch.cat((smi, prop),dim=1))

        encoded_0 = self.encoder(enc_in)

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
        
        self.to_memory = nn.Linear(64, d_model)
        self.smi_embed = smi_emb


    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, dec_input, latent):
        latent = self.to_memory(latent)
        dec_input = self.smi_embed(dec_input)
        x_mask = self.generate_square_subsequent_mask(dec_input.shape[1]).to(device)

        decoded = self.decoder(dec_input, latent, tgt_mask=x_mask)
        return decoded