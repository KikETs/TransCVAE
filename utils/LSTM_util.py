import math
import torch.nn as nn
#from torch_pca import PCA
from torch.nn import LSTM
from utils.utils import *

class LSTM_Encoder(nn.Module):
    def __init__(self, d_model = 256, latent_dim = 64, hidden_dim=128):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        self.smiles_embbed = nn.Embedding(dataset.vocab_size, d_model)
        self.encoder = LSTM(input_size=d_model, hidden_size=d_model, num_layers=2, batch_first=True, dropout=0.2)
    def forward(self, smiles, properties):
        smiles = self.smiles_embbed(smiles)
        input = torch.cat((smiles, properties), dim=1)
        encoded = self.encoder(input)[0]
        
        return encoded

class LSTM_Decoder(nn.Module):
    def __init__(self, d_model = 256, latent_dim = 64, hidden_dim=128):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.decoder = LSTM(input_size=d_model, hidden_size=d_model, num_layers=2, batch_first=True, dropout=0.2)
    def forward(self, latent):
        decoded = self.decoder(latent)[0]
        
        return decoded
