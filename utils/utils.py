import torch, os, sys, pickle, math, random
import numpy as np
import torch.nn as nn
import pandas as pd
from pathlib import Path
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
import matplotlib.pyplot as plt
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
from group_selfies.group_decoder import (
    _tokenize_selfies, Counter,
    selfies_to_graph_iterative, form_rings_bilocally_iterative
)
from group_selfies import(
    fragment_mols,
    Group,
    MolecularGraph,
    GroupGrammar,
    group_encoder
)
from group_selfies.utils.selfies_utils import split_selfies

ess = GroupGrammar.essential_set()

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
from rdkit.Chem import rdmolops
import selfies as sf   # polyselfies가 selfies를 패치해 설치해 줌

mm_scaler = MinMaxScaler()
class load_data(Dataset):
    def __init__(self, csv_path, cache_dir="cache"):
        csv_path = Path(csv_path)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)

        cache_pt   = cache_dir / (csv_path.stem + "Trans_data.pt")
        cache_meta = cache_dir / (csv_path.stem + "Trans_meta.pkl")

        # ---------- ① 캐시가 있으면 즉시 로드 ----------
        if cache_pt.exists() and cache_meta.exists():
            self._load_cache(cache_pt, cache_meta)
            return

        # ---------- ② 없으면 느린 전처리 수행 ----------
        self._build_from_csv(csv_path)

        # ---------- ③ 캐시에 저장해 두기 ----------
        torch.save(
            {
                "SMILES_enc": self.SMILES_enc,
                "SMILES_dec_input": self.SMILES_dec_input,
                "SMILES_dec_output": self.SMILES_dec_output,
                "properties": self.properties
            },
            cache_pt,
        )
        with open(cache_meta, "wb") as f:
            pickle.dump(
                {
                    "vocab": self.vocab,
                    "max_len": self.max_len,
                    "mean_vec": self.mean_vec,
                    "std_vec":  self.std_vec,
                },
                f,
            )

    # -------------------------------------------------
    def _load_cache(self, data_path, meta_path):
        blob = torch.load(data_path, map_location="cpu")
        self.SMILES_enc       = blob["SMILES_enc"]
        self.SMILES_dec_input = blob["SMILES_dec_input"]
        self.SMILES_dec_output= blob["SMILES_dec_output"]
        self.properties       = blob["properties"]

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.vocab    = meta["vocab"]
        self.max_len  = meta["max_len"]
        self.mean_vec = meta["mean_vec"]
        self.std_vec  = meta["std_vec"]
        self.vocab_size = len(self.vocab)
        self.num_data   = self.SMILES_enc.shape[0]
        print(self.vocab)
        print(self.vocab_size)
        print(self.SMILES_enc[0])
        print(self.SMILES_dec_input[0])
        print(self.SMILES_dec_output[0])
    def split_selfies(self, sf_str):
        return sf.split_selfies(sf_str)   # polyselfies가 내부 호출

    def _build_from_csv(self, path):
        #csv 읽기
        self.raw = pd.read_csv(path)

        #SMILES
        self.SMILES = self.raw.iloc[:, 1:2].values
        self.SMILES = np.squeeze(self.SMILES)

        #Li-ion Conductivity
        self.conductivity = self.raw.iloc[:, 6:7].values
        self.conductivity = log_minmax_pipeline.fit_transform(self.conductivity.reshape(-1, 1))

        #Degree of Polymerization
        self.DP = self.raw.iloc[:, 4:5].values
        self.DP = mm_scaler.fit_transform(self.DP.reshape(-1, 1))

        #Transfer Number
        self.tn = self.raw.iloc[:, 10:11].values
        self.tn = mm_scaler.fit_transform(self.tn.reshape(-1, 1))

        self.properties = np.concat((self.conductivity, self.DP, self.tn), axis=-1)
        self.mean_vec = self.properties.mean(axis=0)
        self.std_vec  = self.properties.std(axis=0)
        self.properties = (self.properties - self.mean_vec) / self.std_vec  # 정규화된 3차원 조건 벡터

        print(self.properties.shape)

        print(self.properties[1])
        print("조건 벡터 각 평균")
        print(self.properties[:,0].mean())
        print(self.properties[:,1].mean())
        print(self.properties[:,2].mean())

        #PSMILES 변환        
        psmiles = [PS(smiles).canonicalize.psmiles for smiles in self.SMILES]

        gpselfies = [ess.full_encoder(Chem.MolFromSmiles(s)) for s in psmiles]

        sf_tokens = [list(split_selfies(sf)) for sf in gpselfies]

        self.max_len = max(len(t) for t in sf_tokens) + 1

        # ③ vocab
        corpus = [tok for seq in sf_tokens for tok in seq] + ["[SOS]","[EOS]","[PAD]"]
        vocab  = {tok:i for i,tok in enumerate(sorted(set(corpus)))}
        vocab_size = len(vocab)
        
        self.vocab = vocab

        num_data = len(sf_tokens)
        print(vocab)

        enc   = torch.full((num_data, self.max_len), vocab["[PAD]"], dtype=torch.long)
        dec_in= torch.full_like(enc, vocab["[PAD]"])
        dec_out=torch.full_like(enc, vocab["[PAD]"])


        for i, seq in enumerate(sf_tokens):
            # Encoder = 순수 SELFIES
            for j, tok in enumerate(seq):
                enc[i,j] = vocab[tok]

            # Decoder input = [SOS] + SELFIES
            dec_in[i,0] = vocab["[SOS]"]
            dec_in[i,1:len(seq)+1] = torch.tensor([vocab[t] for t in seq])

            # Decoder output = SELFIES + [EOS]
            dec_out[i,:len(seq)] = torch.tensor([vocab[t] for t in seq])
            dec_out[i,len(seq)]  = vocab["[EOS]"]


        self.SMILES_enc = enc
        self.SMILES_dec_input = dec_in
        self.SMILES_dec_output = dec_out
        vocab_size, num_data
        print("vocab size : ", vocab_size,"\nnumber of data : ",num_data)
        self.num_data = num_data
        
        self.vocab = vocab
        self.vocab_size = vocab_size


        print(self.SMILES_enc.shape)
        self.properties = torch.tensor(self.properties, dtype=torch.float).unsqueeze(-1)
        print(self.properties.shape)

        self.test_data = self.SMILES_enc[50]

        print("PSMILES : ", gpselfies[50])
        print("After AIS encoding : ", enc[50])
        print("After AIS Tokenization : ", enc[50])
        print("After to number : ", enc[50])
        print("Properties : ",self.properties)
        print(len(enc[50]))
        
    
    def __getitem__(self, i):
        return self.SMILES_enc[i], self.SMILES_dec_input[i], self.SMILES_dec_output[i], self.properties[i]
    
    def __len__(self):
        return self.SMILES_enc.shape[0]
    
    def vocab_len(self):
        return self.vocab_size

Polymers = "/home/kiket/바탕화면/torch/Chem/data/simulation-trajectory-aggregate_aligned.csv"
dataset = load_data(Polymers)

generator = torch.Generator()
train_dataset, val_dataset = random_split(dataset, [0.75, 0.25], generator=generator)

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=False)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=False)


eval_dataloader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)


mm_scaler = MinMaxScaler()
class load_data_LSTM(Dataset):
    def __init__(self, csv_path, cache_dir="cache"):
        csv_path = Path(csv_path)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)

        cache_pt   = cache_dir / (csv_path.stem + "LSTM_data.pt")
        cache_meta = cache_dir / (csv_path.stem + "LSTM_meta.pkl")

        # ---------- ① 캐시가 있으면 즉시 로드 ----------
        if cache_pt.exists() and cache_meta.exists():
            self._load_cache(cache_pt, cache_meta)
            return

        # ---------- ② 없으면 느린 전처리 수행 ----------
        self._build_from_csv(csv_path)

        # ---------- ③ 캐시에 저장해 두기 ----------
        torch.save(
            {
                "SMILES_enc": self.SMILES_enc,
                "SMILES_dec_input": self.SMILES_dec_input,
                "SMILES_dec_output": self.SMILES_dec_output,
                "properties": self.properties
            },
            cache_pt,
        )
        with open(cache_meta, "wb") as f:
            pickle.dump(
                {
                    "vocab": self.vocab,
                    "max_len": self.max_len,
                    "mean_vec": self.mean_vec,
                    "std_vec":  self.std_vec,
                },
                f,
            )

    # -------------------------------------------------
    def _load_cache(self, data_path, meta_path):
        blob = torch.load(data_path, map_location="cpu")
        self.SMILES_enc       = blob["SMILES_enc"]
        self.SMILES_dec_input = blob["SMILES_dec_input"]
        self.SMILES_dec_output= blob["SMILES_dec_output"]
        self.properties       = blob["properties"]

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.vocab    = meta["vocab"]
        self.max_len  = meta["max_len"]
        self.mean_vec = meta["mean_vec"]
        self.std_vec  = meta["std_vec"]
        self.vocab_size = len(self.vocab)
        self.num_data   = self.SMILES_enc.shape[0]

    def _build_from_csv(self, path):
        #csv 읽기
        self.raw = pd.read_csv(path)

        #SMILES
        self.SMILES = self.raw.iloc[:, 1:2].values
        self.SMILES = np.squeeze(self.SMILES)

        #Li-ion Conductivity
        self.conductivity = self.raw.iloc[:, 6:7].values
        self.conductivity = log_minmax_pipeline.fit_transform(self.conductivity.reshape(-1, 1))

        #Degree of Polymerization
        self.DP = self.raw.iloc[:, 4:5].values
        self.DP = mm_scaler.fit_transform(self.DP.reshape(-1, 1))

        #Transfer Number
        self.tn = self.raw.iloc[:, 10:11].values
        self.tn = mm_scaler.fit_transform(self.tn.reshape(-1, 1))

        self.properties = np.concat((self.conductivity, self.DP, self.tn), axis=-1)
        self.mean_vec = self.properties.mean(axis=0)
        self.std_vec  = self.properties.std(axis=0)
        self.properties = (self.properties - self.mean_vec) / self.std_vec  # 정규화된 3차원 조건 벡터

        print(self.properties.shape)

        print(self.properties[1])
        print("조건 벡터 각 평균")
        print(self.properties[:,0].mean())
        print(self.properties[:,1].mean())
        print(self.properties[:,2].mean())

        #PSMILES 변환        
        psmiles = [PS(smiles).canonicalize.psmiles for smiles in self.SMILES]

        gpselfies = [ess.full_encoder(Chem.MolFromSmiles(s)) for s in psmiles]

        sf_tokens = [list(split_selfies(sf)) for sf in gpselfies]

        self.max_len = max(len(t) for t in sf_tokens) + 1

        # ③ vocab
        corpus = [tok for seq in sf_tokens for tok in seq] + ["[SOS]","[EOS]","[PAD]"]
        vocab  = {tok:i for i,tok in enumerate(sorted(set(corpus)))}
        vocab_size = len(vocab)
        
        self.vocab = vocab

        num_data = len(sf_tokens)
        print(vocab)

        enc   = torch.full((num_data, self.max_len), vocab["[PAD]"], dtype=torch.long)
        dec_in= torch.full_like(enc, vocab["[PAD]"])
        dec_out=torch.full_like(enc, vocab["[PAD]"])


        for i, seq in enumerate(sf_tokens):
            # Encoder = 순수 SELFIES
            for j, tok in enumerate(seq):
                enc[i,j] = vocab[tok]

            # Decoder input = [SOS] + SELFIES
            dec_in[i,0] = vocab["[SOS]"]
            dec_in[i,1:len(seq)+1] = torch.tensor([vocab[t] for t in seq])

            # Decoder output = SELFIES + [EOS]
            dec_out[i,:len(seq)] = torch.tensor([vocab[t] for t in seq])
            dec_out[i,len(seq)]  = vocab["[EOS]"]


        self.SMILES_enc = enc
        self.SMILES_dec_input = dec_in
        self.SMILES_dec_output = dec_out

        vocab_size, num_data
        print("vocab size : ", vocab_size,"\nnumber of data : ",num_data)
        self.num_data = num_data
        
        self.vocab = vocab
        self.vocab_size = vocab_size


        print(self.SMILES_enc.shape)
        self.properties = torch.tensor(self.properties, dtype=torch.float).unsqueeze(-1)
        print(self.properties.shape)

        self.test_data = self.SMILES_enc[50]

        print("PSMILES : ", gpselfies[50])
        print("After AIS encoding : ", enc[50])
        print("After AIS Tokenization : ", enc[50])
        print("After to number : ", enc[50])
        print("Properties : ",self.properties)
        print(len(enc[50]))
        
    
    def __getitem__(self, i):
        return self.SMILES_dec_input[i], self.SMILES_dec_output[i], self.properties[i]
    
    def __len__(self):
        return self.SMILES_enc.shape[0]
    
    def vocab_len(self):
        return self.vocab_size

dataset_LSTM = load_data_LSTM(Polymers)

generator = torch.Generator()
train_dataset_LSTM, val_dataset_LSTM = random_split(dataset_LSTM, [0.75, 0.25], generator=generator)

train_dataloader_LSTM = DataLoader(train_dataset_LSTM, batch_size=256, shuffle=True, drop_last=False)
val_dataloader_LSTM = DataLoader(val_dataset_LSTM, batch_size=256, shuffle=False, drop_last=False)

eval_dataloader_LSTM = DataLoader(dataset_LSTM, batch_size=256, shuffle=False, drop_last=False)

def decode_keep_star(grammar, selfies, sanitize=False, verbose=False):
    """
    Group SELFIES → RDKit Mol, but KEEP '*' dummy atoms (do not H-cap).
    """
    rings = []
    place_from_idx = {}
    inverse_place = []
    dummy_counter = Counter(1)
    group_atom = {}

    mol = selfies_to_graph_iterative(
        grammar=grammar,
        symbol_iter=_tokenize_selfies(selfies),
        selfies=selfies,
        rings=rings,
        dummy_counter=dummy_counter,
        place_from_idx=place_from_idx,
        inverse_place=inverse_place,
        verbose=verbose,
        group_atom=group_atom,
    )
    form_rings_bilocally_iterative(
        mol, rings, place_from_idx, inverse_place,
        dummy_counter, group_atom, verbose=verbose
    )

    res = mol.GetMol()  # convert RWMol→Mol
    if sanitize:
        # 기본 Sanitize는 '*'에도 대체로 안전하지만 필요시 제약 완화
        Chem.SanitizeMol(res, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)
    return res

id2tok = {idx: token for token, idx in dataset.vocab.items()}
def tok_ids_to_smiles(tok_ids):
    tokens = [id2tok[i] for i in tok_ids]   
    if "[EOS]" in tokens:
        tokens = tokens[:tokens.index("[EOS]")]
    sf_str = "".join(tokens)

    try:
        smiles = decode_keep_star(ess, sf_str)
        smiles = Chem.MolToSmiles(smiles)
    except Exception:
        return None

    # 2) PSMILES canonicalize – 실패 시 원본 유지
    try:
        cand = PS(smiles).canonicalize.psmiles
        if cand.count('*') == 2 and Chem.MolFromSmiles(cand):
            smiles = cand
    except Exception:
        pass

    return smiles


def reverse_one_hot_encoding(one_hot_tensor, vocab):
    # 인덱스 → 토큰 매핑 생성
    index_to_token = {idx: token for token, idx in vocab.items()}

    # 복원된 토큰 시퀀스를 저장할 리스트
    original_tokens_list = []

    # GPU 텐서면 CPU로
    if one_hot_tensor.is_cuda:
        one_hot_tensor = one_hot_tensor.cpu()

    # one-hot → argmax (인덱스 시퀀스)
    try:
        _ = one_hot_tensor.shape[1]
        one_hot_tensor = torch.argmax(one_hot_tensor, dim=-1)
    except:
        pass  # 이미 1D 라면 그대로 둠

    # “값이 166인” 위치만 추출
    idx_166 = torch.nonzero(one_hot_tensor == 10, as_tuple=True)[0]

    # 만약 166 토큰이 하나도 없으면 전체 길이 사용
    end_pos = idx_166[0].item() if len(idx_166) > 0 else one_hot_tensor.size(0)

    # 0부터 end_pos 직전까지 토큰 복원
    for i in range(end_pos):
        token_idx = one_hot_tensor[i].item()
        original_tokens_list.append(index_to_token.get(token_idx, "<UNK>"))

    return original_tokens_list

def build_grammar_mask(prefix_tokens: List[int],
                       idx2token   : Dict[int, str],
                       vocab_size  : int,
                       max_stars   : int = 2,
                       pad_idx     : int = 166,
                       end_idx     : int = 4) -> torch.BoolTensor:
    """
    · pad_idx(166)는
        ─ prefix가 pad로 끝나 있거나
        ─ prefix 마지막이 end_idx(EOS)일 때만 허용
      그 외에는 금지.
    · 괄호·'*' 규칙은 그대로 유지.
    """
    # ─── 상태 계산 ─── #
    paren_depth = 0
    star_count  = 0
    for tid in prefix_tokens:
        tok = idx2token[tid]
        if tok == '(':
            paren_depth += 1
        elif tok == ')':
            paren_depth = max(0, paren_depth - 1)
        elif tok == '*':
            star_count += 1

    mask = torch.ones(vocab_size, dtype=torch.bool)

    # ─── 괄호 제약 ─── #
    if paren_depth == 0:
        for idx, tok in idx2token.items():
            if tok == ')':
                mask[idx] = False

    # ─── '*' 제약 ─── #
    if star_count >= max_stars:
        for idx, tok in idx2token.items():
            if tok == '*':
                mask[idx] = False

    # ─── '[' 직후 제약 ─── #
    if prefix_tokens and idx2token[prefix_tokens[-1]] == '[':
        for idx, tok in idx2token.items():
            if tok != ']':
                mask[idx] = False

    return mask

vocab = {'(': 0, ')': 1, '=': 2, '[O;!R;C]': 3, '[CH3;!R;C]': 4, '[CH2;!R;CN]': 5, '[CH2;!R;CC]': 6, '[CH2;!R;CO]': 7, '[*;!R;C]': 8, '[SOS]': 9, '[EOS]': 10, '[O;!R;CC]': 11, '[NH;!R;CC]': 12, '[*;!R;O]': 13, '[O;!R;*C]': 14, '[C;!R;*OO]': 15, '[CH;!R;CCO]': 16, '[CH;!R;CCN]': 17, '[C;!R;CNO]': 18, '[N;!R;CCC]': 19, '[C;!R;*NO]': 20, '[NH;!R;*C]': 21, '[*;!R;N]': 22, '[CH3;!R;N]': 23, '[CH;!R;CCC]': 24, '[C;!R;CCCO]': 25, '[CH;!R;CC]': 26, '[F;!R;C]': 27, '[CH2;!R;C]': 28, '[C;!R;COO]': 29, '[CH3;!R;O]': 30, '[C;!R;CCCN]': 31, '#': 32, '[C;!R;CCCC]': 33, '[CH2;!R;CS]': 34, '[C;!R;CC]': 35, '[OH;!R;C]': 36, '[S;!R;CC]': 37, '[N;!R;C]': 38, '[C;!R;CN]': 39, '[CH;!R;C]': 40, '[C;!R;CCC]': 41, '[NH2;!R;C]': 42, '[CH;!R;CFF]': 43, '[O;!R;S]': 44, '[O;!R;CN]': 45, '[CH;!R;CCS]': 46, '[CH2;!R;CF]': 47, '[CH3;!R;S]': 48, '[C;!R;CCO]': 49, '[NH;!R;CO]': 50, '[CH2;!R;*C]': 51, '[C;!R;NNO]': 52, '[C;!R;CFFF]': 53, '[C;!R;CCFF]': 54, '[C;!R;NOO]': 55, '[C;!R;OOO]': 56, '[S;!R;CCOO]': 57, '[CH3;!R;Si]': 58, '[C;!R;NNS]': 59, '[S;!R;C]': 60, '[NH;!R;C]': 61, '[Cl;!R;C]': 62, '[OH;!R;P]': 63, '1': 64, '[C;!R;CNN]': 65, '[CH;!R;CN]': 66, '[SH;!R;C]': 67, '[CH;!R;CCF]': 68, '[O;!R;P]': 69, '[CH2;!R;NO]': 70, '[O;R;CC]': 71, '[C;!R;CCCl]': 72, '[CH2;!R;CSi]': 73, '[CH;!R;NN]': 74, '[cH;R;CC]': 75, '[c;R;CCO]': 76, '[CH;!R;CO]': 77, '[Si;!R;CCCC]': 78, '[C;!R;CCN]': 79, '[N;!R;CN]': 80, '[NH;!R;CS]': 81, '[CH;!R;COO]': 82, '[N;!R;CCO]': 83, '[P;!R;COOO]': 84, '[C;!R;CCNN]': 85, '[N;!R;CO]': 86, '[CH;!R;CCl]': 87, '[S;!R;CNOO]': 88, '[C;!R;CCCF]': 89, '[CH2;!R;NSi]': 90, '[O;!R;CS]': 91, '[NH2;!R;O]': 92, '[NH;!R;CN]': 93, '[CH;R;CCO]': 94, '[CH2;R;CO]': 95, '[C;R;OOO]': 96, '[C;!R;CCF]': 97, '[C;!R;CCOO]': 98, '[N;!R;CCN]': 99, '[N;!R;CS]': 100, '[S;!R;CCNO]': 101, '[OH;!R;N]': 102, '[C;!R;NNN]': 103, '[S;!R;CCO]': 104, '[CH2;!R;CP]': 105, '[CH;!R;CNP]': 106, '[OH;!R;S]': 107, '[NH2;!R;N]': 108, '[N;!R;CC]': 109, '[Si;!R;*CCC]': 110, '[*;!R;Si]': 111, '[CH;!R;CS]': 112, '[C;!R;*CO]': 113, '[S;!R;COOO]': 114, '[O;!R;CP]': 115, '[S;!R;NNOO]': 116, '[NH2;!R;S]': 117, '[CH;R;*CN]': 118, '[N;R;CCC]': 119, '2': 120, '[CH2;!R;OO]': 121, '[C;!R;NN]': 122, 'N': 123, '[S;!R;COOS]': 124, '[S;!R;S]': 125, '[C;!R;CCCS]': 126, '[CH2;!R;NS]': 127, '[P;!R;OOOO]': 128, '[CH;!R;CNO]': 129, '[S;!R;OOO]': 130, '[O;!R;*S]': 131, '[CH2;!R;SS]': 132, '[CH;!R;FFO]': 133, '[CH2;!R;OP]': 134, '[C;!R;NSS]': 135, '[CH;!R;COP]': 136, '[C;!R;CFFO]': 137, '[CH;!R;CNS]': 138, '[c;R;CCC]': 139, '[CH;!R;CNN]': 140, '[C;!R;CSi]': 141, '[C;R;NNO]': 142, '[c;R;*CC]': 143, '[CH;!R;*CC]': 144, '[CH2;!R;NN]': 145, '[[N+]': 146, 'O': 147, '[[O-]': 148, '[N;!R;NO]': 149, '[CH;!R;NO]': 150, '[CH2;!R;OSi]': 151, '[CH;!R;NNN]': 152, '[P;!R;CCOO]': 153, '[C;!R;CClCl]': 154, '[C;!R;CCNO]': 155, '[C;!R;COS]': 156, '[N;!R;CCS]': 157, '[C;!R;CCCCl]': 158, '[C;!R;CCFO]': 159, '[C;!R;CCOP]': 160, '[SH;!R;O]': 161, '[S;!R;OOOO]': 162, '[C;!R;OSS]': 163, '[C;!R;CCClCl]': 164, '[PH;!R;CCO]': 165, '[PAD]': 166}
index_to_token = {idx: token for token, idx in vocab.items()}

def generate_batch_sequence(
    model,
    z,
    max_length=265,
    start_token=None,
    end_token=None,
    fixed_seq_len=None,
    pad_token=None,
    device='cuda'
):
    """
    Batch-aware autoregressive generation without temperature scaling,
    top-k / top-p filtering, or fallback re-normalisation.

    Args
    ----
    decoder : nn.Module
        Autoregressive Transformer decoder.
    z : torch.Tensor
        Latent tensor of shape [B, seq_len, embed_dim].
    max_length : int
        Maximum number of tokens to generate (inclusive of <end>).
    start_token : int
        Token ID to place at the beginning of every sequence.
    end_token : int
        Generation stops for a sequence when this ID is produced.
    fixed_seq_len : int | None
        Context window length fed to the decoder each step.
        If None, defaults to `z.size(1)`.
    pad_token : int | None
        Token used to left-pad the context when it is shorter than
        `fixed_seq_len`.
    device : str
        CUDA / CPU device string.
    """

    decoder = model.decoder
    B = z.size(0)
    z = z.to(device)

    if fixed_seq_len is None:
        fixed_seq_len = z.size(1)

    # initialise each sequence with <start>
    generated = [[start_token] for _ in range(B)]
    finished = [False] * B

    for _ in range(max_length):
        # prepare fixed-length contexts
        x_in = [
            (seq + [pad_token] * (fixed_seq_len - len(seq)))[:fixed_seq_len]
            if len(seq) < fixed_seq_len
            else seq[-fixed_seq_len:]
            for seq in generated
        ]
        x_in = torch.tensor(x_in, dtype=torch.long, device=device)  # [B, L]

        # forward pass (single batch)
        with torch.no_grad():
            logits = decoder(x_in, z)          # [B, L, V]
            logits = model.predict(logits)     # [B, L, V]

        all_done = True
        for i in range(B):
            if finished[i]:
                continue
            all_done = False

            pos = min(len(generated[i]) - 1, fixed_seq_len - 1)
            next_logits = logits[i, pos, :]  # [V]

            # grammar-based masking
            mask = build_grammar_mask(
                generated[i], index_to_token, next_logits.size(0)
            ).to(device)
            next_logits = next_logits.masked_fill(~mask, -1e9)

            # convert to probabilities
            probs = F.softmax(next_logits, dim=-1)

            # sample next token
            tok = torch.multinomial(probs, 1).item()
            generated[i].append(tok)

            # stop if <end>
            if tok == end_token:
                finished[i] = True

        if all_done:
            break

    return generated

@torch.no_grad()
def generate_batch_sequence_LSTM(
    model,
    z,                         # [B, latent_dim]
    max_length     = 42,
    start_token    = None,
    end_token      = None,
    pad_token      = None,     # 완성된 뒤 채우고 싶을 때만 사용
    grammar_mask_fn= None,     # build_grammar_mask 등
    index_to_token = None,     # ID → 문자 매핑 (grammar mask용)
    device         = "cuda",
):
    """
    Batch-wise autoregressive generation for the LSTM-CVAE decoder.

    * 한 스텝에 <현재 토큰 1개>씩 넣어 hidden-state를 이어받으며 생성합니다.
    * z → (h0, c0) 초기화는 모델 내부의 `to_decoder` 선형층 로직을 그대로 사용합니다.
    * grammar_mask_fn 이 주어지면 매 스텝마다 (배치별) 허용 토큰을 필터링합니다.
    """

    model = model.to(device).eval()
    B     = z.size(0)
    z     = z.to(device)

    # ------------------------------------------------------------------
    # 1) z   →   초기 hidden-state (층 반복 포함)
    # ------------------------------------------------------------------
    h0_raw, c0_raw = torch.tanh(model.to_decoder(z)).chunk(2, dim=-1)   #  [B, d_model] 두 조각
    num_layers     = model.decoder.num_layers                            # 2 층 LSTM
    h              = h0_raw.unsqueeze(0).repeat(num_layers, 1, 1).contiguous()  # [L,B,d]
    c              = c0_raw.unsqueeze(0).repeat(num_layers, 1, 1).contiguous()

    # ------------------------------------------------------------------
    # 2) <SOS> 토큰으로 시작
    # ------------------------------------------------------------------
    generated = [[start_token] for _ in range(B)]
    cur_tok   = torch.full((B,), start_token, dtype=torch.long, device=device)
    finished  = [False] * B

    # ------------------------------------------------------------------
    # 3) 토큰 한 개씩 autoregressive 루프
    # ------------------------------------------------------------------
    for _ in range(max_length - 1):          # 이미 <SOS> 하나는 넣었으므로 −1
        # (a) 토큰 → 임베딩  (shape [B,1,d_model])
        emb = model.smiles_embbed(cur_tok).unsqueeze(1)

        # (b) LSTM 디코더 전진
        out, (h, c) = model.decoder(emb, (h, c))   # out: [B,1,d_model]
        logits = model.predict(out.squeeze(1))     # [B, vocab]

        # (c) 배치별 다음 토큰 샘플링
        next_tok_list = []
        for i in range(B):
            if finished[i]:
                # 이미 <EOS>를 낸 시퀀스는 pad_token(또는 end_token)으로만 채움
                next_tok_list.append(pad_token if pad_token is not None else end_token)
                continue

            logit_i = logits[i]

            # grammar mask 적용(선택)
            if grammar_mask_fn is not None:
                mask = grammar_mask_fn(
                    generated[i], index_to_token, logit_i.size(0)
                ).to(device)
                logit_i = logit_i.masked_fill(~mask, -1e9)

            # 확률화 & 샘플링
            prob_i = F.softmax(logit_i, dim=-1)
            tok    = torch.multinomial(prob_i, 1).item()

            generated[i].append(tok)
            next_tok_list.append(tok)

            if tok == end_token:
                finished[i] = True

        # 모든 시퀀스가 끝났으면 조기 종료
        if all(finished):
            break

        # 다음 스텝 입력 토큰 업데이트
        cur_tok = torch.tensor(next_tok_list, dtype=torch.long, device=device)

    return generated

@torch.no_grad()
def generate_batch_sequence_LSTM_MHA(
    model,
    z, *,                        # z  말고는 전부 키워드 전용!
    max_length      = 42,
    start_token     = None,
    end_token       = None,
    pad_token       = None,
    grammar_mask_fn = None,
    index_to_token  = None,
    device          = "cuda",
    return_tensor   = False,     # True → [B,L_max] LongTensor 로 리턴
):
    """
    Autoregressive generation for the LSTM-MHA decoder.

    z shape:
        • [B, latent_dim]            → 그대로 사용
        • [B, L_seq, latent_dim]     → dim=1 평균을 내서 사용
    """
    model  = model.to(device).eval()

    # ─── 0. z 전처리 ──────────────────────────────────────────────
    if z.dim() == 3:                         # [B, L, z]
        z_vec = z.mean(dim=1)                # [B, z]
    elif z.dim() == 2:                       # [B, z]
        z_vec = z
    else:
        raise ValueError("z must be [B,z] or [B,L,z]")

    B = z_vec.size(0)
    z_vec = z_vec.to(device)

    # ─── 1. (h0,c0) 초기화 ───────────────────────────────────────
    h0_raw, c0_raw = torch.tanh(model.to_decoder(z_vec)).chunk(2, dim=-1)
    num_layers     = model.decoder.num_layers
    h = h0_raw.unsqueeze(0).repeat(num_layers, 1, 1).contiguous()
    c = c0_raw.unsqueeze(0).repeat(num_layers, 1, 1).contiguous()

    # ─── 2. <SOS> 시작 ───────────────────────────────────────────
    generated = [[start_token] for _ in range(B)]
    cur_tok   = torch.full((B,), start_token, dtype=torch.long, device=device)
    finished  = [False] * B

    # ─── 3. 한 토큰씩 생성 루프 ──────────────────────────────────
    for _ in range(max_length - 1):          # <SOS> 포함했으므로 -1
        emb = model.smiles_embbed(cur_tok).unsqueeze(1)          # [B,1,E]
        out, (h, c) = model.decoder(emb, (h, c))                 # [B,1,E]
        logits = model.predict(out.squeeze(1))                   # [B,V]

        next_tok = []
        for i in range(B):
            if finished[i]:
                next_tok.append(pad_token if pad_token is not None else end_token)
                continue

            log_i = logits[i]
            if grammar_mask_fn is not None:
                mask = grammar_mask_fn(generated[i], index_to_token, log_i.size(0)).to(device)
                log_i = log_i.masked_fill(~mask, -1e9)

            prob_i = F.softmax(log_i, dim=-1)
            tok_i  = torch.multinomial(prob_i, 1).item()

            generated[i].append(tok_i)
            next_tok.append(tok_i)
            if tok_i == end_token:
                finished[i] = True

        if all(finished):
            break

        cur_tok = torch.tensor(next_tok, dtype=torch.long, device=device)

    # ─── 4. 결과 반환 ────────────────────────────────────────────
    if return_tensor:
        if pad_token is None:
            raise ValueError("pad_token must be provided if return_tensor=True")
        return pad_sequence(
            [torch.tensor(seq, dtype=torch.long, device=device) for seq in generated],
            batch_first=True, padding_value=pad_token
        )
    return generated

class ConditionalVAELoss(nn.Module):
    def __init__(
        self,
        vocab_size      : int,
        max_beta        : float = 1.0,     # β-VAE 상한
        cyc_steps       : int   = 400,
        num_cycles      : int   = 4,
        anneal_steps    : int   = 1000,    # β 스케줄 길이
        free_bits       : float = 0.02,    # per-dim nats
        capacity_max    : float = 0.0,     # 0이면 β-VAE, >0이면 Burgess-C
        capacity_inc    : float = 0.002,   # C(t) 증가속도 (nats/step)
        gamma           : float = 5.0,     # Burgess 벌점 계수
        prop_w          : float = 1.0,     # property loss 가중
        nce             : float = 0.02,    # Info NCE loss 가중
        sig_pen_q       : float = 0.003,   # Posterior Sigma Penalty 가중
        sig_pen_p       : float = 0.003,   # Prior Sigma Penalty 가중
        imb             : float = 0.05,     # imbalnace KLD 가중
        latent_dim      : int = 64
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.V  = vocab_size
        self.fb = free_bits
        self.max_beta = max_beta
        self.cyc_steps = cyc_steps
        self.num_cycles = num_cycles
        self.anneal   = anneal_steps
        self.C_max    = capacity_max
        self.C_inc    = capacity_inc
        self.gamma    = gamma
        self.prop_w   = prop_w
        self.proj = nn.Linear(3, self.latent_dim)
        self.nce      = nce
        self.sig_pen_q = sig_pen_q
        self.sig_pen_p = sig_pen_p
        self.imb      = imb
    def cyclical_beta(self, step: int, max_beta: float, cyc_steps: int, num_cycles: int) -> float:
        """Triangular cyclical β schedule."""
        cycle_idx = step // cyc_steps
        if cycle_idx >= num_cycles:
            return max_beta  # 이후엔 β 고정
        pos = (step % cyc_steps) / cyc_steps  # 0→1 선형
        return max_beta * pos
    def info_nce(self, z, y, temperature=0.2):
        """
        z : [B, d]  (posterior sample or mean)
        y : [B, k]  (조건 벡터)
        반환 : scalar loss
        """
        z = F.normalize(z, dim=-1)
        y = F.normalize(y, dim=-1)

        logits = torch.mm(z, y.t()) / temperature        # [B, B]
        labels = torch.arange(z.size(0), device=z.device)
        return F.cross_entropy(logits, labels)
    # --------------------------------------------------------------
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


    # --------------------------------------------------------------
    def forward(
        self, logits, target_tokens,
        mu_q, lv_q, mu_p, lv_p,
        prop_pred_mu, true_prop, prop_pred_z,
        step:int
    ):
        B, L, _ = logits.size()
        D       = self.latent_dim

        # 1) Reconstruction
        recon = F.cross_entropy(
            logits.reshape(-1, self.V),           # (B·L, V)
            target_tokens.view(-1),            # (B·L,)
            reduction='sum',
            ignore_index=dataset.vocab['[PAD]']
        ) / B

        # 2) KL(q‖p)   -------------------------------------------------
        q = Normal(mu_q, torch.exp(0.5 * lv_q))
        p = Normal(mu_p, torch.exp(0.5 * lv_p))
        kld_dim = torch.distributions.kl_divergence(q, p)  # (B, L, D)

        # (1‑a) free‑bits (per‑dim clamp, 단위 = nat)
        if self.fb > 0.0:
            kld_dim = torch.clamp(kld_dim, min=self.fb)

        # (1‑b) 시퀀스·토큰 집계
        kld_token = kld_dim.sum(-1)            # (B, L)  ← Σ_D
        kld_seq   = kld_token.sum(-1)          # (B,)    ← Σ_L

        raw_kld_seq   = kld_seq.mean()         # ⬅ 모니터링용  (nat / sequence)
        kld_per_token = kld_token.mean()       # nat / 토큰  (BCE 비교용)

        # ------------------------------------------------------------------
        # 2) KL term (β‑VAE / cyclical β) ----------------------------------
        beta = self.cyclical_beta(step, self.max_beta, self.cyc_steps, self.num_cycles)

        # capacity OFF (C_max = 0) – pure β‑VAE
        kl_term = beta * kld_per_token

        # ------------------------------------------------------------------
        # 3) Property 회귀 손실 -------------------------------------------
        prop_loss_mu = F.mse_loss(prop_pred_mu, true_prop)
        prop_loss_z  = F.mse_loss(prop_pred_z , true_prop)

        # ------------------------------------------------------------------
        # 4) InfoNCE -------------------------------------------------------
        cond = F.relu(self.proj(true_prop.squeeze(-1)))      # (B, H)
        z    = self.reparameterize(q.loc, q.scale.log())      # (B, L, D) → 모델 util 함수 가정
        info_nce = self.info_nce(z.mean(1), cond)

        # ------------------------------------------------------------------
        # 5) Regularizers --------------------------------------------------
        imb = ((kld_dim - kld_dim.mean()) ** 2).mean()
        sig_pen_q = torch.exp(q.scale.log()).mean()
        sig_pen_p = torch.exp(p.scale.log()).mean()

        # ------------------------------------------------------------------
        # 6) 총 손실 -------------------------------------------------------
        loss = (
            recon
            + kl_term
            + self.prop_w * (prop_loss_mu + 0.5 * prop_loss_z)
            + self.nce * info_nce
            + self.sig_pen_q * sig_pen_q
            + self.sig_pen_p * sig_pen_p
            + self.imb * imb
        )

        # 모니터링 값들 리턴 ------------------------------------------------
        return loss, recon,kl_term.detach(),raw_kld_seq.detach(),kld_per_token.detach(),prop_loss_mu.detach(),




class ConditionalVAELoss_LSTM(nn.Module):
    def __init__(
        self,
        vocab_size      : int,
        max_beta        : float = 1.0,     # β-VAE 상한
        cyc_steps       : int   = 400,
        num_cycles      : int   = 4,
        anneal_steps    : int   = 1000,    # β 스케줄 길이
        free_bits       : float = 0.02,    # per-dim nats
        capacity_max    : float = 0.0,     # 0이면 β-VAE, >0이면 Burgess-C
        capacity_inc    : float = 0.002,   # C(t) 증가속도 (nats/step)
        gamma           : float = 5.0,     # Burgess 벌점 계수
        prop_w          : float = 1.0,     # property loss 가중
        nce             : float = 0.02,    # Info NCE loss 가중
        sig_pen_q       : float = 0.003,   # Posterior Sigma Penalty 가중
        sig_pen_p       : float = 0.003,   # Prior Sigma Penalty 가중
        imb             : float = 0.05,     # imbalnace KLD 가중
        latent_dim      : int = 64
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.V  = vocab_size
        self.fb = free_bits
        self.max_beta = max_beta
        self.cyc_steps = cyc_steps
        self.num_cycles = num_cycles
        self.anneal   = anneal_steps
        self.C_max    = capacity_max
        self.C_inc    = capacity_inc
        self.gamma    = gamma
        self.prop_w   = prop_w
        self.proj = nn.Linear(3, self.latent_dim)
        self.nce      = nce
        self.sig_pen_q = sig_pen_q
        self.sig_pen_p = sig_pen_p
        self.imb      = imb
    def cyclical_beta(self, step: int, max_beta: float, cyc_steps: int, num_cycles: int) -> float:
        """Triangular cyclical β schedule."""
        cycle_idx = step // cyc_steps
        if cycle_idx >= num_cycles:
            return max_beta  # 이후엔 β 고정
        pos = (step % cyc_steps) / cyc_steps  # 0→1 선형
        return max_beta * pos
    def info_nce(self, z, y, temperature=0.2):
        """
        z : [B, d]  (posterior sample or mean)
        y : [B, k]  (조건 벡터)
        반환 : scalar loss
        """
        z = F.normalize(z, dim=-1)
        y = F.normalize(y, dim=-1)

        logits = torch.mm(z, y.t()) / temperature        # [B, B]
        labels = torch.arange(z.size(0), device=z.device)
        return F.cross_entropy(logits, labels)
    # --------------------------------------------------------------
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


    # --------------------------------------------------------------
    def forward(
        self, logits, target_tokens,
        mu_q, lv_q, mu_p, lv_p,
        prop_pred_mu, true_prop, prop_pred_z,
        step:int
    ):
        B, L, _ = logits.size()
        D       = self.latent_dim

        # 1) Reconstruction
        recon = F.cross_entropy(
            logits.reshape(-1, self.V),           # (B·L, V)
            target_tokens.view(-1),            # (B·L,)
            reduction='sum',
            ignore_index=dataset.vocab['[PAD]']
        ) / B

        # 2) KL(q‖p)   -------------------------------------------------
        q = Normal(mu_q, torch.exp(0.5 * lv_q))
        p = Normal(mu_p, torch.exp(0.5 * lv_p))
        kld_dim = torch.distributions.kl_divergence(q, p)  # (B, L, D)

        # (1‑a) free‑bits (per‑dim clamp, 단위 = nat)
        if self.fb > 0.0:
            kld_dim = torch.clamp(kld_dim, min=self.fb)

        # (1‑b) 시퀀스·토큰 집계
        kld_token = kld_dim.sum(-1)            # (B, L)  ← Σ_D
        kld_seq   = kld_token.sum(-1)          # (B,)    ← Σ_L

        raw_kld_seq   = kld_seq.mean()         # ⬅ 모니터링용  (nat / sequence)
        kld_per_token = kld_token.mean()       # nat / 토큰  (BCE 비교용)

        # ------------------------------------------------------------------
        # 2) KL term (β‑VAE / cyclical β) ----------------------------------
        beta = self.cyclical_beta(step, self.max_beta, self.cyc_steps, self.num_cycles)

        # capacity OFF (C_max = 0) – pure β‑VAE
        kl_term = beta * kld_per_token

        # ------------------------------------------------------------------
        # 3) Property 회귀 손실 -------------------------------------------
        prop_loss_mu = F.mse_loss(prop_pred_mu, true_prop)
        prop_loss_z  = F.mse_loss(prop_pred_z , true_prop)

        # ------------------------------------------------------------------
        # 4) InfoNCE -------------------------------------------------------
        cond = F.relu(self.proj(true_prop.squeeze(-1)))      # (B, H)
        z    = self.reparameterize(q.loc, q.scale.log())      # (B, L, D) → 모델 util 함수 가정
        info_nce = self.info_nce(z, cond)

        # ------------------------------------------------------------------
        # 5) Regularizers --------------------------------------------------
        imb = ((kld_dim - kld_dim.mean()) ** 2).mean()
        sig_pen_q = torch.exp(q.scale.log()).mean()
        sig_pen_p = torch.exp(p.scale.log()).mean()

        # ------------------------------------------------------------------
        # 6) 총 손실 -------------------------------------------------------
        loss = (
            recon
            + kl_term
            + self.prop_w * (prop_loss_mu + 0.5 * prop_loss_z)
            + self.nce * info_nce
            + self.sig_pen_q * sig_pen_q
            + self.sig_pen_p * sig_pen_p
            + self.imb * imb
        )

        # 모니터링 값들 리턴 ------------------------------------------------
        return loss, recon,kl_term.detach(),raw_kld_seq.detach(),kld_per_token.detach(),prop_loss_mu.detach(),




PAD_IDX = 166
DEVICE  = 'cuda'

# -------------------------------------------------------------
# 0. 헬퍼 함수
# -------------------------------------------------------------
def decode_tokens(model, z, dec_in, properties):
    
    try: # Trans_MHA
        prop_p = model.input_embedding_p(properties)
        prop_p = model.pos_enc(prop_p)
        z_z    = model.crossattn(z.view(-1, 42, 96), prop_p, prop_p)
        z_z    = model.ff(z_z)
        dec    = model.decoder(dec_in, z_z)
    except: # Trans
        dec    = model.decoder(dec_in, z.view(-1, 42, 96))
    return model.predict(dec)                  # [B, L, vocab]

def log_px_z(logits, target):
    ll = -F.cross_entropy(
        logits.transpose(1, 2), target,
        reduction='none', ignore_index=PAD_IDX
    )
    return ll.sum(dim=1)                       # [B]

def decode_tokens_LSTM(model, z, dec_in, properties):
    prop_p = model.input_embedding_p(properties)
    try: # LSTM_MHA
        z_z    = model.crossattn(z, prop_p, prop_p)
        z_z    = model.ff(z_z)
        dec    = model.decoder(dec_in, z_z)
    except: # LSTM
        dec    = model.decoder(dec_in, z_z)
    return model.predict(dec)                  # [B, L, vocab]

def log_px_z(logits, target):
    ll = -F.cross_entropy(
        logits.transpose(1, 2), target,
        reduction='none', ignore_index=PAD_IDX
    )
    return ll.sum(dim=1)                       # [B]


@torch.inference_mode()
def iwae_bound(
        model, prior,
        sm_enc, sm_dec_in, sm_dec_tgt, props,
        K=64, chunk=8, pad_idx=0):

    B, L_dec = sm_dec_tgt.shape
    enc   = model.encoder(sm_enc, model.input_embedding(props))
    mu_q, lv_q = model.to_means(enc.reshape(B, -1)), model.to_var(enc.reshape(B, -1))
    q     = Normal(mu_q, (0.5 * lv_q).exp())

    mu_p, lv_p = prior(props.squeeze())
    p     = Normal(mu_p, (0.5 * lv_p).exp())

    log_ws = []

    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):   # ③+④
        for k0 in range(0, K, chunk):
            k_eff = min(chunk, K - k0)

            # --- K_e chunk 샘플
            z = q.rsample((k_eff,))                    # [k_eff,B,L,D]

            # flatten batch
            z_2d = z.reshape(k_eff * B, *z.shape[2:])
            logits = decode_tokens(
                model, z_2d,
                sm_dec_in.repeat(k_eff, 1),
                props.repeat(k_eff, 1, 1)
            )                                         # [(kB),L,V]

            # log p(x|z) without keeping full logits
            V = logits.size(-1)
            logp = torch.log_softmax(logits, dim=-1)  # [(kB),L,V]
            ll_tok = logp.gather(
                -1, sm_dec_tgt.repeat(k_eff,1).unsqueeze(-1)
            ).squeeze(-1)                             # [(kB),L]
            ll_tok.masked_fill_(sm_dec_tgt.repeat(k_eff,1)==pad_idx, 0.)
            log_px = ll_tok.sum(-1).reshape(k_eff, B) # [k_eff,B]

            # log q, log p
            reduced_dims = tuple(range(2, z.dim()))
            log_qz = q.log_prob(z).sum(reduced_dims)       # [k_eff,B]
            log_pz = p.log_prob(z).sum(reduced_dims)       # [k_eff,B]

            log_ws.append(log_px + log_pz - log_qz)

    log_ws = torch.cat(log_ws, dim=0)                 # [K,B]
    logw_max = log_ws.max(0, keepdim=True).values
    iwae = logw_max + (log_ws - logw_max).exp().mean(0).log()
    return iwae                                       # [B]

@torch.no_grad()
def kl_decompose_minibatch(mu_b, lv_b, dataset_size):
    m, D = mu_b.size()
    q    = Normal(mu_b, (0.5 * lv_b).exp())
    z    = q.rsample()                                 # [m, D]

    log_q_ij = (-0.5 * (
        ((z[:, None] - mu_b[None]) ** 2) / lv_b.exp()[None]
        + lv_b[None] + math.log(2 * math.pi)
    ).sum(-1))                                         # [m, m]

    log_qz        = torch.logsumexp(log_q_ij, dim=1) - math.log(dataset_size)
    log_qz_prod   = q.log_prob(z).sum(-1)
    log_pz        = Normal(0, 1).log_prob(z).sum(-1)

    mi = (q.log_prob(z).sum(-1) - log_qz).mean()
    tc = (log_qz - log_qz_prod).mean()
    dw = (log_qz_prod - log_pz).mean()
    return {'MI': mi.item(), 'TC': tc.item(), 'DimKL': dw.item()}

def active_units(mu, eps=1e-2):
    return (mu.var(0) > eps).sum().item()

def _rbf(x, sigma):
    diff2 = ((x[:, None] - x[None]) ** 2).sum(-1)
    return torch.exp(-diff2 / (2 * sigma ** 2))

# ---------------------------------------------------
# RFF 기반 HSIC  (unbiased, ^2-norm kernel)
# ---------------------------------------------------
def hsic_rff(x, y, num_feat=2048, sigma=None):
    """
    x : [N, Dx], y : [N, Dy]
    반환 : 스칼라 HSIC
    """
    N, Dx = x.shape
    Dy = y.shape[1]

    # ① adaptive sigma
    if sigma is None:
        sigma = 0.5 * (
            x.var(0, unbiased=False).mean().sqrt() +
            y.var(0, unbiased=False).mean().sqrt()
        )

    # ② Random Fourier Features
    w_x = torch.randn(Dx, num_feat, device=x.device) / sigma
    w_y = torch.randn(Dy, num_feat, device=y.device) / sigma
    b   = 2 * math.pi * torch.rand(num_feat, device=x.device)   # shared phase

    z_x = (x @ w_x + b).cos() * math.sqrt(2.0 / num_feat)      # [N, F]
    z_y = (y @ w_y + b).cos() * math.sqrt(2.0 / num_feat)

    # ③ 센터링
    z_xc = z_x - z_x.mean(0, keepdim=True)
    z_yc = z_y - z_y.mean(0, keepdim=True)

    # ④ Frobenius-norm² 무편향 HSIC
    C = (z_xc.T @ z_yc) / (N - 1)      # [F, F]
    return (C ** 2).sum()              # scalar

def get_random_batch(dataloader):
    idx = random.randrange(len(dataloader))
    for i, batch in enumerate(dataloader):
        if i == idx:
            return batch

import torch
from torch.distributions.normal import Normal
from contextlib import nullcontext

# ════════════════════════════════════════════════════════════════════════════
# 1)  z 시퀀스  → (h0,c0)  → 디코더 → logits  (cross-attention 포함)
# ════════════════════════════════════════════════════════════════════════════
def decode_tokens_crossattn(model,
                            z_seq: torch.Tensor,      # [B, L_z, z_dim]  (or [kB, L_z, z])
                            props: torch.Tensor,      # [B, 3, 1]        (or [kB,3,1])
                            sm_dec_out: torch.Tensor  # [B, L_dec]       (or [kB,L_dec])
                            ) -> torch.Tensor:        # → [B, L_dec, V]

    # ── (a) z–property cross-attention + FFN ────────────────────────────────
    # props → 1-token key/value
    prop_p = model.input_embedding_p(props)                            # [B,3,E′]
    z_attn = model.crossattn(z_seq, prop_p, prop_p)                    # [B,L_z,z]
    z_ff   = model.ff(z_attn)                                          # [B,L_z,z]

    # ── (b) 시퀀스 평균 → (h0,c0) ───────────────────────────────────────────
    z_mean = z_ff.mean(1)                                              # [B,z]
    h0, c0 = torch.tanh(model.to_decoder(z_mean)).chunk(2, dim=-1)
    h0, c0 = h0.unsqueeze(0).repeat(2, 1, 1), c0.unsqueeze(0).repeat(2, 1, 1)

    # ── (c) 디코더 forward (GT 토큰 teacher-forcing) ───────────────────────
    emb_in   = model.smiles_embbed(sm_dec_out)                         # [B,L_dec,d]
    dec_out, _ = model.decoder(emb_in, (h0, c0))                       # [B,L_dec,d]
    logits   = model.predict(dec_out)                                  # [B,L_dec,V]
    return logits


# ════════════════════════════════════════════════════════════════════════════
# 2)  IWAE bound   (cross-attention LSTM-CVAE,  sm_enc + sm_dec_out 버전)
# ════════════════════════════════════════════════════════════════════════════
@torch.inference_mode()
def iwae_bound_lstm_MHA(model,                # LSTM-CVAE (cross-attn)
                         prior,                # prior network  props → (μ, logσ²)
                         sm_enc,               # [B, L_enc]   (<cond> + SMILES)
                         sm_dec_out,           # [B, L_dec]   (GT 토큰, <SOS> 포함)
                         props,                # [B, 3, 1]    (또는 squeeze 후 [B,3])
                         K: int = 64,
                         chunk: int = 8,
                         pad_idx: int = 0):

    B, L_dec = sm_dec_out.shape
    device   = sm_enc.device
    amp_ctx  = torch.amp.autocast(device_type='cuda') if device.type == 'cuda' else nullcontext()

    # ────────────────────────────────────────────────────────────────────
    # 1. q(z | x,c)   – μ_q, log σ²_q (시퀀스 전체)
    # ────────────────────────────────────────────────────────────────────
    prop_tok = model.input_embedding(props)                                     # [B,3,d]
    enc_src  = model.smiles_embbed(sm_enc)                                      # [B,L_enc,d]
    enc_in   = torch.cat([enc_src, prop_tok], dim=1)                            # [B,L_enc+1,d]

    enc_out  = model.encoder(enc_in)[0]                                         # [B,L_q,d]
    mu_q, logv_q = model.to_means(enc_out), model.to_var(enc_out)               # [B,L_q,z]
    q_dist       = Normal(mu_q, (0.5 * logv_q).exp())

    # ────────────────────────────────────────────────────────────────────
    # 2. p(z | c)
    #    – prior network가 시퀀스 길이에 맞게 μ_p, log σ²_p를 내놓는다고 가정
    # ────────────────────────────────────────────────────────────────────
    mu_p, logv_p = prior(props.squeeze())                                       # [B,L_q,z]
    p_dist       = Normal(mu_p, (0.5 * logv_p).exp())

    # ────────────────────────────────────────────────────────────────────
    # 3. Importance-weighted 샘플링
    # ────────────────────────────────────────────────────────────────────
    log_ws = []
    with torch.no_grad(), amp_ctx:
        for k0 in range(0, K, chunk):
            k_eff = min(chunk, K - k0)

            # 3-a) z ~ q
            z = q_dist.rsample((k_eff,))                # [k_eff,B,L_q,z]
            z2d = z.reshape(k_eff * B, *z.shape[2:])    # [kB,L_q,z]

            # 3-b) log p(x | z,c)
            logits = decode_tokens_crossattn(
                model,
                z2d,
                props.repeat(k_eff, 1, 1),
                sm_dec_out.repeat(k_eff, 1)
            )                                           # [(kB),L_dec,V]

            logp = torch.log_softmax(logits, dim=-1)
            ll_tok = logp.gather(
                -1,
                sm_dec_out.repeat(k_eff, 1).unsqueeze(-1)
            ).squeeze(-1)                               # [(kB),L_dec]

            ll_tok.masked_fill_(sm_dec_out.repeat(k_eff,1) == pad_idx, 0.)
            log_px = ll_tok.sum(-1).reshape(k_eff, B)   # [k_eff,B]

            # 3-c) log q(z|x,c)  &  log p(z|c)
            log_qz = q_dist.log_prob(z).sum((-1, -2))   # [k_eff,B]  (sum over z_dim & seq_len)
            log_pz = p_dist.log_prob(z).sum((-1, -2))   # [k_eff,B]

            log_ws.append(log_px + log_pz - log_qz)

    # ────────────────────────────────────────────────────────────────────
    # 4. IWAE bound   (log-mean-exp 안정화)
    # ────────────────────────────────────────────────────────────────────
    log_ws   = torch.cat(log_ws, dim=0)                 # [K,B]
    logw_max = log_ws.max(0, keepdim=True).values
    iwae     = logw_max + (log_ws - logw_max).exp().mean(0).log()  # [B]

    return iwae

# ─────────────────────────────
# z → (h0,c0) → 디코더 → logits
# ─────────────────────────────
def decode_tokens_lstm_v3(model,
                          z: torch.Tensor,          # [B, z_dim]  or [kB, z_dim]
                          sm_dec_out: torch.Tensor  # [B, L_dec]
                          ) -> torch.Tensor:        # → [B, L_dec, V]

    # (1) z → (h0, c0)
    h0, c0 = torch.tanh(model.to_decoder(z)).chunk(2, dim=-1)
    h0, c0 = h0.unsqueeze(0).repeat(2, 1, 1), c0.unsqueeze(0).repeat(2, 1, 1)

    # (2) 디코더 forward (teacher-forcing = GT 토큰 그대로)
    emb_in   = model.smiles_embbed(sm_dec_out)   # [B, L_dec, d]
    dec_out, _ = model.decoder(emb_in, (h0, c0))
    logits   = model.predict(dec_out)            # [B, L_dec, V]
    return logits


# ─────────────────────────────
# IWAE bound  (sm_enc + sm_dec_out)
# ─────────────────────────────
@torch.inference_mode()
def iwae_bound_lstm(model,                # LSTM-CVAE
                       prior,                # prior network (props → μ, logσ²)
                       sm_enc,               # [B, L_enc]  (<cond>+SMILES)
                       sm_dec_out,           # [B, L_dec]  (GT 토큰, <SOS> 포함)
                       props,                # [B, 3]
                       K: int   = 64,
                       chunk: int = 8,
                       pad_idx: int = 0):

    B, L_dec = sm_dec_out.shape
    device   = sm_enc.device
    amp_ctx  = torch.amp.autocast(device_type='cuda') if device.type == 'cuda' else nullcontext()

    # ─── 1. q(z|x,c) 파라미터 ─────────────────────────────
    prop_tok  = model.input_embedding(props)                # [B,3,d]
    enc_src   = model.smiles_embbed(sm_enc)                 # [B,L_enc,d]
    enc_in    = torch.cat([prop_tok, enc_src], dim=1)       # [B,L_enc+3,d]

    h_enc     = model.encoder(enc_in)[1][0][-1]             # [B,d]
    mu_q, logv_q = model.to_means(h_enc), model.to_var(h_enc)  # [B,z]
    q_dist       = Normal(mu_q, (0.5 * logv_q).exp())

    # ─── 2. p(z|c) ─────────────────────────────────────────
    mu_p, logv_p = prior(props.squeeze())                             # [B,z]
    p_dist       = Normal(mu_p, (0.5 * logv_p).exp())

    # ─── 3. Importance-weighted 샘플 ───────────────────────
    log_ws = []
    with torch.no_grad(), amp_ctx:
        for k0 in range(0, K, chunk):
            k_eff = min(chunk, K - k0)

            # (a) z ∼ q
            z = q_dist.rsample((k_eff,))        # [k_eff,B,z]
            z2d = z.reshape(k_eff * B, -1)      # [kB,z]

            # (b) log p(x|z,c)
            logits = decode_tokens_lstm_v3(
                model,
                z2d,
                sm_dec_out.repeat(k_eff, 1)
            )                                   # [(kB),L_dec,V]

            logp = torch.log_softmax(logits, dim=-1)
            ll_tok = logp.gather(
                -1,
                sm_dec_out.repeat(k_eff, 1).unsqueeze(-1)
            ).squeeze(-1)                       # [(kB),L_dec]

            ll_tok.masked_fill_(sm_dec_out.repeat(k_eff,1) == pad_idx, 0.)
            log_px = ll_tok.sum(-1).reshape(k_eff, B)       # [k_eff,B]

            # (c) log q, log p(z|c)
            log_qz = q_dist.log_prob(z).sum(-1)             # [k_eff,B]
            log_pz = p_dist.log_prob(z).sum(-1)             # [k_eff,B]

            log_ws.append(log_px + log_pz - log_qz)

    # ─── 4. IWAE bound (log-mean-exp 안정화) ───────────────
    log_ws   = torch.cat(log_ws, dim=0)                     # [K,B]
    logw_max = log_ws.max(0, keepdim=True).values
    iwae     = logw_max + (log_ws - logw_max).exp().mean(0).log()  # [B]

    return iwae

