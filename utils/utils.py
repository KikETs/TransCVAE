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
from collections import Counter
from IPython.display import clear_output, display, HTML
import ipywidgets as widgets
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import umap.umap_ as umap
import re
import selfies_psmiles as sfp
from rdkit.Chem import rdmolops


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log_transformer = FunctionTransformer(np.log1p, validate=True)
log_minmax_pipeline = Pipeline(steps=[
    ('log', log_transformer),
    ('minmax', MinMaxScaler())
])


def decode_keep_star(selfies_str: str, sanitize: bool = False, verbose: bool = True):
    """
    SELFIES → RDKit Mol, but KEEP '*' dummy atoms (do not add H or rewrite).
    - sf.decoder(selfies) 로 SMILES를 얻고,
    - RDKit으로 파싱하되 sanitize=False로 별표 보존.
    """
    # 1) SELFIES → SMILES (패치된 sf.decoder는 [*] 포함 SMILES 반환)
    
    smiles = sfp.decoder_psmiles(selfies_str)
    if verbose:
        print(f"[decode_keep_star] SELFIES: {selfies_str}")
        print(f"[decode_keep_star] SMILES : {smiles}")

    # 2) RDKit Mol (sanitize=False로 원형 보존; '*' 유지)
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")

    # 요청 시, 아주 완화된 sanitize만 수행(필요 없으면 생략 가능)
    if sanitize:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)

    return mol

def tok_ids_to_smiles(tok_ids, id2tok):
    """
    토큰 ID 시퀀스 → SELFIES 문자열 → SMILES
    - [EOS] 앞까지만 사용
    - sf.decoder 로 디코드
    - PSMILES canonicalize 시도 (실패하면 원본 유지)
    """
    tokens = [id2tok[i] for i in tok_ids]
    if "[EOS]" in tokens:
        tokens = tokens[:tokens.index("[EOS]")]
    sf_str = "".join(tokens)
    try:
        # SELFIES -> RDKit Mol (별표 유지), -> canonical SMILES
        mol = decode_keep_star(sf_str, sanitize=False, verbose=True)
        smiles = Chem.MolToSmiles(mol)
    except Exception:
        return None

    # 2) PSMILES canonicalize – 실패 시 원본 유지
    try:
        # NOTE: 아래 PS(...)는 사용 중인 PSMILES canonicalizer의 API를 그대로 둔 것임.
        cand = PS(smiles).canonicalize.psmiles
        # 별표 2개(양단) 조건과 RDKit 재파싱 검사
        if cand.count('*') == 2 and Chem.MolFromSmiles(cand):
            smiles = cand
    except Exception:
        pass

    return smiles

def compute_ess(log_w: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Effective Sample Size (ESS) 계산
    --------------------------------
    log_w : (..., K)  IWAE 계산 시 나온 log 중요도 가중치
    dim   : 샘플 축(K). 기본 = 마지막 차원
    return: (...,)    배치마다 ESS
    """
    # double로 변환해 언더플로 완화
    log_w = log_w.double()

    # log Σ w   ,   log Σ w²   (안정적 log-sum-exp)
    log_sum_w  = torch.logsumexp(log_w,       dim=dim)
    log_sum_w2 = torch.logsumexp(2.0*log_w,   dim=dim)

    # ESS = (Σ w)² / Σ w²  ≡  exp( 2·logΣw - logΣw² )
    ess = torch.exp(2.0 * log_sum_w - log_sum_w2)
    return ess


def make_src_key_padding_mask(lengths: torch.Tensor, max_len: int | None = None):
    """
    lengths: (batch,) 각 시퀀스의 실제 길이
    return : (batch, max_len) True=PAD, False=valid
    """
    if max_len is None:
        max_len = lengths.max().item()
    range_row = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # (1, max_len)
    mask = range_row >= lengths.unsqueeze(1)                               # (batch, max_len)
    return mask