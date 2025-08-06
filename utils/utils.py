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
from rdkit.Chem import rdmolops
import selfies as sf   # polyselfies가 selfies를 패치해 설치해 줌


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log_transformer = FunctionTransformer(np.log1p, validate=True)
log_minmax_pipeline = Pipeline(steps=[
    ('log', log_transformer),
    ('minmax', MinMaxScaler())
])

ess = GroupGrammar.essential_set()

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

def tok_ids_to_smiles(tok_ids, id2tok):
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