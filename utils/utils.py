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




