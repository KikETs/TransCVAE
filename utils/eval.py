from utils.utils import *
from utils.dataloader import dataset
from rdkit import DataStructs
PAD_IDX = dataset.vocab['[PAD]']
# -------------------------------------------------------------
# 0. 헬퍼 함수
# -------------------------------------------------------------
def decode_tokens(model, z, dec_in, properties):
    
    try: # Trans_MHA
        prop_p = model.input_embedding_p(properties)
        prop_p = model.pos_enc(prop_p)
        z_z    = model.crossattn(z, prop_p, prop_p)
        z_z    = model.ff(z_z)
        dec    = model.decoder(dec_in, z_z)
    except: # Trans
        dec    = model.decoder(dec_in, z)
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
    mu_q, lv_q = model.to_means(enc), model.to_var(enc)
    q     = Normal(mu_q, (0.5 * lv_q).exp())

    mu_p, lv_p = prior(props.squeeze())
    p     = Normal(mu_p, (0.5 * lv_p).exp())

    log_ws = []

    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):   # ③+④
        for k0 in range(0, K, chunk):
            k_eff = min(chunk, K - k0)

            # --- K_e chunk 샘플
            z = q.rsample((k_eff,))                    # [k_eff,B,L,D]
            z2d = z.reshape(k_eff * B, *z.shape[2:])    # [kB,L_q,z]

            logits = decode_tokens(
                model, z2d,
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




#####################################################################
# Utility: SMILES ↔ Fingerprint & Tanimoto
#####################################################################

def _smiles_to_fp(sm: str):
    mol = Chem.MolFromSmiles(sm)
    return Chem.RDKFingerprint(mol) if mol is not None else None


def tanimoto(sm1: str, sm2: str) -> float:
    fp1, fp2 = _smiles_to_fp(sm1), _smiles_to_fp(sm2)
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)

#####################################################################
# Beam‑search decoder (temperature + length‑norm)
#####################################################################

def _beam_search_decode(model, z, sos_id: int, eos_id: int, *,
                        beam_width: int = 20, max_len: int = 128,
                        len_penalty: float = 1.0, vocab_size: int = 9999,
                        temperature: float = 0.9, alpha: float = 0.7):
    """Return list[List[int]] without SOS. OOV ids filtered."""
    device = z.device
    batch  = z.size(0)
    out_tokens = []
    for b in tqdm(range(batch)):
        beams = [([sos_id], 0.0, 0.0)]  # (seq, raw_logp, norm_score)
        finished = []
        for _ in range(max_len):
            new_beams = []
            for seq, raw_lp, _ in beams:
                if seq[-1] == eos_id:
                    finished.append((seq, raw_lp))
                    continue
                hidden = model.decoder(
                    torch.tensor(seq, device=device).unsqueeze(0),
                    z[b:b+1]
                )
                logits = model.predict(hidden)[:, -1] / temperature  # sharpen
                logp   = F.log_softmax(logits, dim=-1).squeeze(0)
                topk_val, topk_idx = torch.topk(logp, k=min(beam_width*2, logp.size(0)))
                # filter OOV
                cand = [(tid, tv) for tid, tv in zip(topk_idx.tolist(), topk_val.tolist()) if tid < vocab_size]
                for tid, tv in cand[:beam_width]:
                    new_seq   = seq + [tid]
                    new_rawlp = raw_lp + tv
                    norm_lp   = new_rawlp / (len(new_seq) ** alpha)
                    new_beams.append((new_seq, new_rawlp, norm_lp))
            if not new_beams:
                break
            new_beams.sort(key=lambda x: x[2], reverse=True)
            beams = new_beams[:beam_width]
        if finished:
            for seq, raw_lp in finished:
                beams.append((seq, raw_lp, raw_lp / (len(seq) ** alpha)))
        best_seq = max(beams, key=lambda x: x[1] / (len(x[0]) ** len_penalty))[0]
        # strip SOS/EOS
        best_seq = best_seq[1: best_seq.index(eos_id)] if eos_id in best_seq else best_seq[1:]
        out_tokens.append(best_seq)
    return out_tokens

#####################################################################
# Reconstruction with z = μ  (greedy or beam)
#####################################################################

def reconstruct_zmu(model, dataloader, vocab: dict, *,
                     beam_width: int = 1, len_penalty: float = 1.0,
                     max_len: int | None = None,
                     temperature: float = 0.9,
                     alpha: float = 0.7):
    """Return (mean_tanimoto, num_pairs). beam_width=1 → greedy."""
    model.eval()
    model.cpu()
    tanis, pairs = [], 0
    device = next(model.parameters()).device

    SOS, EOS, PAD = vocab['[SOS]'], vocab['[EOS]'], vocab['[PAD]']
    if max_len is None:
        max_len = 128

    for enc_in, _, _, props in dataloader:
        enc_in, props = enc_in.to(device), props.to(device)
        with torch.no_grad():
            prop_e  = model.input_embedding(props)
            encoded = model.encoder(enc_in, prop_e)
            mu_q    = model.to_means(encoded)
            z       = mu_q  # z = μ

            if beam_width == 1:
                # Greedy ------------------------------------------------------
                dec_in = torch.full((enc_in.size(0), 1), SOS, device=device)
                done   = torch.zeros(enc_in.size(0), dtype=torch.bool, device=device)
                out_tok= [[] for _ in range(enc_in.size(0))]
                for _ in range(max_len):
                    hidden = model.decoder(dec_in, z)
                    logits = model.predict(hidden)[:, -1] / temperature
                    next_tok = logits.argmax(-1, keepdim=True)
                    dec_in = torch.cat([dec_in, next_tok], dim=1)
                    for i, tok in enumerate(next_tok.squeeze(1).tolist()):
                        if not done[i]:
                            if tok == EOS:
                                done[i] = True
                            else:
                                out_tok[i].append(tok)
                    if done.all():
                        break
            else:
                # Beam --------------------------------------------------------
                out_tok = _beam_search_decode(model, z, SOS, EOS,
                                              beam_width=beam_width,
                                              max_len=max_len,
                                              len_penalty=len_penalty,
                                              vocab_size=len(vocab),
                                              temperature=temperature,
                                              alpha=alpha)
        # SELFIES → SMILES & Tanimoto ----------------------------------------
        for r_tok, o_tok in zip(enc_in.cpu().tolist(), out_tok):
            ref_sm = tok_ids_to_smiles([t for t in r_tok if t not in (PAD,)])
            out_sm = tok_ids_to_smiles([t for t in o_tok if t not in (PAD,)])
            if ref_sm and out_sm:
                tanis.append(tanimoto(ref_sm, out_sm))
                pairs += 1
    return np.mean(tanis) if tanis else 0.0, pairs