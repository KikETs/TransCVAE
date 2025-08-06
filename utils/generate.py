from utils.utils import *
from torch.nn.utils.rnn import pad_sequence
def generate_batch_sequence(
    model,
    z,
    max_length: int,
    start_token: int,
    end_token: int,
    pad_token: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str = "cuda",
):
    """
    Autoregressive generation that *keeps* the whole history.
    ※ z 길이(encoder memory)는 건드리지 않는다.
    """

    model.eval()

    B = z.size(0)
    z = z.to(device)

    # 각 배치마다 시퀀스 초기화
    generated: list[list[int]] = [[start_token] for _ in range(B)]
    finished = [False] * B

    for _ in range(max_length):
        # ── 1) padding-right 로 맞춰서 텐서화 ─────────────────────────
        cur_len = max(len(seq) for seq in generated)
        seq_tensor = torch.full(
            (B, cur_len), pad_token, dtype=torch.long, device=device
        )
        seq_lens = []
        for i, seq in enumerate(generated):
            seq_lens.append(len(seq))
            seq_tensor[i, : len(seq)] = torch.tensor(seq, device=device)


        # ── 3) forward ──────────────────────────────────────────────
        with torch.no_grad():
            logits = model.decoder(
                seq_tensor,
                z,  # z 길이 그대로
            )
            logits = model.predict(logits) / temperature  # [B, cur_len, V]

        # ── 4) 마지막 위치의 로짓만 뽑아서 샘플링 ────────────────────
        last_pos = torch.tensor([l - 1 for l in seq_lens], device=device)
        next_logits = logits[torch.arange(B, device=device), last_pos]  # [B, V]

        # ··· top-k · top-p 필터링(Optional) ···
        if top_k is not None:
            topk_val, _ = torch.topk(next_logits, top_k)
            threshold = topk_val[:, -1, None]
            next_logits[next_logits < threshold] = -float("inf")

        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(next_logits, dim=-1, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative = probs.cumsum(dim=-1)
            mask = cumulative > top_p
            mask[..., 0] = False
            sorted_logits[mask] = -float("inf")
            next_logits.scatter_(1, sorted_idx, sorted_logits)

        probs = F.softmax(next_logits, dim=-1)

        # ── 5) 토큰 선택 & 종료 체크 ─────────────────────────────────
        all_done = True
        for i in range(B):
            if finished[i]:
                continue
            tok = torch.multinomial(probs[i], 1).item()
            generated[i].append(tok)
            if tok == end_token:
                finished[i] = True
            else:
                all_done = False

        if all_done:
            break

    return generated

@torch.no_grad()
def generate_batch_sequence_LSTM(
    model,
    z,                           # [B, latent_dim]
    max_length: int,
    start_token: int,
    end_token: int,
    pad_token: int | None = None,
    grammar_mask_fn=None,
    index_to_token=None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str = "cuda",
):
    """
    LSTM-CVAE 배치 생성 (시퀀스 누적 방식)

    * hidden-state를 이어받으면서도, 매 스텝마다
      지금까지 만든 전체 시퀀스(패딩 포함)를 LSTM에 넣어도 OK.
    * top-k / top-p / temperature 지원.
    """

    model = model.to(device).eval()
    B     = z.size(0)
    z     = z.to(device)

    # ── 1) z → (h0, c0) 초기화 ─────────────────────────────────────
    h0_raw, c0_raw = torch.tanh(model.to_decoder(z)).chunk(2, dim=-1)  # [B,d]
    num_layers     = model.decoder.num_layers
    h = h0_raw.unsqueeze(0).repeat(num_layers, 1, 1).contiguous()
    c = c0_raw.unsqueeze(0).repeat(num_layers, 1, 1).contiguous()

    # ── 2) 시퀀스 시작 ────────────────────────────────────────────
    generated = [[start_token] for _ in range(B)]
    finished  = [False] * B

    # ── 3) 오토리그레시브 루프 ──────────────────────────────────
    for _ in range(max_length - 1):  # <SOS> 포함했으니 −1
        # (a) 배치에서 가장 긴 길이 기준으로 패딩-오른쪽 정렬
        cur_len = max(len(seq) for seq in generated)
        seq_tensor = torch.full(
            (B, cur_len), pad_token, dtype=torch.long, device=device
        )
        for i, seq in enumerate(generated):
            seq_tensor[i, : len(seq)] = torch.tensor(seq, device=device)

        # (b) 토큰 임베딩 & LSTM 통과
        emb = model.smiles_embbed(seq_tensor)           # [B, cur_len, d]
        out, (h, c) = model.decoder(emb, (h, c))        # hidden 이어받음
        logits      = model.predict(out[:, -1]) / temperature  # 마지막 토큰만

        # (c) 배치별 next-token 결정
        next_tok = []
        for i in range(B):
            if finished[i]:
                next_tok.append(pad_token)
                continue

            logit_i = logits[i]

            # grammar mask (선택)
            if grammar_mask_fn is not None:
                mask = grammar_mask_fn(
                    generated[i], index_to_token, logit_i.size(0)
                ).to(device)
                logit_i = logit_i.masked_fill(~mask, -1e9)

            # top-k / top-p 필터링
            if top_k is not None:
                kth, _ = torch.topk(logit_i, top_k)
                logit_i[logit_i < kth[-1]] = -float("inf")
            if top_p is not None:
                sorted_l, sorted_idx = torch.sort(logit_i, descending=True)
                probs = F.softmax(sorted_l, dim=-1)
                cum_p = probs.cumsum(dim=-1)
                mask  = cum_p > top_p
                mask[0] = False
                sorted_l[mask] = -float("inf")
                logit_i.scatter_(0, sorted_idx, sorted_l)

            probs_i = F.softmax(logit_i, dim=-1)
            tok     = torch.multinomial(probs_i, 1).item()

            generated[i].append(tok)
            next_tok.append(tok)

            if tok == end_token:
                finished[i] = True

        if all(finished):
            break

    # ── 4) 후처리 : 패딩 길이 맞추기 (선택) ─────────────────────────
    if pad_token is not None:
        final_len = max(len(seq) for seq in generated)
        generated = [
            seq + [pad_token] * (final_len - len(seq)) for seq in generated
        ]

    return generated

@torch.no_grad()
def generate_batch_sequence_LSTM_MHA(
    model,
    z,                                 # z 외에는 전부 키워드 전용
    max_length: int,
    start_token: int,
    end_token: int,
    pad_token: int,
    grammar_mask_fn = None,
    index_to_token  = None,
    temperature: float   = 1.0,
    top_k: int | None    = None,
    top_p: float | None  = None,
    device: str          = "cuda",
    return_tensor: bool  = False,
):
    """
    LSTM-MHA 디코더용 배치 생성 (시퀀스 누적 방식)

    - hidden state 를 이어받으면서도, 매 스텝마다
      지금까지 만든 시퀀스 전체(패딩 포함)를 디코더에 넣습니다.
    - top-k, top-p, temperature 조절 지원.
    """

    model = model.to(device).eval()

    # ── 0. z 전처리 ──────────────────────────────────────────────
    if z.dim() == 3:                      # [B,L,z] 이면 평균
        z_vec = z.mean(1)
    elif z.dim() == 2:                    # [B,z]
        z_vec = z
    else:
        raise ValueError("z must be [B,z] or [B,L,z]")

    B     = z_vec.size(0)
    z_vec = z_vec.to(device)

    # ── 1. (h0,c0) 초기화 ───────────────────────────────────────
    h0_raw, c0_raw = torch.tanh(model.to_decoder(z_vec)).chunk(2, dim=-1)
    n_layers       = model.decoder.num_layers
    h = h0_raw.unsqueeze(0).repeat(n_layers, 1, 1).contiguous()
    c = c0_raw.unsqueeze(0).repeat(n_layers, 1, 1).contiguous()

    # ── 2. 시퀀스 시작 ───────────────────────────────────────────
    generated = [[start_token] for _ in range(B)]
    finished  = [False] * B

    # ── 3. 오토리그레시브 루프 ──────────────────────────────────
    for _ in range(max_length - 1):            # <SOS> 포함했으니 −1
        # (a) 오른쪽-패딩 텐서화
        cur_len = max(len(seq) for seq in generated)
        seq_tensor = torch.full(
            (B, cur_len), pad_token, dtype=torch.long, device=device
        )
        for i, seq in enumerate(generated):
            seq_tensor[i, : len(seq)] = torch.tensor(seq, device=device)

        # (b) 임베딩 → 디코더
        emb = model.smiles_embbed(seq_tensor)         # [B,cur_len,E]
        out, (h, c) = model.decoder(emb, (h, c))      # hidden 이어받음
        logits = model.predict(out[:, -1]) / temperature   # [B,V] (마지막 토큰)

        next_tok = []
        for i in range(B):
            if finished[i]:
                next_tok.append(pad_token)
                continue

            log_i = logits[i]

            # grammar mask(선택)
            if grammar_mask_fn is not None:
                mask = grammar_mask_fn(
                    generated[i], index_to_token, log_i.size(0)
                ).to(device)
                log_i = log_i.masked_fill(~mask, -1e9)

            # top-k / top-p 필터링
            if top_k is not None:
                kth, _ = torch.topk(log_i, top_k)
                log_i[log_i < kth[-1]] = -float("inf")
            if top_p is not None:
                sorted_l, sorted_idx = torch.sort(log_i, descending=True)
                probs = F.softmax(sorted_l, dim=-1)
                cum_p = probs.cumsum(dim=-1)
                mask = cum_p > top_p
                mask[0] = False
                sorted_l[mask] = -float("inf")
                log_i.scatter_(0, sorted_idx, sorted_l)

            tok = torch.multinomial(F.softmax(log_i, dim=-1), 1).item()
            generated[i].append(tok)
            next_tok.append(tok)

            if tok == end_token:
                finished[i] = True

        if all(finished):
            break

    # ── 4. 결과 반환 ────────────────────────────────────────────
    if return_tensor:
        return pad_sequence(
            [torch.tensor(seq, device=device) for seq in generated],
            batch_first=True,
            padding_value=pad_token,
        )
    return generated