from utils.utils import *

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

