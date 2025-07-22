from utils.dataloader import dataset
from utils.utils import *
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

