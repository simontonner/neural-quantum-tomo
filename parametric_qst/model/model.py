import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# ==============================================================================
# SECTION 1: ARCHITECTURE (conditional_rbm.py)
# ==============================================================================

class Conditioner(nn.Module):
    def __init__(self, num_visible: int, num_hidden: int, cond_dim: int, hidden_width: int):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, 2 * (num_visible + num_hidden))
        self.num_visible = num_visible
        self.num_hidden = num_hidden

    def forward(self, cond: torch.Tensor):
        x = torch.tanh(self.fc1(cond))
        x = self.fc2(x)
        return torch.split(
            x,
            [self.num_visible, self.num_visible, self.num_hidden, self.num_hidden],
            dim=-1
        )

class ConditionalRBM(nn.Module):
    def __init__(
            self,
            num_visible: int,
            num_hidden: int,
            cond_dim: int = 1,
            conditioner_width: int = 64,
            k: int = 10,
            T: float = 1.0
    ):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.T = T

        self.W = nn.Parameter(torch.empty(num_visible, num_hidden))
        self.b = nn.Parameter(torch.zeros(num_visible))
        self.c = nn.Parameter(torch.zeros(num_hidden))

        self.conditioner = Conditioner(num_visible, num_hidden, cond_dim, conditioner_width)

        # Default initialization (weak)
        self.initialize_weights(std=0.01)

    def initialize_weights(self, std: float = 0.01):
        """
        Public method to re-initialize weights.
        Useful for tuning noise levels (e.g. 0.01 vs 0.05).
        """
        nn.init.normal_(self.W, std=std)
        nn.init.constant_(self.b, 0.0)
        nn.init.constant_(self.c, 0.0)

    # ------------------------------------------------------------------
    #  Internal: Energies & Bias
    # ------------------------------------------------------------------
    def _compute_effective_biases(self, cond: torch.Tensor):
        gamma_b, beta_b, gamma_c, beta_c = self.conditioner(cond)
        if cond.dim() == 1:
            b_mod = (1.0 + gamma_b) * self.b + beta_b
            c_mod = (1.0 + gamma_c) * self.c + beta_c
        else:
            b_mod = (1.0 + gamma_b) * self.b.unsqueeze(0) + beta_b
            c_mod = (1.0 + gamma_c) * self.c.unsqueeze(0) + beta_c
        return b_mod, c_mod

    def _free_energy(self, v: torch.Tensor, b_mod: torch.Tensor, c_mod: torch.Tensor) -> torch.Tensor:
        v = v.to(dtype=self.W.dtype, device=self.W.device)
        v_W = v @ self.W
        W_sum = self.W.sum(dim=0)

        linear_v = v_W + c_mod
        linear_f = W_sum.unsqueeze(0) - v_W + c_mod

        # F(v) and F(1-v)
        term2_v = F.softplus(linear_v).sum(dim=-1)
        term2_f = F.softplus(linear_f).sum(dim=-1)
        term1_v = -(v * b_mod).sum(dim=-1)
        term1_f = -((1.0 - v) * b_mod).sum(dim=-1)

        F_v = term1_v - term2_v
        F_f = term1_f - term2_f

        stacked = torch.stack([-F_v, -F_f], dim=-1)
        return -self.T * torch.logsumexp(stacked / self.T, dim=-1)

    # ------------------------------------------------------------------
    #  Internal: Augmented Gibbs Sampling
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_flip(v: torch.Tensor, s0: torch.Tensor) -> torch.Tensor:
        return s0 * v + (1.0 - s0) * (1.0 - v)

    def _gibbs_step_sym_fast(self, v, h, s0, b_mod, c_mod, rng: Optional[torch.Generator]):
        if b_mod.dim() == 1: b_mod = b_mod.unsqueeze(0).expand(v.size(0), -1)
        if c_mod.dim() == 1: c_mod = c_mod.unsqueeze(0).expand(v.size(0), -1)

        # 1. Sample h | v, s
        v_eff = self._apply_flip(v, s0)
        p_h = torch.sigmoid((v_eff @ self.W + c_mod) / self.T)
        h = torch.bernoulli(p_h, generator=rng)

        # Reuse a = Wh
        a = h @ self.W.t()

        # 2. Sample s | v, h
        vb   = (v * b_mod).sum(dim=-1)
        va   = (v * a).sum(dim=-1)
        bsum = b_mod.sum(dim=-1)
        asum = a.sum(dim=-1)
        dE = (-bsum - asum + 2.0 * vb + 2.0 * va)
        p_s0 = torch.sigmoid(dE / self.T)
        s0 = torch.bernoulli(p_s0, generator=rng).to(v.dtype).unsqueeze(-1)

        # 3. Sample v | h, s
        p_v = torch.sigmoid((a + b_mod) / self.T)
        v_eff = torch.bernoulli(p_v, generator=rng)
        v_next = self._apply_flip(v_eff, s0)

        return v_next, h, s0

    # ------------------------------------------------------------------
    #  Public: Training & Scoring
    # ------------------------------------------------------------------
    def log_score(self, v: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b_mod, c_mod = self._compute_effective_biases(cond)
        return -0.5 * self._free_energy(v, b_mod, c_mod) / self.T

    def forward(self, batch: Tuple[torch.Tensor, ...], aux_vars: Dict[str, Any]):
        """
        Training step. aux_vars must contain 'rng'.
        """
        v_data, _, cond = batch
        v_data = v_data.to(device=self.W.device, dtype=self.W.dtype)
        cond = cond.to(device=self.W.device, dtype=self.W.dtype)
        rng = aux_vars.get("rng")

        b_mod, c_mod = self._compute_effective_biases(cond)
        v_model = v_data.clone()

        # Noise Injection
        noise_frac = aux_vars.get("noise_frac", 0.0)
        if noise_frac > 0:
            n_noise = int(v_data.shape[0] * noise_frac)
            if n_noise > 0:
                v_model[:n_noise] = torch.bernoulli(
                    torch.full_like(v_model[:n_noise], 0.5), generator=rng
                )

        B = v_model.size(0)
        s0 = torch.bernoulli(torch.full((B, 1), 0.5, device=v_model.device, dtype=v_model.dtype), generator=rng)
        h = torch.zeros((B, self.num_hidden), device=v_model.device, dtype=v_model.dtype)

        for _ in range(self.k):
            v_model, h, s0 = self._gibbs_step_sym_fast(v_model, h, s0, b_mod, c_mod, rng)
        v_model = v_model.detach()

        loss = self._free_energy(v_data, b_mod, c_mod).mean() - \
               self._free_energy(v_model, b_mod, c_mod).mean()

        return loss, {}

    # ------------------------------------------------------------------
    #  Public: Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
            self,
            cond: torch.Tensor,
            schedule: torch.Tensor,
            n_samples: Optional[int] = None,
            rng: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Generate samples using a pre-defined temperature schedule.

        Args:
            cond: Conditioning values.
            schedule: 1D Tensor [T_0, T_1, ... T_steps].
            n_samples: If set, expands cond to this size.
            rng: Torch generator (Last argument).
        """
        device = self.W.device
        dtype = self.W.dtype
        cond = cond.to(device=device, dtype=dtype)

        if n_samples is not None:
            if cond.dim() == 1: cond = cond.expand(n_samples, -1)
            elif cond.shape[0] == 1: cond = cond.expand(n_samples, -1)

        B = cond.shape[0]
        b_mod, c_mod = self._compute_effective_biases(cond)

        # Random Start
        v = torch.bernoulli(torch.full((B, self.num_visible), 0.5, device=device, dtype=dtype), generator=rng)
        h = torch.zeros((B, self.num_hidden), device=device, dtype=dtype)
        s0 = torch.bernoulli(torch.full((B, 1), 0.5, device=device, dtype=dtype), generator=rng)

        # Schedule Loop
        original_T = self.T
        for t_val in schedule:
            self.T = float(t_val)
            v, h, s0 = self._gibbs_step_sym_fast(v, h, s0, b_mod, c_mod, rng)

        self.T = original_T
        return v

    @torch.no_grad()
    def get_normalized_wavefunction(self, cond: torch.Tensor, all_states: torch.Tensor) -> torch.Tensor:
        if cond.dim() == 1: cond = cond.unsqueeze(0)
        cond_exp = cond.expand(all_states.shape[0], -1)

        old_T = self.T
        self.T = 1.0
        log_psi = self.log_score(all_states, cond_exp)
        self.T = old_T

        log_norm_sq = torch.logsumexp(2.0 * log_psi, dim=0)
        return torch.exp(log_psi - 0.5 * log_norm_sq)


# ==============================================================================
# SECTION 2: TRAINING (training.py)
# ==============================================================================

def get_sigmoid_curve(high, low, steps, falloff, center=None):
    """
    Returns a function f(step) -> value.
    Useful for LR scheduling or constructing manual annealing tensors.
    """
    if center is None: center = steps / 2.0
    def fn(step):
        s = min(step, steps)
        return float(low + (high - low) / (1.0 + math.exp(falloff * (s - center))))
    return fn

def train_loop(
        model,
        optimizer: Optimizer,
        loader,
        num_epochs: int,
        lr_schedule_fn,
        noise_frac: float = 0.1,
        rng: Optional[torch.Generator] = None
):
    global_step = 0
    model.train()

    print(f"{'Epoch':<6} | {'Loss':<10} | {'LR':<10}")
    print("-" * 35)

    for epoch in range(num_epochs):
        tot_loss = 0.0
        for batch in loader:
            lr = lr_schedule_fn(global_step)
            for g in optimizer.param_groups: g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(batch, {"rng": rng, "noise_frac": noise_frac})
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            global_step += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = tot_loss / len(loader)
            print(f"{epoch+1:<6} | {avg_loss:+.4f}     | {lr:.6f}")

    return model


# ==============================================================================
# SECTION 3: IO (io.py)
# ==============================================================================

def save_model(
        model: ConditionalRBM,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        results: list,
        path: Path
):
    path.parent.mkdir(parents=True, exist_ok=True)

    config.update({
        "num_visible": model.num_visible,
        "num_hidden": model.num_hidden,
        "k": model.k
    })

    save_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "results": results
    }
    torch.save(save_dict, path)
    print(f"Model saved to: {path}")

def load_model(path: Path, device: torch.device) -> Tuple[ConditionalRBM, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")

    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint["config"]

    # Handle legacy config keys
    k_val = cfg.get("k", cfg.get("k_steps", 10))

    model = ConditionalRBM(
        num_visible=cfg["num_visible"],
        num_hidden=cfg["num_hidden"],
        cond_dim=cfg.get("cond_dim", 1),
        conditioner_width=cfg.get("conditioner_width", 64),
        k=k_val,
        T=cfg.get("T", 1.0)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, cfg