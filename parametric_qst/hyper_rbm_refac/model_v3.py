import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union

# ==============================================================================
# SECTION 1: ARCHITECTURE
# ==============================================================================

class Hypernetwork(nn.Module):
    def __init__(self, num_v: int, num_h: int, cond_dim: int, hyper_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, hyper_dim)
        self.fc2 = nn.Linear(hyper_dim, 2 * (num_v + num_h))
        self.output_slicing = [num_v, num_v, num_h, num_h]

    def forward(self, cond: torch.Tensor):
        x = torch.tanh(self.fc1(cond))
        x = self.fc2(x)
        return torch.split(x, self.output_slicing, dim=-1)

class SymmetricHyperRBM(nn.Module):
    def __init__(self, num_v: int, num_h: int, hyper_dim: int = 64, k: int = 10):
        super().__init__()
        self.num_v = num_v
        self.num_h = num_h
        self.k = k

        # we fix T=1.0 as physical reference point for the energy definition and also for training
        self.T = 1.0

        self.W = nn.Parameter(torch.empty(num_v, num_h))
        self.b = nn.Parameter(torch.zeros(num_v))
        self.c = nn.Parameter(torch.zeros(num_h))

        # assuming cond_dim = 1 avoids extra complexity throughout the model
        self.conditioner = Hypernetwork(num_v, num_h, 1, hyper_dim)

        self.initialize_weights()

    def initialize_weights(self, std: float = 0.01):
        nn.init.normal_(self.W, std=std)
        nn.init.constant_(self.b, 0.0)
        nn.init.constant_(self.c, 0.0)

    def _compute_mod_biases(self, cond: torch.Tensor):
        gamma_b, beta_b, gamma_c, beta_c = self.conditioner(cond)

        b_mod = (1.0 + gamma_b) * self.b.unsqueeze(0) + beta_b
        c_mod = (1.0 + gamma_c) * self.c.unsqueeze(0) + beta_c

        return b_mod, c_mod

    def _free_energy(self, v: torch.Tensor, b_mod: torch.Tensor, c_mod: torch.Tensor) -> torch.Tensor:
        v = v.to(dtype=self.W.dtype, device=self.W.device)

        # precompute shared terms
        vW = v @ self.W                  # (B, H)
        W_colsum = self.W.sum(dim=0)     # (H,)

        # calculate the free energies for both branches (u=1 and u=0)
        preact_normal  = vW + c_mod
        preact_flipped = (W_colsum.unsqueeze(0) - vW) + c_mod
        sp_normal  = F.softplus(preact_normal).sum(dim=-1)
        sp_flipped = F.softplus(preact_flipped).sum(dim=-1)
        vb_normal  = -(v * b_mod).sum(dim=-1)
        vb_flipped = -((1.0 - v) * b_mod).sum(dim=-1)
        F_normal  = vb_normal  - sp_normal
        F_flipped = vb_flipped - sp_flipped

        # symmetrize by summing branch weights exp(-F/T) in log-space. can be done via logsumexp if we stack sum terms
        negF = torch.stack([-F_normal, -F_flipped], dim=-1)
        return -self.T * torch.logsumexp(negF / self.T, dim=-1)

    def _gibbs_step(self, v, h, u, b_mod, c_mod, rng: Optional[torch.Generator], T: float):
        # pick in which branch we are currently in by flipping v using a branchless conditional selection
        v_branch = u * v + (1.0 - u) * (1.0 - v)

        # step 1: h | v, u -> sample h given v in the selected branch
        p_h = torch.sigmoid((v_branch @ self.W + c_mod) / T)
        h = torch.bernoulli(p_h, generator=rng)

        # precompute shared term
        a = h @ self.W.t()

        # step 2: u | v, h -> compute energy difference between branches, then sample new u
        vb = (v * b_mod).sum(dim=-1)         # v^T b
        va = (v * a).sum(dim=-1)             # v^T (W h)
        bsum = b_mod.sum(dim=-1)             # 1^T b
        asum = a.sum(dim=-1)                 # 1^T (W h)

        dE = (-bsum - asum + 2.0 * vb + 2.0 * va)
        p_u = torch.sigmoid(dE / T)
        u = torch.bernoulli(p_u, generator=rng).to(v.dtype).unsqueeze(-1)

        # step 3: v | h, u -> sample v in the selected branch and update v according to the new u
        p_v_branch = torch.sigmoid((a + b_mod) / T)
        v_branch_new = torch.bernoulli(p_v_branch, generator=rng)

        v_next = u * v_branch_new + (1.0 - u) * (1.0 - v_branch_new)

        return v_next, h, u

    def log_score(self, v: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b_mod, c_mod = self._compute_mod_biases(cond)
        return -0.5 * self._free_energy(v, b_mod, c_mod) / self.T

    def forward(self, batch: Tuple[torch.Tensor, ...], aux_vars: Dict[str, Any]):
        """
        Training step. aux_vars must contain 'rng'.
        """
        rng = aux_vars.get("rng")

        v_data, _, cond = batch
        v_data = v_data.to(device=self.W.device, dtype=self.W.dtype)
        cond = cond.to(device=self.W.device, dtype=self.W.dtype)

        b_mod, c_mod = self._compute_mod_biases(cond)
        v_model = v_data.clone()

        # randomly reset a subset of chains to improve mixing
        noise_frac = aux_vars.get("noise_frac", 0.0)
        num_reset_chains = int(v_data.size(0) * noise_frac)

        if num_reset_chains > 0:
            v_model[:num_reset_chains].bernoulli_(0.5, generator=rng)

        B = v_model.size(0)

        # starting the u chain with ones ensures that we don't flip our CD chain immediately away from the data
        # let's try to form one well first before trying to mirror it
        u = torch.ones((B, 1), device=v_model.device, dtype=v_model.dtype)

        h = torch.zeros((B, self.num_h), device=v_model.device, dtype=v_model.dtype)

        for _ in range(self.k):
            v_model, h, u = self._gibbs_step(v_model, h, u, b_mod, c_mod, rng, T=self.T)
        v_model = v_model.detach()

        loss = self._free_energy(v_data, b_mod, c_mod).mean() - \
               self._free_energy(v_model, b_mod, c_mod).mean()

        return loss, {}

    # the generate function overrides T=1.0 with a user-defined annealing schedule
    @torch.no_grad()
    def generate(self, cond_batch: torch.Tensor, T_schedule: torch.Tensor, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        device = self.W.device
        dtype = self.W.dtype
        cond_batch = cond_batch.to(device=device, dtype=dtype)

        if cond_batch.dim() == 1:
            cond_batch = cond_batch.unsqueeze(0)

        B = cond_batch.shape[0]
        b_mod, c_mod = self._compute_mod_biases(cond_batch)

        v = torch.bernoulli(torch.full((B, self.num_v), 0.5, device=device, dtype=dtype), generator=rng)
        h = torch.zeros((B, self.num_h), device=device, dtype=dtype)

        # compared to the CD training, we now want to start from noise explicitly
        u = torch.bernoulli(torch.full((B, 1), 0.5, device=device, dtype=dtype), generator=rng)

        for T_val in T_schedule:
            v, h, u = self._gibbs_step(v, h, u, b_mod, c_mod, rng, T=float(T_val))

        return v


# ==============================================================================
# SECTION 2: TRAINING (training.py)
# ==============================================================================

def get_sigmoid_curve(high, low, steps, falloff, center=None):
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
            optimizer.param_groups[0]["lr"] = lr        # we use a shared parameter group (0) over all parameters

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
        model: SymmetricHyperRBM,
        config: Dict[str, Any],
        results: list,
        path: Path,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    cfg = dict(config) if config is not None else {}  # don't mutate caller
    cfg.update({
        "num_visible": model.num_v,
        "num_hidden": model.num_h,
        "k": model.k,
        "conditioner_width": model.conditioner.fc1.out_features,
        "T": float(model.T),
    })

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "results": results,
        },
        path,
    )
    print(f"Model saved to: {path}")


def load_model(path: Path, device: torch.device) -> Tuple[SymmetricHyperRBM, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")

    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint["config"]

    k_val = cfg.get("k", cfg.get("k_steps", 10))

    model = SymmetricHyperRBM(
        num_v=cfg["num_visible"],
        num_h=cfg["num_hidden"],
        hyper_dim=cfg.get("conditioner_width", 64),
        k=k_val,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.T = float(cfg.get("T", 1.0))
    model.eval()

    return model, cfg


# ==============================================================================
# SECTION 4: METRICS (metrics.py)
# ==============================================================================

def generate_basis_states(num_qubits: int, device: torch.device) -> torch.Tensor:
    lst = list(itertools.product([0, 1], repeat=num_qubits))
    return torch.tensor(lst, dtype=torch.float32, device=device)

@torch.no_grad()
def get_normalized_wavefunction(model: SymmetricHyperRBM, cond_batch: torch.Tensor, basis_states: torch.Tensor) -> torch.Tensor:

    if cond_batch.dim() == 1:
        cond_batch = cond_batch.unsqueeze(0)

    cond_exp = cond_batch.expand(basis_states.shape[0], -1)

    log_psi = model.log_score(basis_states, cond_exp)

    log_norm_sq = torch.logsumexp(2.0 * log_psi, dim=0)
    return torch.exp(log_psi - 0.5 * log_norm_sq)

@torch.no_grad()
def calculate_exact_overlap(model: SymmetricHyperRBM, cond_val: float,
                            psi_true: torch.Tensor, all_states: torch.Tensor) -> float:

    device = all_states.device

    psi_true = psi_true.real.float().to(device)
    psi_true = psi_true / torch.norm(psi_true)

    cond_batch = torch.tensor([cond_val], device=device, dtype=torch.float32)
    psi_model = get_normalized_wavefunction(model, cond_batch, all_states)

    return torch.abs(torch.dot(psi_true, psi_model)).item()