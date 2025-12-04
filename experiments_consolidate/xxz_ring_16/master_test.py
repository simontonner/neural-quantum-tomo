import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

# === PATH SETUP ===
sys.path.append(str(Path("..").resolve()))
from data_handling import load_measurements_npz, MeasurementDataset, MeasurementLoader

data_dir = Path("measurements")

# ==========================================
# 1. RBM MODEL
# ==========================================
class RBM(nn.Module):
    def __init__(self, num_visible: int, num_hidden: int, k: int, T: float = 1.0):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.T = T

        self.W = nn.Parameter(torch.empty(num_visible, num_hidden))
        self.b = nn.Parameter(torch.zeros(num_visible))
        self.c = nn.Parameter(torch.zeros(num_hidden))

        # "Goldilocks" Initialization (0.05)
        self.initialize_weights(w_std=0.05)

    def initialize_weights(self, w_mean: float = 0.0, w_std: float = 0.05, bias_val: float = 0.0):
        nn.init.normal_(self.W, mean=w_mean, std=w_std)
        nn.init.constant_(self.b, bias_val)
        nn.init.constant_(self.c, bias_val)

    @staticmethod
    def _free_energy(v: torch.Tensor, W: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        v = v.to(dtype=W.dtype, device=W.device)
        term1 = -(v * b).sum(dim=-1)
        term2 = F.softplus(v @ W + c).sum(dim=-1)
        return term1 - term2

    def _gibbs_step(self, v: torch.Tensor, T: float, rng: torch.Generator):
        p_h = torch.sigmoid((v @ self.W + self.c) / T)
        h = torch.bernoulli(p_h, generator=rng)
        p_v = torch.sigmoid((h @ self.W.t() + self.b) / T)
        v_next = torch.bernoulli(p_v, generator=rng)
        return v_next

    def log_score(self, v: torch.Tensor) -> torch.Tensor:
        # log(Psi) = -0.5 * FreeEnergy
        fe = self._free_energy(v, self.W, self.b, self.c)
        return -0.5 * fe / self.T

    def forward(self, batch: Tuple[torch.Tensor, ...], aux_vars: Dict[str, Any]):
        values, _, _ = batch
        v_data = values.to(dtype=self.W.dtype, device=self.W.device)
        rng = aux_vars.get("rng", torch.Generator(device="cpu"))

        # CD-k: Start from Data
        v_model = v_data.clone()
        for _ in range(self.k):
            v_model = self._gibbs_step(v_model, self.T, rng)
        v_model = v_model.detach()

        # Loss
        fe_data = self._free_energy(v_data, self.W, self.b, self.c)
        fe_model = self._free_energy(v_model, self.W, self.b, self.c)
        loss = (fe_data - fe_model).mean()

        # Monitoring
        n_flips = (v_data != v_model).float().sum()
        flip_rate = n_flips / v_data.numel()

        return loss, {"flip_rate": flip_rate}

    @torch.no_grad()
    def generate(self, n_samples: int, T_schedule: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
        device = next(self.parameters()).device
        probs = torch.full((n_samples, self.num_visible), 0.5, device=device, dtype=torch.float32)
        v = torch.bernoulli(probs, generator=rng)

        if T_schedule.dim() == 0: T_schedule = T_schedule.view(1)
        for i in range(int(T_schedule.shape[0])):
            v = self._gibbs_step(v, float(T_schedule[i]), rng)
        return v

    @torch.no_grad()
    def get_full_psi(self):
        """Normally too expensive, but feasible for small N."""
        device = next(self.parameters()).device
        N = self.num_visible

        # Generate all 2^N states
        indices = torch.arange(2**N, device=device).unsqueeze(1)
        powers = 2**torch.arange(N - 1, -1, -1, device=device)
        v_all = (indices.bitwise_and(powers) != 0).float()

        # Calculate Psi
        fe = self._free_energy(v_all, self.W, self.b, self.c)
        log_psi_unnorm = -0.5 * fe

        # Normalize
        log_Z = torch.logsumexp(-fe, dim=0)
        psi = torch.exp(log_psi_unnorm - 0.5 * log_Z)
        return psi

# ==========================================
# METHOD A: EXACT SVD
# ==========================================
def run_exact_svd(model: RBM, N: int):
    print("Running Method A: Exact SVD...")
    t0 = time.time()

    psi_vector = model.get_full_psi()
    results = []

    for l in range(1, N // 2 + 1):
        dim_A = 2**l
        dim_B = 2**(N - l)

        # Schmidt Decomposition
        psi_matrix = psi_vector.view(dim_A, dim_B)

        try:
            S = torch.linalg.svdvals(psi_matrix)
        except:
            S = torch.linalg.svdvals(psi_matrix.cpu())

        # S2 = -ln(Tr(rho^2)) = -ln(Sum(lambda^4))
        trace_rho_sq = torch.sum(S**4).item()
        s2 = -math.log(trace_rho_sq)
        results.append(s2)

    dt = time.time() - t0
    return results, dt

# ==========================================
# METHOD B: VANILLA MONTE CARLO SWAP
# ==========================================
def compute_renyi_entropy_vanilla_mc(
        samples: torch.Tensor,
        subsystem_size: int,
        score_fn: Callable[[torch.Tensor], torch.Tensor]
) -> float:
    """
    Computes S2 = -ln(<Swap>) using the vanilla replica trick.
    (Score function returns log_psi)
    """
    n_samples = samples.shape[0]
    half = n_samples // 2
    if half == 0: return 0.0

    batch1 = samples[:half]
    batch2 = samples[half:2 * half]
    idx_B = torch.arange(subsystem_size, samples.shape[1], device=samples.device)

    # 1. Score Original
    log_psi_1 = score_fn(batch1)
    log_psi_2 = score_fn(batch2)

    # 2. Swap Indices
    batch1_swap = batch1.clone()
    batch1_swap[:, idx_B] = batch2[:, idx_B]
    batch2_swap = batch2.clone()
    batch2_swap[:, idx_B] = batch1[:, idx_B]

    # 3. Score Swapped
    log_psi_1_swap = score_fn(batch1_swap)
    log_psi_2_swap = score_fn(batch2_swap)

    # 4. Calculate Ratio
    # Ratio = (Psi(s1')Psi(s2')) / (Psi(s1)Psi(s2))
    log_ratio = (log_psi_1_swap + log_psi_2_swap) - (log_psi_1 + log_psi_2)

    # Clip for numerical safety
    log_ratio = torch.clamp(log_ratio, max=30.0)

    ratios = torch.exp(log_ratio)
    swap_expectation = ratios.mean().item()

    if swap_expectation <= 1e-12: return 0.0
    return -math.log(swap_expectation)

def run_vanilla_mc_swap(model: RBM, N: int, n_samples: int = 50000):
    print("Running Method B: Vanilla MC Swap...")
    t0 = time.time()

    # 1. Generate Samples (Long chain for good thermalization)
    T_schedule = torch.cat([torch.linspace(2.0, 1.0, 50), torch.ones(150)]).to(next(model.parameters()).device)
    rng = torch.Generator().manual_seed(123)

    with torch.no_grad():
        samples = model.generate(n_samples, T_schedule, rng)

    scorer = lambda v: model.log_score(v)
    results = []

    # 2. Compute Entropy for each L
    for l in range(1, N // 2 + 1):
        s2 = compute_renyi_entropy_vanilla_mc(samples, l, scorer)
        results.append(s2)

    dt = time.time() - t0
    return results, dt

# ==========================================
# UTILS
# ==========================================
def compute_nearest_neighbor_corr(model, rng):
    with torch.no_grad():
        gen = model.generate(2000, torch.ones(5, device=next(model.parameters()).device), rng)
        spins = 2.0 * gen.float() - 1.0
        return (spins[:, :-1] * spins[:, 1:]).mean().item()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":

    # === CONFIG (Anti-Freeze + Structure) ===
    CHAIN_LENGTH = 16
    TARGET_DELTA = 1.00
    TRAIN_SAMPLES = 500_000

    BATCH_SIZE = 512
    HIDDEN_UNITS = 32
    EPOCHS = 200
    LEARNING_RATE = 0.01     # Low LR to prevent freeze
    WEIGHT_DECAY = 1e-4      # L2 for structure
    CD_K = 30                # Long chains
    # ========================================

    # LOAD DATA
    file_name = f"xxz_{CHAIN_LENGTH}_delta{TARGET_DELTA:.2f}_5000000.npz"
    file_path = data_dir / file_name
    print(f"Loading {file_path}...")
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[TRAIN_SAMPLES])

    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, rng=rng_loader)

    # TRAIN
    SEED = 42
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)

    model = RBM(num_visible=CHAIN_LENGTH, num_hidden=HIDDEN_UNITS, k=CD_K, T=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)

    print("\nTraining Model (Anti-Freeze Config)...")
    t_train = time.time()

    print(f"{'Epoch':<6} | {'Loss':<9} | {'FlipRate':<9} | {'C_zz':<9} | {'LR':<9}")
    print("-" * 55)

    for epoch in range(EPOCHS):
        tot_loss = 0.0
        tot_flip = 0.0

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, aux = model(batch, {"rng": rng})
            loss.backward()
            optimizer.step()
            tot_loss += float(loss)
            tot_flip += float(aux['flip_rate'])

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                curr_czz = compute_nearest_neighbor_corr(model, rng)
            print(f"{epoch + 1:<6} | {tot_loss/len(loader):.4f}    | "
                  f"{tot_flip/len(loader):.4f}    | {curr_czz:.4f}    | {scheduler.get_last_lr()[0]:.4f}")

    print(f"\nTraining Done in {time.time()-t_train:.2f}s")

    # === COMPARISON ===
    print("\n" + "="*40)
    print("COMPARISON: EXACT SVD vs VANILLA MC SWAP")
    print("="*40)

    # Method A: Exact
    svd_curve, svd_time = run_exact_svd(model, CHAIN_LENGTH)

    # Method B: Vanilla MC
    mc_curve, mc_time = run_vanilla_mc_swap(model, CHAIN_LENGTH, n_samples=50000)

    print("-" * 40)
    print(f"{'Method':<15} | {'Time (s)':<10} | {'S2 (L=8)':<10}")
    print("-" * 40)
    print(f"{'Exact SVD':<15} | {svd_time:<10.4f} | {svd_curve[-1]:<10.4f}")
    print(f"{'Vanilla MC':<15} | {mc_time:<10.4f} | {mc_curve[-1]:<10.4f}")
    print("-" * 40)

    # PLOT
    ref_file = Path(f"xxz_{CHAIN_LENGTH}_entropy_ref.csv")
    plt.figure(figsize=(9, 6))

    l_axis = list(range(1, len(svd_curve) + 1))

    if ref_file.exists():
        df_ref = pd.read_csv(ref_file)
        l_cols = sorted([c for c in df_ref.columns if c.startswith("l")], key=lambda s: int(s[1:]))
        mask = np.isclose(df_ref["delta"], TARGET_DELTA)
        if mask.any():
            s2_ref = df_ref.loc[mask].iloc[0][l_cols].to_numpy()
            plt.plot(l_axis, s2_ref, 'k--', linewidth=2, label='Reference')

    plt.plot(l_axis, svd_curve, 'ro-', label=f'Exact SVD')
    plt.plot(l_axis, mc_curve, 'bx:', markersize=10, markeredgewidth=2, label=f'Vanilla MC Swap')

    plt.title(f"Method Comparison (XXZ N={CHAIN_LENGTH}, $\Delta=1.0$)")
    plt.xlabel(r"Subsystem size $\ell$")
    plt.ylabel(r"Rényi Entropy $S_2$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()