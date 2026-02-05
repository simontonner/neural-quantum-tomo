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
    def __init__(self, num_visible: int, num_hidden: int, T: float = 1.0):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.T = T

        self.W = nn.Parameter(torch.empty(num_visible, num_hidden))
        self.b = nn.Parameter(torch.zeros(num_visible))
        self.c = nn.Parameter(torch.zeros(num_hidden))

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
        fe = self._free_energy(v, self.W, self.b, self.c)
        return -0.5 * fe / self.T

    def forward(self, batch: Tuple[torch.Tensor, ...], aux_vars: Dict[str, Any]):
        values, _, _ = batch
        v_data = values.to(dtype=self.W.dtype, device=self.W.device)
        rng = aux_vars.get("rng", torch.Generator(device="cpu"))

        # DYNAMIC K
        k_steps = int(aux_vars.get("k", 10))

        # CD-k: Start from Data
        v_model = v_data.clone()
        for _ in range(k_steps):
            v_model = self._gibbs_step(v_model, self.T, rng)
        v_model = v_model.detach()

        fe_data = self._free_energy(v_data, self.W, self.b, self.c)
        fe_model = self._free_energy(v_model, self.W, self.b, self.c)
        loss = (fe_data - fe_model).mean()

        n_flips = (v_data != v_model).float().sum()
        flip_rate = n_flips / v_data.numel()

        return loss, {"flip_rate": flip_rate}

    @torch.no_grad()
    def get_full_psi(self):
        """Exact generation of full state vector for SVD."""
        device = next(self.parameters()).device
        N = self.num_visible
        indices = torch.arange(2**N, device=device).unsqueeze(1)
        powers = 2**torch.arange(N - 1, -1, -1, device=device)
        v_all = (indices.bitwise_and(powers) != 0).float()

        fe = self._free_energy(v_all, self.W, self.b, self.c)
        log_psi_unnorm = -0.5 * fe
        log_Z = torch.logsumexp(-fe, dim=0)
        psi = torch.exp(log_psi_unnorm - 0.5 * log_Z)
        return psi

    @torch.no_grad()
    def generate(self, n_samples: int, T_schedule: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
        device = next(self.parameters()).device
        probs = torch.full((n_samples, self.num_visible), 0.5, device=device, dtype=torch.float32)
        v = torch.bernoulli(probs, generator=rng)

        if T_schedule.dim() == 0: T_schedule = T_schedule.view(1)
        for i in range(int(T_schedule.shape[0])):
            v = self._gibbs_step(v, float(T_schedule[i]), rng)
        return v

# ==========================================
# EVALUATION (EXACT SVD)
# ==========================================
def compute_entropy_exact_svd(model: RBM, N: int):
    model.eval()
    psi_vector = model.get_full_psi()
    results = []

    for l in range(1, N // 2 + 1):
        dim_A = 2**l
        dim_B = 2**(N - l)
        psi_matrix = psi_vector.view(dim_A, dim_B)

        try:
            S = torch.linalg.svdvals(psi_matrix)
        except:
            S = torch.linalg.svdvals(psi_matrix.cpu())

        trace_rho_sq = torch.sum(S**4).item()
        s2 = -math.log(trace_rho_sq)
        results.append(s2)

    return results

def compute_nearest_neighbor_corr(model, rng):
    # Use short generation for quick check
    gen = model.generate(2000, torch.ones(5, device=next(model.parameters()).device), rng)
    spins = 2.0 * gen.float() - 1.0
    return (spins[:, :-1] * spins[:, 1:]).mean().item()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":

    # CONFIG
    CHAIN_LENGTH = 16
    TARGET_DELTA = 1.00
    TRAIN_SAMPLES = 500_000

    BATCH_SIZE = 512
    HIDDEN_UNITS = 32
    EPOCHS = 200

    # Adaptive Params
    LR_START = 0.05
    LR_END = 0.005
    K_START = 10
    K_END = 100              # Ramp up to 100 steps
    WEIGHT_DECAY = 1e-4

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

    model = RBM(num_visible=CHAIN_LENGTH, num_hidden=HIDDEN_UNITS, T=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_START, momentum=0.9, weight_decay=WEIGHT_DECAY)

    # Cosine Annealing for smooth LR decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_END)

    print(f"\nTraining Model (Adaptive k={K_START}->{K_END})...")
    t_train = time.time()

    print(f"{'Epoch':<6} | {'Loss':<9} | {'FlipRate':<9} | {'C_zz':<9} | {'k-steps':<9}")
    print("-" * 55)

    for epoch in range(EPOCHS):
        # Linearly ramp k
        current_k = int(K_START + (K_END - K_START) * (epoch / EPOCHS))

        tot_loss = 0.0
        tot_flip = 0.0

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            # Pass dynamic k
            loss, aux = model(batch, {"rng": rng, "k": current_k})
            loss.backward()
            optimizer.step()
            tot_loss += float(loss)
            tot_flip += float(aux['flip_rate'])

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                curr_czz = compute_nearest_neighbor_corr(model, rng)
            print(f"{epoch + 1:<6} | {tot_loss/len(loader):.4f}    | "
                  f"{tot_flip/len(loader):.4f}    | {curr_czz:.4f}    | {current_k:<9}")

    print(f"\nTraining Done in {time.time()-t_train:.2f}s")

    # === EXACT SVD EVALUATION ===
    print("\nRunning Exact SVD Evaluation...")
    svd_curve = compute_entropy_exact_svd(model, CHAIN_LENGTH)

    # Print numerical results
    print("-" * 30)
    for l, val in enumerate(svd_curve):
        print(f"l={l+1}: S2={val:.4f}")
    print("-" * 30)

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
            plt.plot(l_axis, s2_ref, 'k--', linewidth=2, label='Exact Reference')

    plt.plot(l_axis, svd_curve, 'ro-', linewidth=2, label=f'RBM (Adaptive k)')

    plt.title(f"Adaptive Training (XXZ N={CHAIN_LENGTH}, $\Delta=1.0$)")
    plt.xlabel(r"Subsystem size $\ell$")
    plt.ylabel(r"Rényi Entropy $S_2$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()