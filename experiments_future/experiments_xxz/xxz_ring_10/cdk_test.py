import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

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
# 1. RBM MODEL (Pure CD Version)
# ==========================================
class RBM(nn.Module):
    def __init__(self, num_visible: int, num_hidden: int, k: int):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.T = 1.0

        self.W = nn.Parameter(torch.empty(num_visible, num_hidden))
        self.b = nn.Parameter(torch.zeros(num_visible))
        self.c = nn.Parameter(torch.zeros(num_hidden))

        self.initialize_weights()

    def initialize_weights(self):
        # Standard Gaussian init
        nn.init.normal_(self.W, mean=0.0, std=0.01)
        nn.init.constant_(self.b, 0.0)
        nn.init.constant_(self.c, 0.0)

    @staticmethod
    def _free_energy(v: torch.Tensor, W: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Free Energy F(v) = -b*v - sum(log(1 + exp(W*v + c)))
        v = v.to(dtype=W.dtype, device=W.device)
        term1 = -(v * b).sum(dim=-1)
        term2 = F.softplus(v @ W + c).sum(dim=-1)
        return term1 - term2

    def _gibbs_step(self, v: torch.Tensor, rng: torch.Generator):

        p_h = torch.sigmoid((v @ self.W + self.c) / self.T)
        h = torch.bernoulli(p_h, generator=rng)

        p_v = torch.sigmoid((h @ self.W.t() + self.b) / self.T)
        v_next = torch.bernoulli(p_v, generator=rng)
        return v_next

    def forward(self, batch: Tuple[torch.Tensor, ...], aux_vars: Dict[str, Any]):
        values, _, _ = batch
        v_data = values.to(dtype=self.W.dtype, device=self.W.device)
        rng = aux_vars.get("rng", torch.Generator(device="cpu"))

        # CD-k: Always start from Data
        v_model = v_data.clone()

        # Perform k Gibbs steps
        for _ in range(self.k):
            v_model = self._gibbs_step(v_model, rng)

        # Stop gradients flowing through the chain (standard CD practice)
        v_model = v_model.detach()

        # Loss = F(v_data) - F(v_model)
        fe_data = self._free_energy(v_data, self.W, self.b, self.c)
        fe_model = self._free_energy(v_model, self.W, self.b, self.c)

        return (fe_data - fe_model).mean()

    @torch.no_grad()
    def get_full_psi(self):
        """Exact generation for SVD (N=16 only)."""
        device = next(self.parameters()).device
        N = self.num_visible
        indices = torch.arange(2**N, device=device).unsqueeze(1)
        powers = 2**torch.arange(N - 1, -1, -1, device=device)
        v_all = (indices.bitwise_and(powers) != 0).float()

        fe = self._free_energy(v_all, self.W, self.b, self.c)
        fe_shift = fe - fe.min()
        psi = torch.exp(-0.5 * fe_shift)
        return psi / torch.norm(psi)

    @torch.no_grad()
    def generate(self, n_samples: int, burn_in: int, rng: torch.Generator):
        device = next(self.parameters()).device
        # Start random
        v = torch.bernoulli(torch.full((n_samples, self.num_visible), 0.5, device=device), generator=rng)
        for _ in range(burn_in):
            v = self._gibbs_step(v, rng)
        return v

# ==========================================
# 2. METRICS
# ==========================================
def compute_metrics(model: RBM, N: int, rng: torch.Generator):
    model.eval()

    # 1. Correlations (Czz)
    gen = model.generate(5000, 100, rng)
    spins = 2.0 * gen.float() - 1.0
    czz = (spins[:, :-1] * spins[:, 1:]).mean().item()

    # 2. Entropy (S2) via Exact SVD
    psi_vector = model.get_full_psi()
    s2_curve = []
    for l in range(1, N // 2 + 1):
        dim_A = 2**l
        dim_B = 2**(N - l)
        psi_matrix = psi_vector.view(dim_A, dim_B)
        try:
            S = torch.linalg.svdvals(psi_matrix)
        except:
            S = torch.linalg.svdvals(psi_matrix.cpu())
        trace_rho_sq = torch.sum(S**4).item()
        s2_curve.append(-math.log(max(trace_rho_sq, 1e-10)))

    return czz, s2_curve

# ==========================================
# 3. EXPERIMENT
# ==========================================
def run_training(k_step: int, loader, N: int, H: int):
    print(f"\n>>> TRAINING: CD-{k_step} <<<")

    SEED = 42
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)

    model = RBM(num_visible=N, num_hidden=H, k=k_step)

    # Same Optimizer params for fairness
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start_time = time.time()

    for epoch in range(100): # 100 Epochs
        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch, {"rng": rng})
            loss.backward()
            optimizer.step()

        if (epoch+1) % 20 == 0:
            print(f". (Ep {epoch+1})", end="", flush=True)

    total_time = time.time() - start_time
    print(f"\nDone. Time: {total_time:.2f}s")

    return compute_metrics(model, N, rng), total_time

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    CHAIN_LENGTH = 10
    TARGET_DELTA = 1.00
    TRAIN_SAMPLES = 500_000

    # Load Data
    file_name = f"xxz_{CHAIN_LENGTH}_delta{TARGET_DELTA:.2f}_5000000.npz"
    file_path = data_dir / file_name
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[TRAIN_SAMPLES])

    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=512, shuffle=True, drop_last=True, rng=rng_loader)

    # Comparison
    k_values = [2]
    results = {}

    for k in k_values:
        (czz, curve), duration = run_training(k, loader, CHAIN_LENGTH, H=CHAIN_LENGTH*2)
        results[k] = {'time': duration, 'czz': czz, 'curve': curve}

    # Print Table
    print("\n" + "="*50)
    print("COMPARISON: CD-1 vs CD-5")
    print("="*50)
    print(f"{'Method':<8} | {'Time (s)':<10} | {'C_zz':<10} | {'S2(L=8)':<10}")
    print("-" * 50)
    for k in k_values:
        r = results[k]
        print(f"CD-{k:<3}  | {r['time']:<10.2f} | {r['czz']:<10.4f} | {r['curve'][-1]:<10.4f}")
    print("-" * 50)

    # Plot
    ref_file = Path(f"xxz_{CHAIN_LENGTH}_entropy_ref.csv")
    plt.figure(figsize=(9, 6))
    l_axis = list(range(1, CHAIN_LENGTH // 2 + 1))

    # Reference Line
    if ref_file.exists():
        df_ref = pd.read_csv(ref_file)
        l_cols = sorted([c for c in df_ref.columns if c.startswith("l")], key=lambda s: int(s[1:]))
        mask = np.isclose(df_ref["delta"], TARGET_DELTA)
        if mask.any():
            s2_ref = df_ref.loc[mask].iloc[0][l_cols].to_numpy()
            plt.plot(l_axis, s2_ref, 'k--', linewidth=2, label='Reference')

    colors = {1: "green", 5: "red"}
    for k in k_values:
        plt.plot(l_axis, results[k]['curve'], color=colors[k], marker='o', label=f"CD-{k}")

    plt.title(f"Impact of Gibbs Steps 'k' (XXZ N={CHAIN_LENGTH})")
    plt.xlabel("Subsystem size L")
    plt.ylabel("Rényi Entropy S2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()