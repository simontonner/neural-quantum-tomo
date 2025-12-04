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
# Adjust this if your data_handling.py is elsewhere
sys.path.append(str(Path("..").resolve()))
from data_handling import load_measurements_npz, MeasurementDataset, MeasurementLoader

data_dir = Path("measurements")

# ==========================================
# 1. RBM MODEL
# ==========================================
class RBM(nn.Module):
    def __init__(self, num_visible: int, num_hidden: int, k: int, T: float = 1.0, batch_size: int = 512):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.T = T
        self.batch_size = batch_size

        self.W = nn.Parameter(torch.empty(num_visible, num_hidden))
        self.b = nn.Parameter(torch.zeros(num_visible))
        self.c = nn.Parameter(torch.zeros(num_hidden))

        self.initialize_weights()

        # Buffer for PCD (Persistent Chain)
        # Initialize with 0.5 probability (random noise)
        self.register_buffer('persistent_chain', torch.bernoulli(torch.full((batch_size, num_visible), 0.5)))

    def initialize_weights(self):
        # Slightly smaller initialization for stability
        nn.init.normal_(self.W, mean=0.0, std=0.01)
        nn.init.constant_(self.b, 0.0)
        nn.init.constant_(self.c, 0.0)

    @staticmethod
    def _free_energy(v: torch.Tensor, W: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        v = v.to(dtype=W.dtype, device=W.device)
        term1 = -(v * b).sum(dim=-1)
        term2 = F.softplus(v @ W + c).sum(dim=-1)
        return term1 - term2

    def _gibbs_step(self, v: torch.Tensor, T: float, rng: torch.Generator):
        # h ~ p(h|v)
        p_h = torch.sigmoid((v @ self.W + self.c) / T)
        h = torch.bernoulli(p_h, generator=rng)
        # v ~ p(v|h)
        p_v = torch.sigmoid((h @ self.W.t() + self.b) / T)
        v_next = torch.bernoulli(p_v, generator=rng)
        return v_next

    def forward(self, batch: Tuple[torch.Tensor, ...], aux_vars: Dict[str, Any]):
        values, _, _ = batch
        v_data = values.to(dtype=self.W.dtype, device=self.W.device)
        rng = aux_vars.get("rng", torch.Generator(device="cpu"))
        method = aux_vars.get("method", "CD")

        # --- 1. POSITIVE PHASE (Data) ---
        # No sampling needed for free energy of data

        # --- 2. NEGATIVE PHASE (Model) ---
        if method == "PCD":
            # Start from persistent chain
            # Ensure chain size matches batch size (handle last batch drop)
            if self.persistent_chain.shape[0] != v_data.shape[0]:
                # Resize buffer if batch size changes (rare with drop_last=True)
                self.persistent_chain = torch.bernoulli(torch.full((v_data.shape[0], self.num_visible), 0.5)).to(v_data.device)

            v_model = self.persistent_chain.detach()
        else:
            # CD: Start from data
            v_model = v_data.clone()

        # Gibbs Sampling
        for _ in range(self.k):
            v_model = self._gibbs_step(v_model, self.T, rng)

        # Update Persistent Chain
        if method == "PCD":
            self.persistent_chain = v_model.detach()

        v_model = v_model.detach()

        # --- 3. LOSS ---
        fe_data = self._free_energy(v_data, self.W, self.b, self.c)
        fe_model = self._free_energy(v_model, self.W, self.b, self.c)
        loss = (fe_data - fe_model).mean()

        return loss

    @torch.no_grad()
    def generate(self, n_samples: int, k_steps: int, rng: torch.Generator) -> torch.Tensor:
        device = next(self.parameters()).device
        v = torch.bernoulli(torch.full((n_samples, self.num_visible), 0.5, device=device), generator=rng)
        for _ in range(k_steps):
            v = self._gibbs_step(v, self.T, rng)
        return v

    @torch.no_grad()
    def get_full_psi(self):
        """Exact generation for SVD (N=16 only)."""
        device = next(self.parameters()).device
        N = self.num_visible
        indices = torch.arange(2**N, device=device).unsqueeze(1)
        powers = 2**torch.arange(N - 1, -1, -1, device=device)
        v_all = (indices.bitwise_and(powers) != 0).float()

        # Calculate unnormalized probability
        fe = self._free_energy(v_all, self.W, self.b, self.c)
        # Numerical stability trick: subtract min
        fe_shift = fe - fe.min()
        psi = torch.exp(-0.5 * fe_shift) # Psi = sqrt(Prob) = exp(-0.5 * Energy)

        # Normalize
        norm = torch.norm(psi)
        return psi / norm

# ==========================================
# 2. UTILS
# ==========================================
def compute_entropy_exact_svd(model: RBM, N: int):
    # This is the exact SVD method you used before
    model.eval()
    psi_vector = model.get_full_psi()
    results = []
    # Calculate for all cuts
    for l in range(1, N // 2 + 1):
        dim_A = 2**l
        dim_B = 2**(N - l)
        psi_matrix = psi_vector.view(dim_A, dim_B)
        try:
            S = torch.linalg.svdvals(psi_matrix)
        except: # Fallback to CPU if GPU runs out
            S = torch.linalg.svdvals(psi_matrix.cpu())

        # Renyi-2 Entropy: S2 = -ln(Tr(rho^2)) = -ln(sum(sigma^4))
        trace_rho_sq = torch.sum(S**4).item()
        s2 = -math.log(max(trace_rho_sq, 1e-10))
        results.append(s2)
    return results

def compute_nearest_neighbor_corr(model, rng):
    with torch.no_grad():
        # Generate enough samples to be accurate
        gen = model.generate(5000, 100, rng)
    spins = 2.0 * gen.float() - 1.0
    return (spins[:, :-1] * spins[:, 1:]).mean().item()

# ==========================================
# 3. EXPERIMENT RUNNER
# ==========================================
def run_training(method_name: str, loader, params: Dict, N: int, H: int):
    print(f"\n>>> TRAINING: {method_name.upper()} | LR: {params['lr']} | k: {params['k']}")

    SEED = 42
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)

    model = RBM(num_visible=N, num_hidden=H, k=params['k'], batch_size=params['batch_size'])

    # === KEY FIX: ADAM OPTIMIZER ===
    # Adam is much better for PCD than SGD because it adapts the learning rate
    # per parameter, preventing the persistent chain from diverging.
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    start_time = time.time()

    for epoch in range(params['epochs']):
        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch, {"rng": rng, "method": method_name})
            loss.backward()
            optimizer.step()

        if (epoch+1) % 25 == 0:
            print(f". (Ep {epoch+1})", end="", flush=True)

    total_time = time.time() - start_time
    print(f"\nDone. Time: {total_time:.2f}s")

    # Metrics
    czz = compute_nearest_neighbor_corr(model, rng)
    svd_curve = compute_entropy_exact_svd(model, N)

    return total_time, czz, svd_curve

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":

    CHAIN_LENGTH = 16
    TARGET_DELTA = 1.00
    TRAIN_SAMPLES = 100_000

    # DATA LOADING
    file_name = f"xxz_{CHAIN_LENGTH}_delta{TARGET_DELTA:.2f}_5000000.npz"
    file_path = data_dir / file_name
    print(f"Loading {file_path}...")
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[TRAIN_SAMPLES])

    rng_loader = torch.Generator().manual_seed(42)
    # Using a moderate batch size that is safe for both
    loader = MeasurementLoader(dataset=ds, batch_size=512, shuffle=True, drop_last=True, rng=rng_loader)

    # === HYPERPARAMETER SETUP ===
    # We decouple the configs to give each method its best chance

    configs = {
        "CD": {
            "epochs": 100,
            "lr": 1e-3,       # CD needs standard learning rate
            "k": 5,           # CD needs k > 1 to learn correlations well
            "batch_size": 512
        },
        "PCD": {
            "epochs": 100,
            "lr": 5e-4,       # PCD needs LOWER learning rate to stay stable
            "k": 2,           # PCD mixes over epochs, so small k is fine (and faster)
            "batch_size": 512
        }
    }

    results = {}

    for method in ["CD", "PCD"]:
        t, czz, curve = run_training(method, loader, configs[method], CHAIN_LENGTH, H=CHAIN_LENGTH*2)
        results[method] = {'time': t, 'czz': czz, 'curve': curve}

    # === REPORT & PLOT ===
    print("\n" + "="*50)
    print("FINAL COMPARISON: CD vs PCD (Optimized Params)")
    print("="*50)
    print(f"{'Method':<8} | {'Time':<8} | {'C_zz':<10} | {'S2(L=8)':<10}")
    print("-" * 50)

    for m in results:
        r = results[m]
        print(f"{m:<8} | {r['time']:<8.2f} | {r['czz']:<10.4f} | {r['curve'][-1]:<10.4f}")
    print("-" * 50)

    # Plot
    ref_file = Path(f"xxz_{CHAIN_LENGTH}_entropy_ref.csv")
    plt.figure(figsize=(9, 6))

    l_axis = list(range(1, CHAIN_LENGTH // 2 + 1))

    if ref_file.exists():
        df_ref = pd.read_csv(ref_file)
        l_cols = sorted([c for c in df_ref.columns if c.startswith("l")], key=lambda s: int(s[1:]))
        mask = np.isclose(df_ref["delta"], TARGET_DELTA)
        if mask.any():
            s2_ref = df_ref.loc[mask].iloc[0][l_cols].to_numpy()
            plt.plot(l_axis, s2_ref, 'k--', linewidth=2, label='Reference (Exact)')

    colors = {"CD": "red", "PCD": "blue"}
    for m in results:
        plt.plot(l_axis, results[m]['curve'], color=colors[m], marker='o',
                 label=f"{m} (LR={configs[m]['lr']}, k={configs[m]['k']})")

    plt.title(f"Optimized Training Comparison (XXZ N={CHAIN_LENGTH})")
    plt.xlabel(r"Subsystem size $\ell$")
    plt.ylabel(r"Rényi Entropy $S_2$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()