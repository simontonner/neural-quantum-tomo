import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple, List

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
        # Increased std slightly to help break symmetry early
        nn.init.normal_(self.W, mean=0.0, std=0.05)
        nn.init.constant_(self.b, 0.0)
        nn.init.constant_(self.c, 0.0)

    @staticmethod
    def _free_energy(v: torch.Tensor, W: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        v = v.to(dtype=W.dtype, device=W.device)
        term1 = -(v * b).sum(dim=-1)
        term2 = F.softplus(v @ W + c).sum(dim=-1)
        return term1 - term2

    def _gibbs_step(self, v: torch.Tensor, rng: torch.Generator):
        # v -> h
        p_h = torch.sigmoid((v @ self.W + self.c) / self.T)
        h = torch.bernoulli(p_h, generator=rng)
        # h -> v
        p_v = torch.sigmoid((h @ self.W.t() + self.b) / self.T)
        v_next = torch.bernoulli(p_v, generator=rng)
        return v_next

    def forward(self, batch: Tuple[torch.Tensor, ...], aux_vars: Dict[str, Any]):
        values, _, _ = batch
        v_data = values.to(dtype=self.W.dtype, device=self.W.device)
        rng = aux_vars.get("rng", torch.Generator(device="cpu"))

        # CD-k: Start from Data
        v_model = v_data.clone()

        # Perform k Gibbs steps (Crucial for getting correlations right)
        for _ in range(self.k):
            v_model = self._gibbs_step(v_model, rng)

        v_model = v_model.detach()

        # CD Loss = F(v_data) - F(v_model)
        fe_data = self._free_energy(v_data, self.W, self.b, self.c)
        fe_model = self._free_energy(v_model, self.W, self.b, self.c)

        return (fe_data - fe_model).mean()

    @torch.no_grad()
    def generate(self, n_samples: int, burn_in: int, rng: torch.Generator):
        device = next(self.parameters()).device
        v = torch.bernoulli(torch.full((n_samples, self.num_visible), 0.5, device=device), generator=rng)
        # Increased burn-in for generation to ensure we don't just see noise
        for _ in range(burn_in):
            v = self._gibbs_step(v, rng)
        return v

    @torch.no_grad()
    def get_full_psi(self):
        """Exact generation for SVD (N<=20)."""
        device = next(self.parameters()).device
        N = self.num_visible
        indices = torch.arange(2**N, device=device).unsqueeze(1)
        powers = 2**torch.arange(N - 1, -1, -1, device=device)
        v_all = (indices.bitwise_and(powers) != 0).float()

        fe = self._free_energy(v_all, self.W, self.b, self.c)
        # Subtract min for numerical stability before exp
        fe_shift = fe - fe.min()
        psi = torch.exp(-0.5 * fe_shift)
        return psi / torch.norm(psi)

# ==========================================
# 2. METRICS & HELPERS
# ==========================================

def compute_data_czz(loader) -> float:
    """Calculates the ground truth Czz from the training dataset."""
    print("Calculating baseline Czz from dataset...", end=" ", flush=True)
    total_czz = 0.0
    count = 0
    for batch in loader:
        values, _, _ = batch
        spins = 2.0 * values.float() - 1.0
        batch_avg = (spins[:, :-1] * spins[:, 1:]).mean().item()
        total_czz += batch_avg
        count += 1
    avg_czz = total_czz / count if count > 0 else 0.0
    print(f"Done. Found {avg_czz:.4f}")
    return avg_czz

def get_current_czz(model: RBM, rng: torch.Generator) -> float:
    model.eval()
    # Need reasonable burn-in to ensure samples are from the model's equilibrium
    gen = model.generate(2000, 50, rng)
    spins = 2.0 * gen.float() - 1.0
    czz = (spins[:, :-1] * spins[:, 1:]).mean().item()
    model.train()
    return czz

def run_training(
        loader,
        N: int,
        H: int,
        k_step: int,
        epochs: int,
        lr: float,
        l1_weight: float,
        l2_weight: float
):
    print(f"\n>>> TRAINING: CD-{k_step} | Hidden: {H} | LR: {lr} <<<")

    SEED = 42
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)

    model = RBM(num_visible=N, num_hidden=H, k=k_step)

    # Updated Optimizer: L1 removed from loop, L2 kept small
    # Using a Scheduler to allow high learning initially, then fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_weight)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//2, gamma=0.2)

    history = {'loss': [], 'czz': []}
    start_time = time.time()

    print(f"{'Epoch':<6} | {'Loss':<10} | {'C_zz':<10} | {'LR':<8} | {'Time':<8}")
    print("-" * 55)

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in loader:
            optimizer.zero_grad()

            # Pure CD Loss (Removed manual L1 regularization loop to allow weights to grow)
            loss = model(batch, {"rng": rng})

            if l1_weight > 0:
                l1_reg = l1_weight * torch.sum(torch.abs(model.W))
                (loss + l1_reg).backward()
            else:
                loss.backward()

            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        current_czz = get_current_czz(model, rng)

        history['loss'].append(avg_loss)
        history['czz'].append(current_czz)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"{epoch+1:<6} | {avg_loss:<10.4f} | {current_czz:<10.4f} | {current_lr:<8.1e} | {elapsed:<8.1f}s")

    total_time = time.time() - start_time
    print(f"\nDone. Total Time: {total_time:.2f}s")

    return model, history

# ==========================================
# 3. ENTROPY CALCULATION
# ==========================================
def calculate_entropy_curve(model: RBM, N: int):
    model.eval()
    print("Calculating Entropy via Exact SVD...")
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

    return s2_curve

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    # --- CONFIGURATION (UPDATED) ---
    CHAIN_LENGTH = 10
    TARGET_DELTA = 1.00
    TRAIN_SAMPLES = 100_000

    # Key Changes Here:
    CD_K = 100              # Increased from 1 to 20 for better gradient mixing
    EPOCHS = 100           # Train longer
    HIDDEN_UNITS = CHAIN_LENGTH * 2  # Increased capacity (alpha=4)
    LEARNING_RATE = 0.01   # Start higher to escape local minima

    # Regularization
    L1_REG = 0.0           # Turn OFF L1 (Sparsity hurts dense entanglement)
    L2_REG = 1e-5          # Very weak L2 just to prevent explosion
    # -------------------------------

    # Load Data
    file_name = f"xxz_{CHAIN_LENGTH}_delta{TARGET_DELTA:.2f}_5000000.npz"
    file_path = data_dir / file_name

    print(f"Loading data from {file_path}...")
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[TRAIN_SAMPLES])

    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=128, shuffle=True, drop_last=True, rng=rng_loader)

    # Baseline
    baseline_czz = compute_data_czz(loader)
    print("\n" + "="*40)
    print(f"BASELINE DATA C_zz: {baseline_czz:.4f}")
    print("="*40)

    # Run
    model, history = run_training(
        loader,
        N=CHAIN_LENGTH,
        H=HIDDEN_UNITS,
        k_step=CD_K,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        l1_weight=L1_REG,
        l2_weight=L2_REG
    )

    # Entropy
    entropy_curve = calculate_entropy_curve(model, CHAIN_LENGTH)

    # --- PLOTTING ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Training History
    ax[0].plot(history['loss'], label='Reconstruction Loss', color='blue', alpha=0.6)
    ax[0].set_ylabel("Loss", color='blue')
    ax[0].set_xlabel("Epoch")
    ax[0].tick_params(axis='y', labelcolor='blue')
    ax[0].set_title(f"Training Dynamics (CD-{CD_K})")

    ax2 = ax[0].twinx()
    ax2.plot(history['czz'], label=r'Model $C_{zz}$', color='orange', linewidth=2)
    ax2.axhline(y=baseline_czz, color='green', linestyle=':', linewidth=2, label=r'Data $C_{zz}$')
    ax2.set_ylabel(r"Correlation $C_{zz}$", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Legend trick
    lines_1, labels_1 = ax[0].get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax[0].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    # Plot 2: Entropy
    l_axis = list(range(1, CHAIN_LENGTH // 2 + 1))

    ref_file = Path(f"xxz_{CHAIN_LENGTH}_entropy_ref.csv")
    if ref_file.exists():
        df_ref = pd.read_csv(ref_file)
        l_cols = sorted([c for c in df_ref.columns if c.startswith("l")], key=lambda s: int(s[1:]))
        mask = np.isclose(df_ref["delta"], TARGET_DELTA)
        if mask.any():
            s2_ref = df_ref.loc[mask].iloc[0][l_cols].to_numpy()
            ax[1].plot(l_axis, s2_ref, 'k--', linewidth=2, label='Reference')

    ax[1].plot(l_axis, entropy_curve, 'ro-', label=f"RBM (CD-{CD_K})")
    ax[1].set_xlabel("Subsystem size L")
    ax[1].set_ylabel("Rényi Entropy S2")
    ax[1].set_title(f"Final Entanglement Entropy")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()