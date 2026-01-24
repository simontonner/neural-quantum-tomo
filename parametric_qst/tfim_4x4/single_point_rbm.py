import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

# --- IMPORTS FROM YOUR PROJECT ---
# Adjust these paths if necessary
sys.path.append(str(Path("..").resolve()))
from data_handling import load_measurements_npz, MeasurementDataset, MeasurementLoader
from wavefunction_overlap import generate_basis_states, calculate_exact_overlap, load_gt_wavefunction

# --- CONFIGURATION ---
TARGET_H = 3.00       # The specific h-point to test
N_SAMPLES = 2000      # Keep this matching your main experiment
SIDE_LENGTH = 4
EPOCHS = 50
BATCH_SIZE = 1024
HIDDEN_DIM = 64
K_STEPS = 10
SEED = 42

# The Learning Rates to test
# Your current default is 1e-2. We test smaller and larger.
LR_CANDIDATES = [1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]

# Check device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("measurements")
STATE_DIR = Path("state_vectors")

# --- STANDARD SYMMETRIC RBM (NO HYPERNET) ---
class SymmetricRBM(nn.Module):
    def __init__(self, num_v: int, num_h: int, k: int = 10):
        super().__init__()
        self.num_v = num_v
        self.num_h = num_h
        self.k = k
        self.T = 1.0
        self.W = nn.Parameter(torch.empty(num_v, num_h))
        self.b = nn.Parameter(torch.zeros(num_v))
        self.c = nn.Parameter(torch.zeros(num_h))
        self.initialize_weights()

    def initialize_weights(self, std: float = 0.01):
        nn.init.normal_(self.W, std=std)
        nn.init.constant_(self.b, 0.0)
        nn.init.constant_(self.c, 0.0)

    def _free_energy(self, v: torch.Tensor) -> torch.Tensor:
        # v is expected to be float here
        b = self.b.unsqueeze(0)
        c = self.c.unsqueeze(0)

        vW = v @ self.W
        W_colsum = self.W.sum(dim=0)

        preact_normal  = vW + c
        preact_flipped = (W_colsum.unsqueeze(0) - vW) + c

        negF = torch.stack([
            (v * b).sum(-1) - F.softplus(preact_normal).sum(-1),
            ((1.0 - v) * b).sum(-1) - F.softplus(preact_flipped).sum(-1)
        ], dim=-1)

        return -self.T * torch.logsumexp(negF / self.T, dim=-1)

    def forward(self, batch, aux_vars):
        rng = aux_vars.get("rng")

        # --- FIX: Ensure input is float ---
        v_data = batch[0].to(device=self.W.device, dtype=self.W.dtype)

        v_model = v_data.clone()

        # Gibbs Sampling
        B = v_data.size(0)
        u = torch.bernoulli(torch.full((B, 1), 0.5, device=self.W.device), generator=rng)
        h = torch.zeros((B, self.num_h), device=self.W.device)

        b, c = self.b.unsqueeze(0), self.c.unsqueeze(0)

        for _ in range(self.k):
            # 1. h | v, u
            v_branch = u * v_model + (1.0 - u) * (1.0 - v_model)
            h = torch.bernoulli(torch.sigmoid((v_branch @ self.W + c)), generator=rng)

            # Precompute 'a' for next steps
            a = h @ self.W.t()

            # 2. u | v, h
            vb = (v_model * b).sum(dim=-1)
            va = (v_model * a).sum(dim=-1)
            bsum = b.sum(dim=-1)
            asum = a.sum(dim=-1)

            dE = (-bsum - asum + 2.0 * vb + 2.0 * va)
            u = torch.bernoulli(torch.sigmoid(dE), generator=rng).unsqueeze(-1)

            # 3. v | h, u
            v_new = torch.bernoulli(torch.sigmoid(a + b), generator=rng)
            v_model = u * v_new + (1.0 - u) * (1.0 - v_new)

        loss = self._free_energy(v_data).mean() - self._free_energy(v_model.detach()).mean()
        return loss

    def log_score(self, v, cond=None):
        # Helper for overlap calculation
        return -0.5 * self._free_energy(v) / self.T

# --- MAIN EXPERIMENT ---
def run_comparison():
    print(f"--- Running Sensitivity Analysis on Single RBM (h={TARGET_H}) ---")
    print(f"Baseline LR used in paper: 1e-2")

    # 1. Load Data
    file_path = DATA_DIR / f"tfim_{SIDE_LENGTH}x{SIDE_LENGTH}_h{TARGET_H:.2f}_20000.npz"
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    dataset = MeasurementDataset([file_path], load_measurements_npz, ["h"], [N_SAMPLES])

    # Load GT for evaluation
    gt_path = STATE_DIR / f"tfim_{SIDE_LENGTH}x{SIDE_LENGTH}_h{TARGET_H:.2f}.npz"
    if not gt_path.exists():
        print(f"Error: GT State not found at {gt_path}")
        return

    psi_true = load_gt_wavefunction(gt_path, DEVICE)
    basis_states = generate_basis_states(SIDE_LENGTH**2, DEVICE)

    results = []

    for lr in LR_CANDIDATES:
        print(f"\nTraining with LR: {lr} ...")

        # Reset Seed for fairness (same weight init, same batch shuffling)
        torch.manual_seed(SEED)
        rng = torch.Generator(device=DEVICE).manual_seed(SEED)

        loader = MeasurementLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, rng=rng)

        # Init Model
        model = SymmetricRBM(dataset.num_qubits, HIDDEN_DIM, K_STEPS).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Train
        model.train()
        final_loss = 0
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                loss = model(batch, {"rng": rng})
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            final_loss = epoch_loss / len(loader)

        # Evaluate
        model.eval()
        overlap = calculate_exact_overlap(model, TARGET_H, psi_true, basis_states)

        print(f"  -> Final Loss: {final_loss:.4f} | Overlap: {overlap:.5f}")
        results.append({"lr": lr, "overlap": overlap, "loss": final_loss})

    # --- PLOTTING ---
    df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))

    # Plot 1: Overlap vs LR
    plt.subplot(1, 2, 1)
    plt.plot(df["lr"], df["overlap"], 'o-', color='crimson', label="Single RBM Performance")
    plt.xscale('log')
    plt.axvline(x=1e-2, color='gray', linestyle='--', label="Your Current LR (1e-2)")
    plt.xlabel("Learning Rate")
    plt.ylabel("Overlap")
    plt.title("Sensitivity: Overlap vs LR")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Training Loss vs LR
    plt.subplot(1, 2, 2)
    plt.plot(df["lr"], df["loss"], 's-', color='navy', label="Training Loss (Free Energy Diff)")
    plt.xscale('log')
    plt.axvline(x=1e-2, color='gray', linestyle='--')
    plt.xlabel("Learning Rate")
    plt.ylabel("Training Loss (Lower is better fit)")
    plt.title("Is it Underfitting?")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("single_rbm_sensitivity.png")
    plt.show()

    # --- VERDICT ---
    best_row = df.loc[df["overlap"].idxmax()]
    baseline_row = df[df["lr"] == 1e-2].iloc[0]

    print("\n" + "="*40)
    print("VERDICT:")
    print(f"Overlap at your chosen LR (1e-2): {baseline_row['overlap']:.5f}")
    print(f"Best Overlap found (LR={best_row['lr']}): {best_row['overlap']:.5f}")

    diff = best_row['overlap'] - baseline_row['overlap']

    if diff < 0.005:
        print(">> RESULT: The User is RIGHT.")
        print("   Tuning the LR makes negligible difference.")
        print("   The Single RBM is failing due to limited data, not bad optimization.")
    else:
        print(">> RESULT: The Colleague is RIGHT.")
        print(f"   Tuning improved overlap by {diff:.5f}.")
        print("   The Single RBM was indeed handicapped by the fixed LR.")
    print("="*40)

if __name__ == "__main__":
    run_comparison()