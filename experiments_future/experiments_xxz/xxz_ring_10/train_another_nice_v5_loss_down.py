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
# 1. SYMMETRIC RBM (Convolutional)
# ==========================================
class SymmetricRBM(nn.Module):
    def __init__(self, num_visible: int, alpha: int, k: int):
        super().__init__()
        self.num_visible = num_visible
        self.alpha = alpha
        self.num_hidden = alpha * num_visible
        self.k = k
        self.T = 1.0

        # Kernel shape: [1, Alpha, N] -> Represents the unique weights W_{0,j}
        self.kernel = nn.Parameter(torch.empty(1, alpha, num_visible))

        # Bias: Scalar (shared across all sites)
        self.b_scalar = nn.Parameter(torch.zeros(1))
        # Hidden Bias: One per feature channel (shared spatially)
        self.c_vector = nn.Parameter(torch.zeros(1, alpha))

        self.initialize_weights()

    def initialize_weights(self):
        # Small noise to break symmetry
        nn.init.normal_(self.kernel, mean=0.0, std=0.05)
        nn.init.constant_(self.b_scalar, 0.0)
        nn.init.constant_(self.c_vector, 0.0)

    def compute_energy_term(self, v: torch.Tensor):
        """Computes W*v using convolution with circular padding."""
        # Input v: [Batch, N] -> [Batch, 1, N]
        v_reshaped = v.unsqueeze(1)

        # Weight for Conv1d: [Out_Channels, In_Channels, Kernel_Width]
        # We want Alpha features per site.
        # Shape: [Alpha, 1, N]
        weight = self.kernel.view(self.alpha, 1, self.num_visible)

        # 1. Manual Circular Padding (Fixes the RuntimeError)
        # Pad right side by (N-1) to simulate circular conv over N
        padded_v = F.pad(v_reshaped, (0, self.num_visible - 1), mode='circular')

        # 2. Convolution
        # Output: [Batch, Alpha, N]
        activation = F.conv1d(padded_v, weight)

        # Flatten: [Batch, Alpha * N]
        return activation.view(v.shape[0], -1)

    def _free_energy(self, v: torch.Tensor) -> torch.Tensor:
        v = v.float()

        # Visible Term: -b * sum(v)
        term1 = -self.b_scalar * v.sum(dim=-1)

        # Hidden Term: -sum(softplus(Wv + c))
        wv = self.compute_energy_term(v)

        # Expand c_vector to match [Batch, Alpha * N]
        # c_vector is [1, Alpha]. We repeat it N times interleaved.
        # Easier: Reshape wv to [Batch, Alpha, N] and broadcast c [1, Alpha, 1]
        wv_reshaped = wv.view(-1, self.alpha, self.num_visible)
        c_reshaped = self.c_vector.view(1, self.alpha, 1)

        term2 = -F.softplus(wv_reshaped + c_reshaped).sum(dim=(1, 2))

        return term1 + term2

    def _gibbs_step(self, v: torch.Tensor, rng: torch.Generator):
        # --- Forward: v -> h ---
        # 1. Compute activation
        wv = self.compute_energy_term(v) # [Batch, Alpha * N]
        wv_reshaped = wv.view(-1, self.alpha, self.num_visible)
        c_reshaped = self.c_vector.view(1, self.alpha, 1)

        # 2. Sample h
        p_h = torch.sigmoid((wv_reshaped + c_reshaped) / self.T)
        h = torch.bernoulli(p_h, generator=rng) # [Batch, Alpha, N]

        # --- Backward: h -> v ---
        # We need Transpose Convolution. For circular 1D, this is Convolution with Flipped Kernel.

        # Weight: [1, Alpha, N] (In=Alpha, Out=1)
        # We flip the spatial dimension (dim -1)
        w_back = torch.flip(self.kernel, dims=[-1]).view(1, self.alpha, self.num_visible)

        # Manual Circular Padding for H
        padded_h = F.pad(h, (0, self.num_visible - 1), mode='circular')

        # Convolve
        wh = F.conv1d(padded_h, w_back) # [Batch, 1, N]
        wh = wh.view(v.shape[0], self.num_visible)

        # Sample v
        p_v = torch.sigmoid((wh + self.b_scalar) / self.T)
        v_next = torch.bernoulli(p_v, generator=rng)

        return v_next

    def forward(self, batch: Tuple[torch.Tensor, ...], aux_vars: Dict[str, Any]):
        values, _, _ = batch
        v_data = values.to(device=self.kernel.device).float()
        rng = aux_vars.get("rng", torch.Generator(device="cpu"))

        # CD-k
        v_model = v_data.clone()
        for _ in range(self.k):
            v_model = self._gibbs_step(v_model, rng)
        v_model = v_model.detach()

        fe_data = self._free_energy(v_data)
        fe_model = self._free_energy(v_model)

        return (fe_data - fe_model).mean()

    @torch.no_grad()
    def generate(self, n_samples: int, burn_in: int, rng: torch.Generator):
        device = next(self.parameters()).device
        v = torch.bernoulli(torch.full((n_samples, self.num_visible), 0.5, device=device), generator=rng)
        for _ in range(burn_in):
            v = self._gibbs_step(v, rng)
        return v

   # @torch.no_grad()
   # def get_full_psi(self):
   #     # ... (lines 135-144: generate states and calculate amplitude) ...
#
   #     # 1. Get Amplitude |Psi| (The RBM output)
   #     fe = self._free_energy(v_all)
   #     psi_abs = torch.exp(-0.5 * (fe - fe.min()))
#
   #     # ==========================================================
   #     # 2. MARSHALL SIGN RULE (Hard-coded here)
   #     # ==========================================================
   #     # Logic: Sign is (-1) raised to the number of Up-spins on odd sites
#
   #     # A. Identify odd positions (indices 1, 3, 5, 7, 9)
   #     odd_indices = torch.arange(1, self.num_visible, 2, device=device)
#
   #     # B. Count how many '1's (up spins) are at those positions for every state
   #     n_up_odd = v_all[:, odd_indices].sum(dim=1)
#
   #     # C. Calculate sign: (-1)^count
   #     signs = (-1.0) ** n_up_odd
#
   #     # D. Apply to wavefunction
   #     psi = psi_abs * signs
   #     # ==========================================================
#
   #     return psi / torch.norm(psi)

    @torch.no_grad()
    def get_full_psi(self):
        """Exact generation for SVD with Marshall Sign Rule manually applied."""
        device = next(self.parameters()).device
        N = self.num_visible

        # 1. Generate all 2^N states
        indices = torch.arange(2**N, device=device).unsqueeze(1)
        powers = 2**torch.arange(N - 1, -1, -1, device=device)
        v_all = (indices.bitwise_and(powers) != 0).float()

        # 2. Get Amplitude |Psi|
        fe = self._free_energy(v_all)
        psi_abs = torch.exp(-0.5 * (fe - fe.min()))

        # 3. APPLY MARSHALL SIGN RULE (Crucial for correct Entropy)
        # Signs = (-1)^(number of up spins at odd positions)
        odd_indices = torch.arange(1, N, 2, device=device)
        n_up_odd = v_all[:, odd_indices].sum(dim=1)
        signs = (-1.0) ** n_up_odd

        psi = psi_abs * signs

        return psi / torch.norm(psi)

# ==========================================
# 2. METRICS
# ==========================================
def compute_data_czz(loader) -> float:
    print("Calculating baseline Czz...", end=" ", flush=True)
    total_czz = 0.0
    count = 0
    for batch in loader:
        values, _, _ = batch
        spins = 2.0 * values.float() - 1.0
        batch_avg = (spins[:, :-1] * spins[:, 1:]).mean().item()
        total_czz += batch_avg
        count += 1
    avg = total_czz / count if count > 0 else 0.0
    print(f" {avg:.4f}")
    return avg

def get_current_czz(model, rng):
    model.eval()
    gen = model.generate(2000, 50, rng)
    spins = 2.0 * gen.float() - 1.0
    return (spins[:, :-1] * spins[:, 1:]).mean().item()

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def run_training(loader, N, alpha, k_step, epochs, lr):
    print(f"\n>>> TRAINING SYMMETRIC RBM | Alpha: {alpha} | CD-{k_step} <<<")

    SEED = 42
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)

    model = SymmetricRBM(num_visible=N, alpha=alpha, k=k_step)

    # Weight Decay usually not needed for Symmetric RBM as params are shared/constrained
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    # Simple Step Decay
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//2, gamma=0.5)

    history = {'loss': [], 'czz': []}
    start_time = time.time()

    print(f"{'Epoch':<6} | {'Loss':<10} | {'C_zz':<10} | {'Time':<8}")
    print("-" * 45)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch, {"rng": rng})
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = epoch_loss / len(loader)
            curr_czz = get_current_czz(model, rng)
            history['loss'].append(avg_loss)
            history['czz'].append(curr_czz)

            elapsed = time.time() - start_time
            print(f"{epoch+1:<6} | {avg_loss:<10.4f} | {curr_czz:<10.4f} | {elapsed:<8.1f}s")

    return model, history

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    CHAIN_LENGTH = 10
    TARGET_DELTA = 1.00
    TRAIN_SAMPLES = 50_000

    # Config
    ALPHA = 4          # Hidden unit density
    CD_K = 10          # Gibbs steps (Symmetric converges faster, so 10-20 is often enough)
    EPOCHS = 100
    LR = 0.01

    # Load
    file_name = f"xxz_{CHAIN_LENGTH}_delta{TARGET_DELTA:.2f}_5000000.npz"
    file_path = data_dir / file_name
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[TRAIN_SAMPLES])
    loader = MeasurementLoader(dataset=ds, batch_size=256, shuffle=True, drop_last=True, rng=torch.Generator())

    # Baseline
    base_czz = compute_data_czz(loader)

    # Train
    model, history = run_training(loader, CHAIN_LENGTH, ALPHA, CD_K, EPOCHS, LR)

    # Entropy
    print("\nCalculating Entropy...")
    psi = model.get_full_psi()

    s2_curve = []
    for l in range(1, CHAIN_LENGTH // 2 + 1):
        dim_A = 2**l
        dim_B = 2**(CHAIN_LENGTH - l)
        mat = psi.view(dim_A, dim_B)

        try:
            S = torch.linalg.svdvals(mat)
        except:
            S = torch.linalg.svdvals(mat.cpu())

        s2 = -math.log(torch.sum(S**4).item())
        s2_curve.append(s2)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Czz Plot
    ax[0].plot(range(0, EPOCHS+1, 5) if len(history['czz']) > EPOCHS/5 else range(len(history['czz'])), history['czz'], 'b-o', label='Model Czz')
    ax[0].axhline(base_czz, color='g', linestyle='--', label='Data Czz')
    ax[0].set_title("Training: Correlation Convergence")
    ax[0].set_xlabel("Steps (x5 Epochs)")
    ax[0].legend()

    # Entropy Plot
    ax[1].plot(range(1, 6), s2_curve, 'r-o', label='Symmetric RBM')

    ref_file = Path(f"xxz_{CHAIN_LENGTH}_entropy_ref.csv")
    if ref_file.exists():
        df = pd.read_csv(ref_file)
        row = df[np.isclose(df['delta'], TARGET_DELTA)].iloc[0]
        ref = [row[f'l{i}'] for i in range(1, 6)]
        ax[1].plot(range(1, 6), ref, 'k--', label='Reference')

    ax[1].set_title("Final Entropy")
    ax[1].set_xlabel("Subsystem Size L")
    ax[1].legend()
    plt.tight_layout()
    plt.show()