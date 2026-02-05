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
# 1. SYMMETRIC RBM (THE AMPLITUDE LEARNER)
# ==========================================
class SymmetricRBM(nn.Module):
    """
    Learns the probability distribution P(v) = |Psi(v)|^2.
    Uses 1D Convolution to enforce translational symmetry.
    """
    def __init__(self, num_visible: int, alpha: int, k: int):
        super().__init__()
        self.num_visible = num_visible
        self.alpha = alpha
        self.k = k
        self.T = 1.0

        # One kernel of size N per hidden feature alpha
        # Shape: [1, Alpha, N]
        self.kernel = nn.Parameter(torch.empty(1, alpha, num_visible))

        self.b_scalar = nn.Parameter(torch.zeros(1))
        self.c_vector = nn.Parameter(torch.zeros(1, alpha))

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.kernel, mean=0.0, std=0.02) # Small weights to prevent freezing
        nn.init.constant_(self.b_scalar, 0.0)
        nn.init.constant_(self.c_vector, 0.0)

    def compute_energy_term(self, v: torch.Tensor):
        # v: [Batch, N] -> [Batch, 1, N]
        v_in = v.unsqueeze(1)

        # Manual Circular Padding
        # Pad right side by N-1
        v_padded = F.pad(v_in, (0, self.num_visible - 1), mode='circular')

        # Conv weights: [Alpha, 1, N]
        weight = self.kernel.view(self.alpha, 1, self.num_visible)

        # Result: [Batch, Alpha, N]
        activation = F.conv1d(v_padded, weight)
        return activation.view(v.shape[0], -1)

    def _free_energy(self, v: torch.Tensor) -> torch.Tensor:
        v = v.float()
        term1 = -self.b_scalar * v.sum(dim=-1)

        wv = self.compute_energy_term(v)
        # Broadcast c across the N positions
        # wv is [Batch, Alpha*N]. c is [Alpha].
        # Reshape to sum correctly
        wv_r = wv.view(-1, self.alpha, self.num_visible)
        c_r = self.c_vector.view(1, self.alpha, 1)

        term2 = -F.softplus(wv_r + c_r).sum(dim=(1, 2))
        return term1 + term2

    def _gibbs_step(self, v: torch.Tensor, rng: torch.Generator):
        # Forward
        wv = self.compute_energy_term(v).view(-1, self.alpha, self.num_visible)
        c_r = self.c_vector.view(1, self.alpha, 1)
        p_h = torch.sigmoid((wv + c_r) / self.T)
        h = torch.bernoulli(p_h, generator=rng)

        # Backward (Transpose Conv)
        # Flip kernel spatially
        w_back = torch.flip(self.kernel, dims=[-1]).view(1, self.alpha, self.num_visible)

        # Circular Pad H
        h_padded = F.pad(h, (0, self.num_visible - 1), mode='circular')

        wh = F.conv1d(h_padded, w_back) # [Batch, 1, N]
        wh = wh.view(v.shape[0], self.num_visible)

        p_v = torch.sigmoid((wh + self.b_scalar) / self.T)
        v_next = torch.bernoulli(p_v, generator=rng)
        return v_next

    def forward(self, batch: Tuple[torch.Tensor, ...], aux_vars: Dict[str, Any]):
        values, _, _ = batch
        v_data = values.to(device=self.kernel.device).float()
        rng = aux_vars.get("rng", torch.Generator(device="cpu"))

        v_model = v_data.clone()
        for _ in range(self.k):
            v_model = self._gibbs_step(v_model, rng)
        v_model = v_model.detach()

        return (self._free_energy(v_data) - self._free_energy(v_model)).mean()

    @torch.no_grad()
    def get_full_psi_with_marshall_sign(self):
        """
        1. Generates RBM Amplitude |Psi| (Positive)
        2. Multiplies by Marshall Sign Rule manually
        """
        device = next(self.parameters()).device
        N = self.num_visible

        # 1. Generate all states
        indices = torch.arange(2**N, device=device).unsqueeze(1)
        powers = 2**torch.arange(N - 1, -1, -1, device=device)
        v_all = (indices.bitwise_and(powers) != 0).float()

        # 2. Get Amplitude from RBM
        fe = self._free_energy(v_all)
        psi_abs = torch.exp(-0.5 * (fe - fe.min()))

        # 3. APPLY MARSHALL SIGN RULE MANUALLY
        # This is allowed because we know the physics of the system we measured.
        odd_indices = torch.arange(1, N, 2, device=device)
        n_up_odd = v_all[:, odd_indices].sum(dim=1)
        signs = (-1.0) ** n_up_odd

        psi = psi_abs * signs

        return psi / torch.norm(psi)

# ==========================================
# 2. MAIN SCRIPT
# ==========================================
if __name__ == "__main__":
    CHAIN_LENGTH = 10
    TARGET_DELTA = 1.00
    TRAIN_SAMPLES = 50_000 # 50k is enough if symmetric

    # Config
    ALPHA = 2          # Hidden density
    CD_K = 10          # CD steps
    EPOCHS = 100
    LR = 0.01

    # Load Data
    file_name = f"xxz_{CHAIN_LENGTH}_delta{TARGET_DELTA:.2f}_5000000.npz"
    file_path = data_dir / file_name
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[TRAIN_SAMPLES])
    # IMPORTANT: Shuffle is crucial for CD
    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=256, shuffle=True, drop_last=True, rng=rng_loader)

    # Calculate Data Czz
    print("Calculating Data Czz...", end="")
    total_czz = 0
    for b in loader:
        s = 2.0*b[0].float()-1.0
        total_czz += (s[:,:-1]*s[:,1:]).mean().item()
    data_czz = total_czz / len(loader)
    print(f" {data_czz:.4f}")

    # Train
    print("Training Symmetric RBM...")
    torch.manual_seed(42)
    rng = torch.Generator().manual_seed(42)
    model = SymmetricRBM(CHAIN_LENGTH, ALPHA, CD_K)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)

    hist_czz = []

    for epoch in range(EPOCHS):
        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch, {"rng": rng})
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Monitor
        if (epoch+1)%5 == 0:
            model.eval()
            gen = model._gibbs_step(torch.bernoulli(torch.full((2000, CHAIN_LENGTH), 0.5)), rng) # fast check
            for _ in range(10): gen = model._gibbs_step(gen, rng)
            s = 2.0*gen.float()-1.0
            czz = (s[:,:-1]*s[:,1:]).mean().item()
            model.train()
            hist_czz.append(czz)
            print(f"Ep {epoch+1}: Czz={czz:.4f} (Target {data_czz:.4f})")

    # Final Entropy Calculation
    print("\nCalculating Entropy...")
    psi = model.get_full_psi_with_marshall_sign() # <--- THIS FIXES THE ENTROPY

    s2_curve = []
    for l in range(1, CHAIN_LENGTH//2 + 1):
        dim_A = 2**l
        dim_B = 2**(CHAIN_LENGTH-l)
        mat = psi.view(dim_A, dim_B)
        S = torch.linalg.svdvals(mat)
        s2 = -math.log(torch.sum(S**4).item())
        s2_curve.append(s2)
        print(f"L={l}: S2={s2:.4f}")

    # Simple Plot
    plt.figure(figsize=(6,4))
    plt.plot(range(1, 6), s2_curve, 'r-o', label='Model (w/ Sign Rule)')
    plt.xlabel('L'); plt.ylabel('S2')
    plt.title(f'Final Entropy (Delta={TARGET_DELTA})')
    plt.grid(True)
    plt.show()