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
# 1. SYMMETRIC RBM (Convolutional)
# ==========================================
class SymmetricRBM(nn.Module):
    def __init__(self, num_visible: int, alpha: int, k: int):
        super().__init__()
        self.num_visible = num_visible
        self.alpha = alpha
        self.k = k
        self.T = 1.0

        # Weights: [1, Alpha, N]
        self.kernel = nn.Parameter(torch.empty(1, alpha, num_visible))
        self.b_scalar = nn.Parameter(torch.zeros(1))
        self.c_vector = nn.Parameter(torch.zeros(1, alpha))

        self.initialize_weights()

    def initialize_weights(self):
        # Keep initialization small
        nn.init.normal_(self.kernel, mean=0.0, std=0.01)
        nn.init.constant_(self.b_scalar, 0.0)
        nn.init.constant_(self.c_vector, 0.0)

    def compute_energy_term(self, v: torch.Tensor):
        # Circular Conv Implementation
        v_in = v.unsqueeze(1)
        v_padded = F.pad(v_in, (0, self.num_visible - 1), mode='circular')
        weight = self.kernel.view(self.alpha, 1, self.num_visible)
        activation = F.conv1d(v_padded, weight)
        return activation.view(v.shape[0], -1)

    def _free_energy(self, v: torch.Tensor) -> torch.Tensor:
        v = v.float()
        term1 = -self.b_scalar * v.sum(dim=-1)
        wv = self.compute_energy_term(v)
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

        # Backward (Transpose via Flip)
        w_back = torch.flip(self.kernel, dims=[-1]).view(1, self.alpha, self.num_visible)
        h_padded = F.pad(h, (0, self.num_visible - 1), mode='circular')
        wh = F.conv1d(h_padded, w_back).view(v.shape[0], self.num_visible)

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
    def generate(self, n_samples: int, burn_in: int, rng: torch.Generator):
        device = next(self.parameters()).device
        v = torch.bernoulli(torch.full((n_samples, self.num_visible), 0.5, device=device), generator=rng)
        for _ in range(burn_in):
            v = self._gibbs_step(v, rng)
        return v

    @torch.no_grad()
    def get_full_psi(self):
        device = next(self.parameters()).device
        N = self.num_visible
        indices = torch.arange(2**N, device=device).unsqueeze(1)
        powers = 2**torch.arange(N - 1, -1, -1, device=device)
        v_all = (indices.bitwise_and(powers) != 0).float()

        fe = self._free_energy(v_all)
        log_probs = -fe
        log_probs -= log_probs.max()
        probs = torch.exp(log_probs)
        psi = torch.sqrt(probs)
        return psi / torch.norm(psi)

# ==========================================
# 2. TRAINING
# ==========================================
if __name__ == "__main__":
    CHAIN_LENGTH = 10
    TARGET_DELTA = 1.00
    TRAIN_SAMPLES = 20_000

    # === SMOOTHING CONFIGURATION ===
    CD_K = 20
    ALPHA = 8          # INCREASED: More hidden units to capture "tails" of distribution
    EPOCHS = 300
    LR = 0.002
    BATCH_SIZE = 128
    WEIGHT_DECAY = 1e-4 # ADDED: Penalizes large weights to prevent "sharp" peaks (classical mode collapse)

    # Load Data
    file_name = f"xxz_{CHAIN_LENGTH}_delta{TARGET_DELTA:.2f}_5000000.npz"
    file_path = data_dir / file_name
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[TRAIN_SAMPLES])

    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, rng=rng_loader)

    # Baseline Czz
    print("Calculating Data Czz...", end="")
    total_czz = 0
    for b in loader:
        s = 2.0*b[0].float()-1.0
        total_czz += (s[:,:-1]*s[:,1:]).mean().item()
    data_czz = total_czz / len(loader)
    print(f" {data_czz:.4f}")

    # Train
    print(f"Training Symmetric RBM (Alpha={ALPHA}, WD={WEIGHT_DECAY})...")
    torch.manual_seed(42)
    rng = torch.Generator().manual_seed(42)

    model = SymmetricRBM(CHAIN_LENGTH, ALPHA, CD_K)
    # Weight decay added here
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    hist_czz = []

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch, {"rng": rng})
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if (epoch+1) % 20 == 0:
            model.eval()
            gen = model.generate(2000, 100, rng)
            s = 2.0*gen.float()-1.0
            czz = (s[:,:-1]*s[:,1:]).mean().item()
            model.train()
            hist_czz.append(czz)
            print(f"Ep {epoch+1}: Loss={epoch_loss/len(loader):.4f} | Czz={czz:.4f} (Target {data_czz:.4f})")

    # Final Entropy
    print("\nCalculating Entropy...")
    psi = model.get_full_psi()

    s2_curve = []
    for l in range(1, CHAIN_LENGTH//2 + 1):
        dim_A = 2**l
        dim_B = 2**(CHAIN_LENGTH-l)
        mat = psi.view(dim_A, dim_B)
        try: S = torch.linalg.svdvals(mat)
        except: S = torch.linalg.svdvals(mat.cpu())
        s2 = -math.log(torch.sum(S**4).item())
        s2_curve.append(s2)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Czz
    ax[0].plot(range(20, EPOCHS+1, 20), hist_czz, 'b-o', label='Model')
    ax[0].axhline(data_czz, color='g', linestyle='--', label='Data')
    ax[0].set_title(f'Correlation (Target: {data_czz:.4f})')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Entropy
    l_axis = list(range(1, len(s2_curve)+1))
    ax[1].plot(l_axis, s2_curve, 'r-o', label='RBM Model')

    ref_file = Path(f"xxz_{CHAIN_LENGTH}_entropy_ref.csv")
    if ref_file.exists():
        df = pd.read_csv(ref_file)
        mask = np.isclose(df['delta'], TARGET_DELTA)
        if mask.any():
            row = df[mask].iloc[0]
            l_cols = sorted([c for c in df.columns if c.startswith("l")], key=lambda x: int(x[1:]))
            ref = row[l_cols].to_numpy()
            ax[1].plot(l_axis, ref[:len(l_axis)], 'k--', label='Reference')

    ax[1].set_title('Entropy (S2)')
    ax[1].set_ylim(0, 1.0) # Set reasonable y-limit for entropy
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()