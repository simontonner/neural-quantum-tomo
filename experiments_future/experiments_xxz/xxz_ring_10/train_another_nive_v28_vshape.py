import os
import sys
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# === PATH SETUP ===
sys.path.append(str(Path("..").resolve()))
from data_handling import load_measurements_npz, MeasurementDataset, MeasurementLoader

data_dir = Path("measurements")

# ==========================================
# 1. STABLE VANILLA RBM
# ==========================================
class VanillaRBM(nn.Module):
    def __init__(self, num_visible: int, num_hidden: int):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.T = 1.0
        # Smaller init to prevent NaN start
        self.W = nn.Parameter(torch.empty(num_visible, num_hidden))
        self.v_bias = nn.Parameter(torch.zeros(num_visible))
        self.h_bias = nn.Parameter(torch.zeros(num_hidden))
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.W, mean=0.0, std=0.01)
        nn.init.constant_(self.v_bias, 0.0)
        nn.init.constant_(self.h_bias, 0.0)

    def _free_energy(self, v: torch.Tensor) -> torch.Tensor:
        v_term = v.matmul(self.v_bias)
        wx_b = v.matmul(self.W) + self.h_bias
        # Safe Softplus to avoid Inf
        h_term = F.softplus(wx_b).sum(dim=1)
        return -v_term - h_term

    def _gibbs_step(self, v: torch.Tensor, rng: torch.Generator):
        wx_b = v.matmul(self.W) + self.h_bias
        p_h = torch.sigmoid(wx_b / self.T)
        h = torch.bernoulli(p_h, generator=rng)
        wh_b = h.matmul(self.W.t()) + self.v_bias
        p_v = torch.sigmoid(wh_b / self.T)
        return torch.bernoulli(p_v, generator=rng)

    def forward(self, batch, aux_vars):
        v_data = batch[0].float()
        v_pos = torch.cat([v_data, 1.0 - v_data], dim=0) # Augmentation
        v_neg = self._gibbs_step(v_pos, aux_vars['rng']) # CD-1
        v_neg = v_neg.detach()
        return (self._free_energy(v_pos) - self._free_energy(v_neg)).mean()

# ==========================================
# 2. DIAGNOSTIC EXECUTION
# ==========================================
if __name__ == "__main__":
    # Settings
    N = 10
    HIDDEN = 32
    BATCH = 256
    EPOCHS = 40  # Short run to see the structure
    LR = 0.005

    # Load Data
    file_name = f"xxz_10_delta1.00_5000000.npz"
    file_path = data_dir / file_name
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[20_000])
    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=BATCH, shuffle=True, drop_last=True, rng=rng_loader)

    print(f"Training Diagnostic RBM...")
    torch.manual_seed(42)
    model = VanillaRBM(N, HIDDEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # Train loop
    for epoch in range(EPOCHS):
        for batch in loader:
            optimizer.zero_grad()
            model(batch, {"rng": torch.Generator()}).backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} complete.")

    # === THE LANDSCAPE SCAN ===
    print("Scanning Energy Landscape...")
    device = next(model.parameters()).device

    # 1. Generate ALL 1024 states
    indices = torch.arange(2**N, device=device).unsqueeze(1)
    powers = 2**torch.arange(N - 1, -1, -1, device=device)
    all_states = (indices.bitwise_and(powers) != 0).float()

    # 2. Compute Properties
    with torch.no_grad():
        fe = model._free_energy(all_states).cpu().numpy()
        # Calculate Magnetization (Number of Up spins - 5)
        mag = (all_states.sum(dim=1) - 5).cpu().numpy()

    # 3. PLOTTING
    plt.figure(figsize=(10, 6))

    # Scatter plot: Free Energy vs Magnetization
    # We add random jitter to x-axis so points don't overlap perfectly
    jitter = np.random.normal(0, 0.1, size=len(mag))

    plt.scatter(mag + jitter, fe, alpha=0.5, c=fe, cmap='viridis')
    plt.colorbar(label='Free Energy (Lower = More Probable)')

    plt.xlabel('Magnetization (Shifted from N/2)')
    plt.ylabel('Free Energy')
    plt.title('RBM Energy Landscape vs Magnetization Sector')
    plt.grid(True, alpha=0.3)

    # Draw "Walls"
    plt.axvline(0, color='r', linestyle='--', alpha=0.5, label='Physical Sector (Sz=0)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 4. RATIO CHECK (Corrected)
    # Compare Néel (Valid) vs Swapped-Neighbor (Valid) vs Flipped-Neighbor (Invalid)

    neel_state = torch.tensor([[1,0,1,0,1,0,1,0,1,0]], dtype=torch.float32) # Sz=0

    # Valid Neighbor: Swap bits 0 and 1 -> 0,1,1,0... (Sz=0)
    swap_neighbor = neel_state.clone()
    swap_neighbor[0,0] = 0; swap_neighbor[0,1] = 1

    # Invalid Neighbor: Flip bit 0 -> 0,0,1,0... (Sz=-1)
    flip_neighbor = neel_state.clone()
    flip_neighbor[0,0] = 0

    with torch.no_grad():
        fe_neel = model._free_energy(neel_state).item()
        fe_swap = model._free_energy(swap_neighbor).item()
        fe_flip = model._free_energy(flip_neighbor).item()

    # Probabilities relative to Neel
    p_swap = math.exp(fe_neel - fe_swap)
    p_flip = math.exp(fe_neel - fe_flip)

    print(f"\n--- SECTOR ANALYSIS ---")
    print(f"Prob(Valid Neighbor) / Prob(Néel)   : {p_swap:.4f} (Should be < 1.0 but > 0)")
    print(f"Prob(Invalid Neighbor) / Prob(Néel) : {p_flip:.4f} (Should be NEAR ZERO)")

    if p_flip > 0.01:
        print(">> CRITICAL ISSUE: The model leaks significant mass to invalid sectors.")
        print(">> This explains why Czz is weak (diluted by invalid states).")
    else:
        print(">> SECTOR INTEGRITY GOOD. Issue lies in intra-sector entropy.")