import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple
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

# === REFERENCE VALUES ===
REF_CZZ = -0.5996
REF_ENTROPY = {1: 0.693, 2: 0.650, 3: 0.860, 4: 0.780, 5: 0.910}

# ==========================================
# 1. SYMMETRIC RBM (Fixed Bias)
# ==========================================
class SymmetricRBM(nn.Module):
    def __init__(self, num_visible: int, alpha: int):
        super().__init__()
        self.num_visible = num_visible
        self.alpha = alpha
        self.T = 1.0

        # Weights: [1, Alpha, N]
        self.kernel = nn.Parameter(torch.empty(1, alpha, num_visible))

        # === FIX 1: FREEZE BIAS ===
        # The Hamiltonian has Particle-Hole symmetry (Up <-> Down).
        # A non-zero bias would break this and cause magnetization drift.
        # We freeze it to exactly 0.0.
        self.b_scalar = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.c_vector = nn.Parameter(torch.zeros(1, alpha))

        # Initialization
        nn.init.normal_(self.kernel, mean=0.0, std=0.05)
        nn.init.constant_(self.c_vector, 0.0)

    def compute_energy_term(self, v: torch.Tensor):
        v_in = v.unsqueeze(1)
        v_padded = F.pad(v_in, (0, self.num_visible - 1), mode='circular')
        weight = self.kernel.view(self.alpha, 1, self.num_visible)
        return F.conv1d(v_padded, weight).view(v.shape[0], -1)

    def _free_energy(self, v: torch.Tensor) -> torch.Tensor:
        v = v.float()
        # term1 is always 0 because b_scalar is 0
        term1 = -self.b_scalar * v.sum(dim=-1)
        wv = self.compute_energy_term(v)
        wv_r = wv.view(-1, self.alpha, self.num_visible)
        c_r = self.c_vector.view(1, self.alpha, 1)
        term2 = -F.softplus(wv_r + c_r).sum(dim=(1, 2))
        return term1 + term2

    def _gibbs_step(self, v: torch.Tensor, rng: torch.Generator):
        # CD-1 Step
        wv = self.compute_energy_term(v).view(-1, self.alpha, self.num_visible)
        c_r = self.c_vector.view(1, self.alpha, 1)
        p_h = torch.sigmoid((wv + c_r) / self.T)
        h = torch.bernoulli(p_h, generator=rng)

        w_back = torch.flip(self.kernel, dims=[-1]).view(1, self.alpha, self.num_visible)
        h_padded = F.pad(h, (0, self.num_visible - 1), mode='circular')
        wh = F.conv1d(h_padded, w_back).view(v.shape[0], self.num_visible)
        p_v = torch.sigmoid((wh + self.b_scalar) / self.T)
        return torch.bernoulli(p_v, generator=rng)

    def forward(self, batch: Tuple[torch.Tensor, ...], aux_vars: Dict[str, Any]):
        values, _, _ = batch
        v_data = values.to(device=self.kernel.device).float()
        rng = aux_vars.get("rng", torch.Generator(device="cpu"))

        # CD-1 Training
        v_model = self._gibbs_step(v_data, rng)
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
        log_probs = -fe - (-fe).max()
        probs = torch.exp(log_probs)
        psi = torch.sqrt(probs)
        return psi / torch.norm(psi)

# ==========================================
# 2. TRAINING WITH POST-SELECTION
# ==========================================
if __name__ == "__main__":
    # Hyperparameters
    ALPHA = 24
    CD_K = 1
    BATCH_SIZE = 256
    WEIGHT_DECAY = 1e-5
    EPOCHS = 45
    LR = 0.005

    # Load Data
    file_name = f"xxz_10_delta1.00_5000000.npz"
    file_path = data_dir / file_name
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[20_000])

    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, rng=rng_loader)

    print(f"Training RBM (Alpha={ALPHA}, CD={CD_K}, Bias=Fixed 0.0)...")

    torch.manual_seed(42)
    rng = torch.Generator().manual_seed(42)

    model = SymmetricRBM(10, alpha=ALPHA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    hist_czz = []
    hist_valid_pct = []

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch, {"rng": rng})
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # === EVALUATION WITH POST-SELECTION ===
        if (epoch+1) % 5 == 0:
            model.eval()
            # Generate extra samples because we will discard the unphysical ones
            gen_raw = model.generate(5000, 100, rng)

            # Filter: Keep only samples with exactly N/2 up spins (Conservation Law)
            magnetization = gen_raw.sum(dim=1)
            target_mag = model.num_visible // 2
            mask = (magnetization == target_mag)

            gen_physical = gen_raw[mask]
            valid_pct = len(gen_physical) / len(gen_raw) * 100

            if len(gen_physical) > 10:
                s = 2.0 * gen_physical.float() - 1.0
                czz = (s[:,:-1] * s[:,1:]).mean().item()
            else:
                czz = 0.0 # Failed to generate physical states

            model.train()

            hist_czz.append(czz)
            hist_valid_pct.append(valid_pct)

            print(f"Ep {epoch+1}: Loss={epoch_loss/len(loader):.4f} | Czz={czz:.4f} (Ref {REF_CZZ:.4f}) | Valid={valid_pct:.1f}%")

    # --- FINAL ANALYSIS ---
    print("\nCalculating Final Entropy...")
    psi = model.get_full_psi()

    s2_curve = []
    for l in range(1, 6):
        dim_A = 2**l
        dim_B = 2**(10-l)
        mat = psi.view(dim_A, dim_B)
        try: S = torch.linalg.svdvals(mat)
        except: S = torch.linalg.svdvals(mat.cpu())
        s2 = -math.log(torch.sum(S**4).item())
        s2_curve.append(s2)

    # PLOT
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Czz (Physical)
    ax[0].plot(range(5, EPOCHS+1, 5), hist_czz, 'b-o', label='Filtered Czz')
    ax[0].axhline(REF_CZZ, color='g', linestyle='--', label='Target')
    ax[0].set_title('Correlation (Physical Sector)')
    ax[0].set_ylim(-0.7, -0.4)
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # 2. Validity (Drift Check)
    ax[1].plot(range(5, EPOCHS+1, 5), hist_valid_pct, 'r-o')
    ax[1].set_title('% Samples in S_z=0 Sector')
    ax[1].set_ylim(0, 100)
    ax[1].grid(True, alpha=0.3)

    # 3. Entropy
    l_axis = list(REF_ENTROPY.keys())
    ax[2].plot(l_axis, list(REF_ENTROPY.values()), 'k--', label='Ref')
    ax[2].plot(l_axis, s2_curve, 'r-o', label='Model')
    ax[2].set_title('Entropy S2')
    ax[2].legend()
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()