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

# === REFERENCE VALUES (L=10, Delta=1.0) ===
REF_CZZ = -0.5996
REF_ENTROPY = {1: 0.693, 2: 0.650, 3: 0.860, 4: 0.780, 5: 0.910}

# ==========================================
# 1. OPTIMIZED SYMMETRIC RBM
# ==========================================
class SymmetricRBM(nn.Module):
    def __init__(self, num_visible: int, alpha: int):
        super().__init__()
        self.num_visible = num_visible
        self.alpha = alpha
        self.T = 1.0

        # Weights: [1, Alpha, N]
        self.kernel = nn.Parameter(torch.empty(1, alpha, num_visible))
        self.b_scalar = nn.Parameter(torch.zeros(1))
        self.c_vector = nn.Parameter(torch.zeros(1, alpha))

        # Standard Initialization
        nn.init.normal_(self.kernel, mean=0.0, std=0.05)
        nn.init.constant_(self.b_scalar, 0.0)
        nn.init.constant_(self.c_vector, 0.0)

    def compute_energy_term(self, v: torch.Tensor):
        v_in = v.unsqueeze(1)
        v_padded = F.pad(v_in, (0, self.num_visible - 1), mode='circular')
        weight = self.kernel.view(self.alpha, 1, self.num_visible)
        return F.conv1d(v_padded, weight).view(v.shape[0], -1)

    def _free_energy(self, v: torch.Tensor) -> torch.Tensor:
        v = v.float()
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
# 2. TRAINING EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- HYPERPARAMETERS ---
    # Based on findings: Alpha 12, CD-1, Weak Decay, Batch 256
    ALPHA = 12
    CD_K = 1
    BATCH_SIZE = 256
    WEIGHT_DECAY = 1e-5

    # Short run to catch the "Physical Peak" before collapse
    EPOCHS = 100
    LR_START = 0.005

    # Load Data (20k is sufficient)
    file_name = f"xxz_10_delta1.00_5000000.npz"
    file_path = data_dir / file_name
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[20_000])

    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, rng=rng_loader)

    print(f"Training Final Config: Alpha={ALPHA}, CD={CD_K}, WD={WEIGHT_DECAY}, Epochs={EPOCHS}")

    torch.manual_seed(42)
    rng = torch.Generator().manual_seed(42)

    model = SymmetricRBM(10, alpha=ALPHA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_START, weight_decay=WEIGHT_DECAY)

    # Simple Step Decay to settle into the solution
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # History Tracking
    hist_czz = []
    hist_lr = []
    hist_loss = []

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch, {"rng": rng})
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Record Learning Rate
        current_lr = scheduler.get_last_lr()[0]
        hist_lr.append(current_lr)
        hist_loss.append(epoch_loss / len(loader))

        scheduler.step()

        # Measure Czz at EVERY epoch for the detailed plot
        model.eval()
        gen = model.generate(2000, 100, rng)
        s = 2.0*gen.float()-1.0
        czz = (s[:,:-1]*s[:,1:]).mean().item()
        model.train()

        hist_czz.append(czz)

        if (epoch+1) % 5 == 0:
            print(f"Ep {epoch+1}: Loss={hist_loss[-1]:.4f} | Czz={czz:.4f} (Ref {REF_CZZ:.4f}) | LR={current_lr:.5f}")

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

    # --- PLOTTING ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Czz History
    ax[0].plot(range(1, EPOCHS+1), hist_czz, 'b-o', markersize=4, label='Model')
    ax[0].axhline(REF_CZZ, color='g', linestyle='--', linewidth=2, label='Target (-0.60)')
    ax[0].set_title('Correlation (C_zz) Trajectory')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('C_zz')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    # 2. Learning Rate & Loss
    ax2_twin = ax[1].twinx()
    l1 = ax[1].plot(range(1, EPOCHS+1), hist_loss, 'r-', label='Loss')
    l2 = ax2_twin.plot(range(1, EPOCHS+1), hist_lr, 'k--', label='Learning Rate')
    ax[1].set_title('Training Dynamics')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Free Energy Loss')
    ax2_twin.set_ylabel('Learning Rate')

    # Legend trick
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax[1].legend(lns, labs, loc='center right')
    ax[1].grid(True, alpha=0.3)

    # 3. Entropy
    l_axis = list(REF_ENTROPY.keys())
    ax[2].plot(l_axis, list(REF_ENTROPY.values()), 'k--', linewidth=2, label='Exact Ref')
    ax[2].plot(l_axis, s2_curve, 'r-o', label='RBM Final')
    ax[2].set_title('Entanglement Entropy (S2)')
    ax[2].set_xlabel('Subsystem Size L')
    ax[2].set_ylim(0, 1.0)
    ax[2].legend()
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()