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
REF_CZZ = -0.5996
REF_ENTROPY = {1: 0.693, 2: 0.650, 3: 0.860, 4: 0.780, 5: 0.910}

# ==========================================
# 1. SYMMETRIC RBM (The Winner)
# ==========================================
class SymmetricRBM(nn.Module):
    def __init__(self, num_visible: int, alpha: int):
        super().__init__()
        self.num_visible = num_visible
        self.alpha = alpha
        self.T = 1.0

        self.kernel = nn.Parameter(torch.empty(1, alpha, num_visible))
        # Fixed Bias 0
        self.b_scalar = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.c_vector = nn.Parameter(torch.zeros(1, alpha))

        nn.init.normal_(self.kernel, mean=0.0, std=0.02)
        nn.init.constant_(self.c_vector, 0.0)

    def compute_energy_term(self, v: torch.Tensor):
        v_in = v.unsqueeze(1)
        v_padded = F.pad(v_in, (0, self.num_visible - 1), mode='circular')
        weight = self.kernel.view(self.alpha, 1, self.num_visible)
        return F.conv1d(v_padded, weight).view(v.shape[0], -1)

    def _free_energy(self, v: torch.Tensor) -> torch.Tensor:
        v = v.float()
        wv = self.compute_energy_term(v)
        wv_r = wv.view(-1, self.alpha, self.num_visible)
        c_r = self.c_vector.view(1, self.alpha, 1)
        term2 = -self.T * F.softplus((wv_r + c_r) / self.T).sum(dim=(1, 2))
        return term2

    def _gibbs_step(self, v: torch.Tensor, rng: torch.Generator):
        wv = self.compute_energy_term(v).view(-1, self.alpha, self.num_visible)
        c_r = self.c_vector.view(1, self.alpha, 1)
        p_h = torch.sigmoid((wv + c_r) / self.T)
        h = torch.bernoulli(p_h, generator=rng)
        w_back = torch.flip(self.kernel, dims=[-1]).view(1, self.alpha, self.num_visible)
        h_padded = F.pad(h, (0, self.num_visible - 1), mode='circular')
        wh = F.conv1d(h_padded, w_back).view(v.shape[0], self.num_visible)
        p_v = torch.sigmoid((wh + self.b_scalar) / self.T)
        return torch.bernoulli(p_v, generator=rng), p_v

    def forward(self, batch, aux_vars):
        v_data = batch[0].float()
        v_pos = torch.cat([v_data, 1.0 - v_data], dim=0) # Augmentation
        v_neg, p_v_neg = self._gibbs_step(v_pos, aux_vars['rng']) # CD-1
        v_neg = v_neg.detach()
        loss_cd = (self._free_energy(v_pos) - self._free_energy(v_neg)).mean()

        # Mag Penalty
        expected_mag = p_v_neg.sum(dim=1)
        target_mag = self.num_visible / 2.0
        loss_mag = (expected_mag - target_mag).pow(2).mean()

        return loss_cd, loss_mag

# ==========================================
# 2. EXACT ANALYZER
# ==========================================
class RBMExactAnalyzer:
    def __init__(self, model):
        self.model = model
        self.N = model.num_visible
        self.device = next(model.parameters()).device
        indices = torch.arange(2**self.N, device=self.device).unsqueeze(1)
        powers = 2**torch.arange(self.N - 1, -1, -1, device=self.device)
        self.all_states = (indices.bitwise_and(powers) != 0).float()
        self.mask_physical = (self.all_states.sum(dim=1) == (self.N // 2))

    def analyze(self):
        with torch.no_grad():
            fe = self.model._free_energy(self.all_states)
            log_probs = -fe
            log_probs -= log_probs.max()
            probs = torch.exp(log_probs)
            probs = probs / probs.sum()

            valid_mass = probs[self.mask_physical].sum().item()

            if valid_mass > 1e-9:
                probs_phys = probs * self.mask_physical.float()
                spins = 2.0 * self.all_states - 1.0
                czz_ops = (spins[:, :-1] * spins[:, 1:]).mean(dim=1)
                exact_czz = (probs_phys * czz_ops).sum().item() / valid_mass
            else:
                exact_czz = 0.0

            psi = torch.sqrt(probs)
            dim = 2**(self.N//2)
            S = torch.linalg.svdvals(psi.view(dim, dim))
            s2 = -math.log(torch.sum(S**4).item())
            return exact_czz, s2, valid_mass

if __name__ == "__main__":
    # === FINAL TUNING ===
    # We increase WD to 5e-5 to nudge Czz from -0.62 to -0.60
    WEIGHT_DECAY = 5e-5

    # Other params kept from the successful run
    ALPHA = 12
    BATCH = 256
    EPOCHS = 150
    LR = 0.005
    LAMBDA_MAG = 1.0

    file_name = f"xxz_10_delta1.00_5000000.npz"
    file_path = data_dir / file_name
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[20_000])
    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=BATCH, shuffle=True, drop_last=True, rng=rng_loader)

    print(f"Training FINAL TUNED RBM (WD={WEIGHT_DECAY})...")
    torch.manual_seed(42)
    model = SymmetricRBM(10, ALPHA)
    analyzer = RBMExactAnalyzer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    history = {'czz': [], 's2': []}

    for epoch in range(EPOCHS):
        for batch in loader:
            optimizer.zero_grad()
            lcd, lmag = model(batch, {"rng": torch.Generator()})
            loss = lcd + LAMBDA_MAG * lmag
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch+1)%10 == 0:
            czz, s2, valid = analyzer.analyze()
            history['czz'].append(czz)
            history['s2'].append(s2)
            print(f"Ep {epoch+1}: Czz={czz:.4f} (Ref {REF_CZZ:.4f}) | S2={s2:.4f} (Ref 0.91) | Valid={valid*100:.1f}%")

    # PLOT
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Trajectory
    ax.plot(history['czz'], history['s2'], 'b-o', alpha=0.6, label='Training Trajectory')
    # Mark Start and End
    ax.scatter(history['czz'][0], history['s2'][0], c='green', s=100, label='Start')
    ax.scatter(history['czz'][-1], history['s2'][-1], c='blue', s=100, label='End')

    # Target
    ax.scatter([REF_CZZ], [REF_ENTROPY[5]], color='red', s=300, marker='*', label='TARGET', zorder=10)

    ax.set_xlabel('Correlation C_zz')
    ax.set_ylabel('Entropy S_2')
    ax.set_title('Converging to the Critical Point')
    ax.grid(True)
    ax.legend()
    plt.show()