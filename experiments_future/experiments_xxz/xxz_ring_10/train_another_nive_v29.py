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
# 1. SYMMETRIC RBM (With Mag Penalty Support)
# ==========================================
class SymmetricRBM(nn.Module):
    def __init__(self, num_visible: int, alpha: int):
        super().__init__()
        self.num_visible = num_visible
        self.alpha = alpha
        self.T = 1.0 # Temperature Knob

        self.kernel = nn.Parameter(torch.empty(1, alpha, num_visible))
        # Fixed Bias 0
        self.b_scalar = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.c_vector = nn.Parameter(torch.zeros(1, alpha))

        nn.init.normal_(self.kernel, mean=0.0, std=0.025)
        nn.init.constant_(self.c_vector, 0.0)

    def compute_energy_term(self, v: torch.Tensor):
        v_in = v.unsqueeze(1)
        v_padded = F.pad(v_in, (0, self.num_visible - 1), mode='circular')
        weight = self.kernel.view(self.alpha, 1, self.num_visible)
        return F.conv1d(v_padded, weight).view(v.shape[0], -1)

    def _free_energy(self, v: torch.Tensor) -> torch.Tensor:
        # F(v) = -T * ln(Z_h)
        # But for training we usually ignore T in Free Energy def,
        # however for correct sampling P = exp(-F/T), F must match.
        # Standard RBM Free Energy: -b'v - sum ln(1 + exp(Wv+c))
        # With T: -b'v - T * sum ln(1 + exp((Wv+c)/T))

        v = v.float()
        # Bias is 0
        wv = self.compute_energy_term(v)
        wv_r = wv.view(-1, self.alpha, self.num_visible)
        c_r = self.c_vector.view(1, self.alpha, 1)

        # Input to softplus
        x = (wv_r + c_r) / self.T

        # FE = - T * Sum Softplus(x)
        term2 = -self.T * F.softplus(x).sum(dim=(1, 2))
        return term2

    def _gibbs_step(self, v: torch.Tensor, rng: torch.Generator):
        # Forward
        wv = self.compute_energy_term(v).view(-1, self.alpha, self.num_visible)
        c_r = self.c_vector.view(1, self.alpha, 1)

        # Prob H: sigmoid( (Wv+c)/T )
        p_h = torch.sigmoid((wv + c_r) / self.T)
        h = torch.bernoulli(p_h, generator=rng)

        # Backward
        w_back = torch.flip(self.kernel, dims=[-1]).view(1, self.alpha, self.num_visible)
        h_padded = F.pad(h, (0, self.num_visible - 1), mode='circular')
        wh = F.conv1d(h_padded, w_back).view(v.shape[0], self.num_visible)

        # Prob V: sigmoid( (Wh+b)/T )
        p_v = torch.sigmoid((wh + self.b_scalar) / self.T)
        v_next = torch.bernoulli(p_v, generator=rng)

        return v_next, p_v

    def forward(self, batch, aux_vars):
        v_data = batch[0].float()
        v_pos = torch.cat([v_data, 1.0 - v_data], dim=0) # Augmentation

        # CD-1
        v_neg, p_v_neg = self._gibbs_step(v_pos, aux_vars['rng'])
        v_neg = v_neg.detach()

        # Standard Loss
        loss_cd = (self._free_energy(v_pos) - self._free_energy(v_neg)).mean()

        # Mag Penalty
        expected_mag = p_v_neg.sum(dim=1)
        target_mag = self.num_visible / 2.0
        loss_mag = (expected_mag - target_mag).pow(2).mean()

        return loss_cd, loss_mag

# ==========================================
# 2. EXACT ANALYZER (Temperature Aware)
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

    def analyze(self, temp=1.0):
        # Temporarily set model temperature
        old_T = self.model.T
        self.model.T = temp

        with torch.no_grad():
            fe = self.model._free_energy(self.all_states)
            # P(v) = exp(-F(v)/T) -- wait, free_energy function already handles T scaling?
            # Looking at code: _free_energy returns -T * Softplus(x/T).
            # So P(v) = exp( - (-T*Softplus) / T ) = exp(Softplus)
            # This logic cancels out. Let's rely on the definition: P ~ exp(-FE/T)
            # BUT: My _free_energy implementation ALREADY scales by -T.
            # Let's verify: F = -T ln Z. -> P = exp(-F/T).

            # The _free_energy function I wrote returns: -T * sum(softplus((Wx+c)/T))
            # So -F/T = sum(softplus((Wx+c)/T)).
            # This looks correct.

            log_probs = -fe / temp
            log_probs -= log_probs.max()
            probs = torch.exp(log_probs)
            probs = probs / probs.sum()

            # 1. Valid Mass
            valid_mass = probs[self.mask_physical].sum().item()

            # 2. Czz (Physical)
            if valid_mass > 1e-9:
                probs_phys = probs * self.mask_physical.float()
                spins = 2.0 * self.all_states - 1.0
                czz_ops = (spins[:, :-1] * spins[:, 1:]).mean(dim=1)
                exact_czz = (probs_phys * czz_ops).sum().item() / valid_mass
            else:
                exact_czz = 0.0

            # 3. Entropy
            psi = torch.sqrt(probs)
            dim = 2**(self.N//2)
            S = torch.linalg.svdvals(psi.view(dim, dim))
            s2 = -math.log(torch.sum(S**4).item())

        self.model.T = old_T
        return exact_czz, s2, valid_mass

if __name__ == "__main__":
    # CONFIG
    ALPHA = 12
    BATCH = 256
    EPOCHS = 100     # Train long enough to settle
    LR = 0.005
    LAMBDA_MAG = 1.0 # Strong penalty to keep V shape

    file_name = f"xxz_10_delta1.00_5000000.npz"
    file_path = data_dir / file_name
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[20_000])
    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=BATCH, shuffle=True, drop_last=True, rng=rng_loader)

    print(f"Training Symmetric RBM (Alpha={ALPHA}, MagPenalty={LAMBDA_MAG})...")
    torch.manual_seed(42)
    model = SymmetricRBM(10, ALPHA)
    analyzer = RBMExactAnalyzer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # TRAIN
    for epoch in range(EPOCHS):
        for batch in loader:
            optimizer.zero_grad()
            lcd, lmag = model(batch, {"rng": torch.Generator()})
            loss = lcd + LAMBDA_MAG * lmag
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch+1)%10 == 0:
            czz, s2, v = analyzer.analyze(temp=1.0)
            print(f"Ep {epoch+1}: Czz={czz:.4f} | S2={s2:.4f} | Valid={v*100:.1f}%")

    print("\n=== POST-TRAINING TEMPERATURE SWEEP ===")
    print("We vary T to see if the Critical State exists at T != 1.0")

    temps = np.linspace(0.5, 2.0, 30)
    res_czz = []
    res_s2 = []

    print(f"{'Temp':<6} | {'Czz':<8} | {'S2':<8}")
    print("-" * 26)

    for t in temps:
        czz, s2, _ = analyzer.analyze(temp=t)
        res_czz.append(czz)
        res_s2.append(s2)
        print(f"{t:.2f}   | {czz:.4f}   | {s2:.4f}")

    # PLOT SWEEP
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Trajectory
    ax.plot(res_czz, res_s2, 'b-o', label='RBM at varying T')

    # Target Point
    ax.scatter([REF_CZZ], [REF_ENTROPY[5]], color='red', s=200, marker='*', label='Target State')

    # Annotate Temps
    for i, t in enumerate(temps):
        if i % 3 == 0:
            ax.annotate(f"T={t:.1f}", (res_czz[i], res_s2[i]), fontsize=8)

    ax.set_xlabel('Correlation C_zz')
    ax.set_ylabel('Entropy S_2')
    ax.set_title('Phase Space Trajectory via Temperature Scaling')
    ax.grid(True)
    ax.legend()
    plt.show()