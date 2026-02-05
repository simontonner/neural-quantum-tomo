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
# Ensure this points to your project root where data_handling.py is located
sys.path.append(str(Path("..").resolve()))
from data_handling import load_measurements_npz, MeasurementDataset, MeasurementLoader

data_dir = Path("measurements")

# === REFERENCE VALUES (L=10, Delta=1.0) ===
REF_CZZ = -0.5996
REF_ENTROPY = {
    1: 0.693,
    2: 0.650,
    3: 0.860,
    4: 0.780,
    5: 0.910
}

# ==========================================
# 1. SYMMETRIC RBM (Convolutional)
# ==========================================
class SymmetricRBM(nn.Module):
    def __init__(self, num_visible: int, alpha: int):
        super().__init__()
        self.num_visible = num_visible
        self.alpha = alpha
        self.T = 1.0

        # Convolutional Kernel: [1, Alpha, N]
        self.kernel = nn.Parameter(torch.empty(1, alpha, num_visible))

        # Fixed Bias 0 (Enforces Particle-Hole Symmetry)
        self.b_scalar = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.c_vector = nn.Parameter(torch.zeros(1, alpha))

        self.initialize_weights()

    def initialize_weights(self):
        # Small init for stability
        nn.init.normal_(self.kernel, mean=0.0, std=0.02)
        nn.init.constant_(self.c_vector, 0.0)

    def compute_energy_term(self, v: torch.Tensor):
        v_in = v.unsqueeze(1)
        # Circular padding for periodic boundary conditions
        v_padded = F.pad(v_in, (0, self.num_visible - 1), mode='circular')
        weight = self.kernel.view(self.alpha, 1, self.num_visible)
        return F.conv1d(v_padded, weight).view(v.shape[0], -1)

    def _free_energy(self, v: torch.Tensor) -> torch.Tensor:
        v = v.float()
        # Bias term is 0
        wv = self.compute_energy_term(v)
        wv_r = wv.view(-1, self.alpha, self.num_visible)
        c_r = self.c_vector.view(1, self.alpha, 1)

        # Free Energy F(v) = -T * Sum(Softplus(Input/T))
        term2 = -self.T * F.softplus((wv_r + c_r) / self.T).sum(dim=(1, 2))
        return term2

    def _gibbs_step(self, v: torch.Tensor, rng: torch.Generator):
        # Forward Pass (Visible -> Hidden)
        wv = self.compute_energy_term(v).view(-1, self.alpha, self.num_visible)
        c_r = self.c_vector.view(1, self.alpha, 1)
        p_h = torch.sigmoid((wv + c_r) / self.T)
        h = torch.bernoulli(p_h, generator=rng)

        # Backward Pass (Hidden -> Visible)
        # Flip kernel for transpose convolution (circular)
        w_back = torch.flip(self.kernel, dims=[-1]).view(1, self.alpha, self.num_visible)
        h_padded = F.pad(h, (0, self.num_visible - 1), mode='circular')
        wh = F.conv1d(h_padded, w_back).view(v.shape[0], self.num_visible)

        p_v = torch.sigmoid((wh + self.b_scalar) / self.T)

        return torch.bernoulli(p_v, generator=rng), p_v

    def forward(self, batch, aux_vars):
        v_data = batch[0].float()

        # 1. DATA AUGMENTATION
        # Force the model to see both v and (1-v) to prevent Symmetry Breaking
        v_pos = torch.cat([v_data, 1.0 - v_data], dim=0)

        # 2. CD-1 SAMPLING
        # Single step to maintain noise (Entropy)
        rng = aux_vars.get('rng', torch.Generator(device=v_data.device))
        v_neg, p_v_neg = self._gibbs_step(v_pos, rng)
        v_neg = v_neg.detach()

        # 3. CONTRASTIVE DIVERGENCE LOSS
        loss_cd = (self._free_energy(v_pos) - self._free_energy(v_neg)).mean()

        # 4. MAGNETIZATION PENALTY
        # Forces the model to stay in the physical N/2 sector
        expected_mag = p_v_neg.sum(dim=1)
        target_mag = self.num_visible / 2.0
        loss_mag = (expected_mag - target_mag).pow(2).mean()

        return loss_cd, loss_mag

    @torch.no_grad()
    def get_full_psi(self):
        """Calculates exact wavefunction amplitude vector."""
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
# 2. EXACT ANALYZER (Robust)
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
            # Robust Probability Calculation
            fe = self.model._free_energy(self.all_states)

            # Check for instability
            if torch.isnan(fe).any() or torch.isinf(fe).any():
                return 0.0, 0.0, 0.0

            log_probs = -fe
            log_probs -= log_probs.max()
            probs = torch.exp(log_probs)
            probs = probs / probs.sum()

            # Valid Mass (Check if model stays in Sz=0)
            valid_mass = probs[self.mask_physical].sum().item()

            # Exact Czz (Conditioned on Physical Sector)
            if valid_mass > 1e-9:
                probs_phys = probs * self.mask_physical.float()
                spins = 2.0 * self.all_states - 1.0
                czz_ops = (spins[:, :-1] * spins[:, 1:]).mean(dim=1)
                exact_czz = (probs_phys * czz_ops).sum().item() / valid_mass
            else:
                exact_czz = 0.0

            # Exact Entropy (L=5)
            # Use Try/Except to handle potential SVD instability in early epochs
            try:
                psi = torch.sqrt(probs)
                dim = 2**(self.N//2)
                mat = psi.view(dim, dim)

                # Move to CPU for SVD if CUDA is unstable
                if self.device.type == 'cuda':
                    mat = mat.cpu()

                S = torch.linalg.svdvals(mat)
                s2 = -math.log(torch.sum(S**4).item())
            except Exception as e:
                print(f"Warning: SVD skipped due to instability: {e}")
                s2 = 0.0

            return exact_czz, s2, valid_mass

# ==========================================
# 3. TRAINING SCRIPT
# ==========================================
if __name__ == "__main__":
    # === FINAL HYPERPARAMETERS ===
    ALPHA = 12
    BATCH = 256
    EPOCHS = 120
    LR = 0.005
    WEIGHT_DECAY = 5e-5 # Tuned for Czz -0.60
    LAMBDA_MAG = 1.0    # Sector Enforcer
    CLIP_VALUE = 1.0    # Prevents Weight Explosion

    # Load Data
    file_name = f"xxz_10_delta1.00_5000000.npz"
    file_path = data_dir / file_name

    print(f"Loading data from {file_path}...")
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[20_000])

    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=BATCH, shuffle=True, drop_last=True, rng=rng_loader)

    print(f"Training RBM (Alpha={ALPHA}, WD={WEIGHT_DECAY}, Epochs={EPOCHS})...")

    torch.manual_seed(42)
    rng_train = torch.Generator().manual_seed(42)

    model = SymmetricRBM(10, ALPHA)
    analyzer = RBMExactAnalyzer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    history = {'czz': [], 's2': []}

    # === TRAINING LOOP ===
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            # Forward pass returns tuple (CD_Loss, Mag_Loss)
            lcd, lmag = model(batch, {"rng": rng_train})
            loss = lcd + LAMBDA_MAG * lmag
            loss.backward()

            # === CRITICAL FIX: GRADIENT CLIPPING ===
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_VALUE)

            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Log Progress
        if (epoch+1) % 10 == 0:
            czz, s2, valid = analyzer.analyze()
            history['czz'].append(czz)
            history['s2'].append(s2)
            print(f"Ep {epoch+1}: Czz={czz:.4f} (Ref {REF_CZZ:.4f}) | S2={s2:.4f} (Ref 0.91) | Valid={valid*100:.1f}%")

    # ==========================================
    # 4. FINAL FULL SPECTRUM ANALYSIS
    # ==========================================
    print("\n=== FINAL ENTANGLEMENT SPECTRUM (L=1..5) ===")

    # Get exact wavefunction from model
    psi = model.get_full_psi()

    model_s2 = []
    ref_s2_list = []
    l_list = sorted(REF_ENTROPY.keys())

    print(f"{'L':<5} | {'Model S2':<10} | {'Ref S2':<10} | {'Error %':<10}")
    print("-" * 45)

    for l in l_list:
        dim_A = 2**l
        dim_B = 2**(10-l)
        mat = psi.view(dim_A, dim_B)

        try:
            # Move to CPU for final analysis to avoid CUDA SVD issues
            S = torch.linalg.svdvals(mat.cpu())
            val = -math.log(torch.sum(S**4).item())
        except Exception as e:
            val = 0.0
            print(f"Error computing S2 for L={l}: {e}")

        model_s2.append(val)

        ref = REF_ENTROPY[l]
        ref_s2_list.append(ref)
        err = abs(val - ref)/ref * 100

        print(f"{l:<5} | {val:.4f}     | {ref:.4f}     | {err:.2f}%")

    # ==========================================
    # 5. PLOTTING
    # ==========================================
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Training Trajectory (Phase Space)
    ax[0].plot(history['czz'], history['s2'], 'b-o', alpha=0.6, label='Trajectory')
    if len(history['czz']) > 0:
        ax[0].scatter(history['czz'][0], history['s2'][0], c='green', s=100, label='Start')
        ax[0].scatter(history['czz'][-1], history['s2'][-1], c='blue', s=100, label='End')
    ax[0].scatter([REF_CZZ], [REF_ENTROPY[5]], color='red', s=300, marker='*', label='TARGET', zorder=10)

    ax[0].set_xlabel('Correlation C_zz')
    ax[0].set_ylabel('Entropy S_2 (L=5)')
    ax[0].set_title('Convergence to Critical Point')
    ax[0].grid(True, alpha=0.5)
    ax[0].legend()

    # Plot 2: Full Entanglement Spectrum
    ax[1].plot(l_list, ref_s2_list, 'k--o', label='Exact Ground Truth', linewidth=2)
    ax[1].plot(l_list, model_s2, 'r-o', label='RBM Prediction', linewidth=2, alpha=0.8)

    ax[1].set_title('Entanglement Spectrum Scaling')
    ax[1].set_xlabel('Subsystem Size L')
    ax[1].set_ylabel('Renyi Entropy S2')
    ax[1].set_xticks(l_list)
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    plt.tight_layout()
    plt.show()