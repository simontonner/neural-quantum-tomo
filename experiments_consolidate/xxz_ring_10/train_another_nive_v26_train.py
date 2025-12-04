import os
import sys
from pathlib import Path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# === PATH SETUP ===
sys.path.append(str(Path("..").resolve()))
from data_handling import load_measurements_npz, MeasurementDataset, MeasurementLoader

data_dir = Path("measurements")
REF_CZZ = -0.5996
REF_ENTROPY = {5: 0.910}

# ==========================================
# 1. SYMMETRIC RBM (Stabilized)
# ==========================================
class SymmetricRBM(nn.Module):
    def __init__(self, num_visible: int, alpha: int):
        super().__init__()
        self.num_visible = num_visible
        self.alpha = alpha
        self.T = 1.0

        # Convolutional Weights
        self.kernel = nn.Parameter(torch.empty(1, alpha, num_visible))

        # Bias Fixed to 0 (Enforce Symmetry)
        self.b_scalar = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.c_vector = nn.Parameter(torch.zeros(1, alpha))

        self.initialize_weights()

    def initialize_weights(self):
        # Small init to prevent early explosion
        nn.init.normal_(self.kernel, mean=0.0, std=0.01)
        nn.init.constant_(self.c_vector, 0.0)

    def compute_energy_term(self, v: torch.Tensor):
        v_in = v.unsqueeze(1)
        v_padded = F.pad(v_in, (0, self.num_visible - 1), mode='circular')
        weight = self.kernel.view(self.alpha, 1, self.num_visible)
        return F.conv1d(v_padded, weight).view(v.shape[0], -1)

    def _free_energy(self, v: torch.Tensor) -> torch.Tensor:
        v = v.float()
        # b_scalar is 0, so visible term is 0
        wv = self.compute_energy_term(v)
        wv_r = wv.view(-1, self.alpha, self.num_visible)
        c_r = self.c_vector.view(1, self.alpha, 1)
        # Softplus is safe, but large inputs make free energy large negative
        term2 = -F.softplus(wv_r + c_r).sum(dim=(1, 2))
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
        return torch.bernoulli(p_v, generator=rng)

    def forward(self, batch, aux_vars):
        v_data = batch[0].float()

        # 1. Augmentation (Enforces P(v) = P(1-v))
        v_pos = torch.cat([v_data, 1.0 - v_data], dim=0)

        # 2. CD-1
        v_neg = self._gibbs_step(v_pos, aux_vars['rng'])
        v_neg = v_neg.detach()

        return (self._free_energy(v_pos) - self._free_energy(v_neg)).mean()

# ==========================================
# 2. ROBUST EXACT ANALYZER
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
            # 1. Robust Probability Calculation
            fe = self.model._free_energy(self.all_states)
            # Subtract min to prevent exp(huge) -> Inf
            log_probs = -fe
            log_probs = log_probs - log_probs.max()
            probs = torch.exp(log_probs)
            probs = probs / probs.sum() # Normalize

            # 2. Calculate Metrics
            # Valid Mass (Sz=0 sector)
            valid_mass = probs[self.mask_physical].sum().item()

            # Czz (Physical Sector Only)
            if valid_mass > 1e-6:
                probs_phys = probs * self.mask_physical.float()
                spins = 2.0 * self.all_states - 1.0
                czz_ops = (spins[:, :-1] * spins[:, 1:]).mean(dim=1)
                exact_czz = (probs_phys * czz_ops).sum().item() / valid_mass
            else:
                exact_czz = 0.0

            # Entropy (Safe SVD)
            psi = torch.sqrt(probs)
            dim = 2**(self.N//2)
            try:
                S = torch.linalg.svdvals(psi.view(dim, dim))
                s2 = -math.log(torch.sum(S**4).item())
            except Exception as e:
                print(f"SVD Warning: {e}")
                s2 = 0.0

            return exact_czz, s2, valid_mass

if __name__ == "__main__":
    # CONFIG (Safe Mode)
    ALPHA = 12       # Good balance
    BATCH = 256
    EPOCHS = 60
    LR = 0.002       # Lowered to prevent explosion
    WEIGHT_DECAY = 1e-4
    CLIP_GRAD = 1.0  # Clipping to prevent crash

    file_name = f"xxz_10_delta1.00_5000000.npz"
    file_path = data_dir / file_name
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[20_000])
    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=BATCH, shuffle=True, drop_last=True, rng=rng_loader)

    print(f"Training Stabilized Symmetric RBM (Alpha={ALPHA}, LR={LR})...")
    torch.manual_seed(42)

    model = SymmetricRBM(10, ALPHA)
    analyzer = RBMExactAnalyzer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = {'czz': [], 's2': []}

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch, {"rng": torch.Generator()})
            loss.backward()

            # === CRITICAL: CLIP GRADIENTS ===
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)

            optimizer.step()
            epoch_loss += loss.item()

        if (epoch+1) % 5 == 0:
            czz, s2, valid = analyzer.analyze()
            history['czz'].append(czz)
            history['s2'].append(s2)
            print(f"Ep {epoch+1}: Loss={epoch_loss:.2f} | Czz={czz:.4f} (Ref -0.60) | S2={s2:.4f} (Ref 0.91) | Valid={valid*100:.1f}%")

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(range(5, EPOCHS+1, 5), history['czz'], 'b-o')
    ax[0].axhline(REF_CZZ, color='g', linestyle='--')
    ax[0].set_title('Correlation (Czz)')

    ax[1].plot(range(5, EPOCHS+1, 5), history['s2'], 'r-o')
    ax[1].axhline(REF_ENTROPY[5], color='k', linestyle='--')
    ax[1].set_title('Entropy (S2)')
    plt.show()