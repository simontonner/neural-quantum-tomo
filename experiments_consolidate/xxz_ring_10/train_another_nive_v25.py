import os
import sys
from pathlib import Path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append(str(Path("..").resolve()))
from data_handling import load_measurements_npz, MeasurementDataset, MeasurementLoader

data_dir = Path("measurements")
REF_CZZ = -0.5996
REF_ENTROPY = {1: 0.693, 2: 0.650, 3: 0.860, 4: 0.780, 5: 0.910}

# ==========================================
# 1. VANILLA RBM (Standard)
# ==========================================
class VanillaRBM(nn.Module):
    def __init__(self, num_visible: int, num_hidden: int):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.T = 1.0

        self.W = nn.Parameter(torch.empty(num_visible, num_hidden))
        # Fixed bias to help symmetry
        self.v_bias = nn.Parameter(torch.zeros(num_visible), requires_grad=False)
        self.h_bias = nn.Parameter(torch.zeros(num_hidden))

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.W, mean=0.0, std=0.02)
        nn.init.constant_(self.h_bias, 0.0)

    def _free_energy(self, v: torch.Tensor) -> torch.Tensor:
        v_term = v.matmul(self.v_bias)
        wx_b = v.matmul(self.W) + self.h_bias
        h_term = F.softplus(wx_b).sum(dim=1)
        return -v_term - h_term

    def _gibbs_step(self, v: torch.Tensor, rng: torch.Generator):
        # Visible -> Hidden
        wx_b = v.matmul(self.W) + self.h_bias
        p_h = torch.sigmoid(wx_b / self.T)
        h = torch.bernoulli(p_h, generator=rng)

        # Hidden -> Visible
        wh_b = h.matmul(self.W.t()) + self.v_bias
        p_v = torch.sigmoid(wh_b / self.T) # We need this prob for the penalty
        v_next = torch.bernoulli(p_v, generator=rng)

        return v_next, p_v # Return probability too

    def forward(self, batch, aux_vars):
        v_data = batch[0].float()

        # 1. Augmentation
        v_pos = torch.cat([v_data, 1.0 - v_data], dim=0)

        rng = aux_vars['rng']

        # 2. CD-1
        v_neg, p_v_neg = self._gibbs_step(v_pos, rng)
        v_neg = v_neg.detach()

        # 3. Standard CD Loss
        loss_cd = (self._free_energy(v_pos) - self._free_energy(v_neg)).mean()

        # 4. MAGNETIZATION PENALTY (New)
        # We want the expected magnetization of the reconstruction to be N/2 (5.0)
        # p_v_neg is [Batch, N]
        expected_mag = p_v_neg.sum(dim=1) # [Batch]
        target_mag = self.num_visible / 2.0

        # Strong penalty to force physics
        mag_loss = (expected_mag - target_mag).pow(2).mean()

        return loss_cd, mag_loss

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

            # Mag Check
            mag_op = (2.0 * self.all_states - 1.0).sum(dim=1)
            avg_mag_sq = (probs * (mag_op ** 2)).sum().item()

            valid_mass = probs[self.mask_physical].sum().item()

            if valid_mass > 1e-6:
                probs_phys = probs * self.mask_physical.float()
                spins = 2.0 * self.all_states - 1.0
                czz_ops = (spins[:, :-1] * spins[:, 1:]).mean(dim=1)
                exact_czz = (probs_phys * czz_ops).sum().item() / valid_mass
            else:
                exact_czz = 0.0

            psi = torch.sqrt(probs)
            dim_A = 2**5
            mat = psi.view(dim_A, dim_A)
            S = torch.linalg.svdvals(mat)
            s2 = -math.log(torch.sum(S**4).item())

            return exact_czz, s2, valid_mass, avg_mag_sq

if __name__ == "__main__":
    # CONFIG
    N_HIDDEN = 64
    WEIGHT_DECAY = 1e-4 # Reduced slightly, relying on Mag Penalty for regularization
    BATCH_SIZE = 256
    EPOCHS = 100
    LR = 0.005

    # === NEW PARAM ===
    LAMBDA_MAG = 0.5 # Strength of Magnetization Penalty

    file_name = f"xxz_10_delta1.00_5000000.npz"
    file_path = data_dir / file_name
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[20_000])
    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, rng=rng_loader)

    print(f"Training RBM (Hidden={N_HIDDEN}, MagPenalty={LAMBDA_MAG})...")
    torch.manual_seed(42)

    model = VanillaRBM(num_visible=10, num_hidden=N_HIDDEN)
    analyzer = RBMExactAnalyzer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    hist_czz = []
    hist_s2 = []
    hist_valid = []

    for epoch in range(EPOCHS):
        epoch_cd_loss = 0.0
        epoch_mag_loss = 0.0

        for batch in loader:
            optimizer.zero_grad()
            loss_cd, loss_mag = model(batch, {"rng": torch.Generator()})

            # Total Loss
            loss = loss_cd + LAMBDA_MAG * loss_mag

            loss.backward()
            optimizer.step()

            epoch_cd_loss += loss_cd.item()
            epoch_mag_loss += loss_mag.item()

        scheduler.step()

        if (epoch+1) % 5 == 0:
            czz, s2, valid, mag2 = analyzer.analyze()
            hist_czz.append(czz)
            hist_s2.append(s2)
            hist_valid.append(valid*100)

            print(f"Ep {epoch+1}: CD_Loss={epoch_cd_loss:.2f} | Mag_Loss={epoch_mag_loss:.2f}")
            print(f"  > Czz: {czz:.4f} (Ref -0.60) | S2: {s2:.4f} (Ref 0.91)")
            print(f"  > Valid Mass: {valid*100:.1f}% | <M^2>: {mag2:.2f}")

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(range(5, EPOCHS+1, 5), hist_czz, 'b-o')
    ax[0].axhline(REF_CZZ, color='g', linestyle='--')
    ax[0].set_title('Correlation')

    ax[1].plot(range(5, EPOCHS+1, 5), hist_s2, 'r-o')
    ax[1].axhline(REF_ENTROPY[5], color='k', linestyle='--')
    ax[1].set_title('Entropy')

    ax[2].plot(range(5, EPOCHS+1, 5), hist_valid, 'k-o')
    ax[2].set_title('Valid Sector %')

    plt.show()