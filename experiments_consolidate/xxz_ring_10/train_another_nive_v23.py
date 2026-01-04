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
# VANILLA RBM (Same Class)
# ==========================================
class VanillaRBM(nn.Module):
    def __init__(self, num_visible: int, num_hidden: int):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.T = 1.0
        self.W = nn.Parameter(torch.empty(num_visible, num_hidden))
        self.v_bias = nn.Parameter(torch.zeros(num_visible), requires_grad=False)
        self.h_bias = nn.Parameter(torch.zeros(num_hidden))
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.W, mean=0.0, std=0.05) # Slightly higher std for small H
        nn.init.constant_(self.h_bias, 0.0)

    def _free_energy(self, v: torch.Tensor) -> torch.Tensor:
        v_term = v.matmul(self.v_bias)
        wx_b = v.matmul(self.W) + self.h_bias
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
        rng = aux_vars['rng']
        v_neg = self._gibbs_step(v_pos, rng) # CD-1
        v_neg = v_neg.detach()
        return (self._free_energy(v_pos) - self._free_energy(v_neg)).mean()

# ==========================================
# EXACT ANALYZER
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
            return exact_czz, s2, valid_mass

if __name__ == "__main__":
    # === THE "BOTTLENECK" CONFIG ===
    # Drastically reduced capacity to force generalization
    N_HIDDEN = 20

    # Weight Decay is mandatory now to keep the landscape flat
    WEIGHT_DECAY = 1e-4

    BATCH_SIZE = 256
    EPOCHS = 300
    LR = 0.01

    file_name = f"xxz_10_delta1.00_5000000.npz"
    file_path = data_dir / file_name
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[20_000])
    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, rng=rng_loader)

    print(f"Training CONSTRAINED RBM (Hidden={N_HIDDEN}, WD={WEIGHT_DECAY})...")
    torch.manual_seed(42)

    model = VanillaRBM(num_visible=10, num_hidden=N_HIDDEN)
    analyzer = RBMExactAnalyzer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    hist_czz = []
    hist_s2 = []

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch, {"rng": torch.Generator()})
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if (epoch+1) % 5 == 0:
            czz, s2, valid = analyzer.analyze()
            hist_czz.append(czz)
            hist_s2.append(s2)
            print(f"Ep {epoch+1}: Loss={epoch_loss:.2f}")
            print(f"  > Czz: {czz:.4f} (Ref -0.60) | S2: {s2:.4f} (Ref 0.91) | Valid: {valid*100:.1f}%")

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(range(5, EPOCHS+1, 5), hist_czz, 'b-o', label='Czz')
    ax[0].axhline(REF_CZZ, color='g', linestyle='--')
    ax[0].set_title('Correlation')
    ax[1].plot(range(5, EPOCHS+1, 5), hist_s2, 'r-o', label='Entropy')
    ax[1].axhline(REF_ENTROPY[5], color='k', linestyle='--')
    ax[1].set_title('Entropy')
    plt.show()