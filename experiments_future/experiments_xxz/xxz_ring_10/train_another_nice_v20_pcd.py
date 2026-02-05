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

sys.path.append(str(Path("..").resolve()))
from data_handling import load_measurements_npz, MeasurementDataset, MeasurementLoader

data_dir = Path("measurements")

# REF
REF_CZZ = -0.5996
REF_ENTROPY = {1: 0.693, 2: 0.650, 3: 0.860, 4: 0.780, 5: 0.910}

class SymmetricRBM(nn.Module):
    def __init__(self, num_visible: int, alpha: int):
        super().__init__()
        self.num_visible = num_visible
        self.alpha = alpha
        self.T = 1.0
        self.kernel = nn.Parameter(torch.empty(1, alpha, num_visible))
        # Freeze Bias to 0 to assist symmetry
        self.b_scalar = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.c_vector = nn.Parameter(torch.zeros(1, alpha))
        nn.init.normal_(self.kernel, mean=0.0, std=0.05)
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
        v_raw = values.to(device=self.kernel.device).float()

        # === INGREDIENT 1: AUGMENTATION ===
        # Force the model to see the Spin-Flipped partner of every state.
        # This prevents Spontaneous Symmetry Breaking (Entropy Collapse).
        v_data = torch.cat([v_raw, 1.0 - v_raw], dim=0)

        rng = aux_vars.get("rng", torch.Generator(device="cpu"))

        # === INGREDIENT 2: CD-1 ===
        # Keep the noise to maintain quantum fluctuations
        v_model = v_data.clone()
        v_model = self._gibbs_step(v_model, rng) # K=1
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

if __name__ == "__main__":
    # CONFIG
    ALPHA = 12
    BATCH_SIZE = 256
    EPOCHS = 50
    LR = 0.005
    WEIGHT_DECAY = 1e-5

    file_name = f"xxz_10_delta1.00_5000000.npz"
    file_path = data_dir / file_name
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[20_000])
    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, rng=rng_loader)

    print(f"Training FINAL COMBINATION (Augment + CD-1 + PostSelect)...")
    torch.manual_seed(42)
    rng = torch.Generator().manual_seed(42)

    model = SymmetricRBM(10, alpha=ALPHA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

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

        if (epoch+1) % 5 == 0:
            model.eval()
            gen_raw = model.generate(5000, 100, rng)

            # === INGREDIENT 3: POST-SELECTION ===
            # Filter for physics
            magnetization = gen_raw.sum(dim=1)
            target_mag = 5
            mask = (magnetization == target_mag)
            gen_physical = gen_raw[mask]

            if len(gen_physical) > 10:
                s = 2.0 * gen_physical.float() - 1.0
                czz = (s[:,:-1] * s[:,1:]).mean().item()
            else:
                czz = 0.0

            model.train()
            hist_czz.append(czz)
            valid_pct = len(gen_physical)/len(gen_raw)*100
            print(f"Ep {epoch+1}: Loss={epoch_loss/len(loader):.4f} | Czz={czz:.4f} (Ref {REF_CZZ:.4f}) | Valid={valid_pct:.1f}%")

    print("\nCalculating Final Entropy...")
    psi = model.get_full_psi()
    s2_curve = []
    print(f"{'L':<3} | {'S2 Model':<10} | {'S2 Ref':<10}")
    print("-" * 30)
    for l in range(1, 6):
        dim_A = 2**l
        dim_B = 2**(10-l)
        mat = psi.view(dim_A, dim_B)
        try: S = torch.linalg.svdvals(mat)
        except: S = torch.linalg.svdvals(mat.cpu())
        s2 = -math.log(torch.sum(S**4).item())
        s2_curve.append(s2)
        print(f"{l:<3} | {s2:.4f}     | {REF_ENTROPY[l]:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(range(5, EPOCHS+1, 5), hist_czz, 'b-o')
    ax[0].axhline(REF_CZZ, color='g', linestyle='--')
    ax[0].set_title('Physical Czz')

    l_axis = list(REF_ENTROPY.keys())
    ax[1].plot(l_axis, list(REF_ENTROPY.values()), 'k--', label='Ref')
    ax[1].plot(l_axis, s2_curve, 'r-o', label='Model')
    ax[1].set_title('Entropy')
    ax[1].legend()
    plt.tight_layout()
    plt.show()