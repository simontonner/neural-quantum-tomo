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
REF_CZZ = -0.5996
REF_ENTROPY = {1: 0.693, 2: 0.650, 3: 0.860, 4: 0.780, 5: 0.910}

# ==========================================
# MODEL
# ==========================================
class SymmetricRBM(nn.Module):
    def __init__(self, num_visible: int, alpha: int):
        super().__init__()
        self.num_visible = num_visible
        self.alpha = alpha
        self.T = 1.0

        self.kernel = nn.Parameter(torch.empty(1, alpha, num_visible))
        self.b_scalar = nn.Parameter(torch.zeros(1))
        self.c_vector = nn.Parameter(torch.zeros(1, alpha))

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

        # DYNAMIC CD-K: Get k from aux_vars
        current_k = aux_vars.get("k", 1)

        v_model = v_data.clone()
        for _ in range(current_k):
            v_model = self._gibbs_step(v_model, rng)
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
# MAIN
# ==========================================
if __name__ == "__main__":
    # CONFIGURATION
    ALPHA = 12
    BATCH_SIZE = 256

    # === THE TWO-STAGE SCHEDULE ===
    # Phase 1: High noise to get Entropy right
    EPOCHS_PHASE_1 = 30
    K_PHASE_1 = 1
    LR_PHASE_1 = 0.005

    # Phase 2: Low noise + Low LR to recover Czz
    EPOCHS_PHASE_2 = 30
    K_PHASE_2 = 10
    LR_PHASE_2 = 0.0005 # 10x smaller LR to prevent collapse

    TOTAL_EPOCHS = EPOCHS_PHASE_1 + EPOCHS_PHASE_2

    # Load Data
    file_name = f"xxz_10_delta1.00_5000000.npz"
    file_path = data_dir / file_name
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[20_000])
    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, rng=rng_loader)

    print(f"Starting Two-Stage Annealing (Alpha={ALPHA})...")
    torch.manual_seed(42)
    rng = torch.Generator().manual_seed(42)

    model = SymmetricRBM(10, alpha=ALPHA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_PHASE_1, weight_decay=1e-5)

    hist_czz = []

    for epoch in range(TOTAL_EPOCHS):
        # === SWITCH PHASE ===
        if epoch < EPOCHS_PHASE_1:
            current_k = K_PHASE_1
            phase = "MELT"
        else:
            if epoch == EPOCHS_PHASE_1:
                print(">>> SWITCHING TO PHASE 2: CRYSTALLIZE (CD-10, Low LR) <<<")
                # Manual LR adjustment
                for param_group in optimizer.param_groups:
                    param_group['lr'] = LR_PHASE_2
            current_k = K_PHASE_2
            phase = "FREEZE"

        # Train
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            # Pass K dynamically
            loss = model(batch, {"rng": rng, "k": current_k})
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Measure
        model.eval()
        gen = model.generate(2000, 100, rng)
        s = 2.0*gen.float()-1.0
        czz = (s[:,:-1]*s[:,1:]).mean().item()
        model.train()
        hist_czz.append(czz)

        if (epoch+1) % 5 == 0:
            print(f"Ep {epoch+1} ({phase}): Loss={epoch_loss/len(loader):.4f} | Czz={czz:.4f}")

    # --- FINAL CHECK ---
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

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(range(1, TOTAL_EPOCHS+1), hist_czz, 'b-o')
    ax[0].axvline(EPOCHS_PHASE_1, color='k', linestyle='--', label='Phase Switch')
    ax[0].axhline(REF_CZZ, color='g', linestyle='--')
    ax[0].set_title('Czz Trajectory')
    ax[0].legend()

    l_axis = list(REF_ENTROPY.keys())
    ax[1].plot(l_axis, list(REF_ENTROPY.values()), 'k--', label='Ref')
    ax[1].plot(l_axis, s2_curve, 'r-o', label='Hybrid Model')
    ax[1].set_title('Final Entropy')
    ax[1].legend()

    plt.tight_layout()
    plt.show()