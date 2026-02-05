import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append(str(Path("..").resolve()))
from data_handling import load_measurements_npz, MeasurementDataset, MeasurementLoader

data_dir = Path("measurements")

# === REF ===
REF_ENTROPY = {1: 0.693, 2: 0.650, 3: 0.860, 4: 0.780, 5: 0.910}
REF_CZZ = -0.5996

# ==========================================
# MODEL (Standard Symmetric)
# ==========================================
class SymmetricRBM(nn.Module):
    def __init__(self, num_visible: int, alpha: int):
        super().__init__()
        self.num_visible = num_visible
        self.alpha = alpha
        self.T = 1.0

        # Init
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
        # CD-1 Only
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

        # CD-1
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
    def get_entropy_curve(self):
        device = next(self.parameters()).device
        N = self.num_visible
        indices = torch.arange(2**N, device=device).unsqueeze(1)
        powers = 2**torch.arange(N - 1, -1, -1, device=device)
        v_all = (indices.bitwise_and(powers) != 0).float()
        fe = self._free_energy(v_all)
        log_probs = -fe - (-fe).max()
        probs = torch.exp(log_probs)
        psi = torch.sqrt(probs)
        psi = psi / torch.norm(psi)

        s2_curve = []
        for l in range(1, N//2 + 1):
            dim_A = 2**l
            dim_B = 2**(N-l)
            mat = psi.view(dim_A, dim_B)
            try: S = torch.linalg.svdvals(mat)
            except: S = torch.linalg.svdvals(mat.cpu())
            s2 = -math.log(torch.sum(S**4).item())
            s2_curve.append(s2)
        return s2_curve

def run_experiment(config, loader):
    print(f"\n--- Running: {config['name']} ---")
    print(f"Params: WD={config['wd']}")

    torch.manual_seed(42)
    rng = torch.Generator().manual_seed(42)

    # FIXED ARCHITECTURE
    model = SymmetricRBM(10, alpha=12)

    # VARIABLE WEIGHT DECAY
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=config['wd'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)

    czz_history = []

    for epoch in range(200):
        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch, {"rng": rng})
            loss.backward()
            optimizer.step()

        scheduler.step()

        if (epoch+1) % 20 == 0:
            model.eval()
            gen = model.generate(2000, 100, rng)
            s = 2.0*gen.float()-1.0
            czz = (s[:,:-1]*s[:,1:]).mean().item()

            # Monitoring Weight Size
            w_norm = model.kernel.norm().item()

            model.train()
            print(f"  Ep {epoch+1}: Czz={czz:.4f} | W_Norm={w_norm:.2f}")
            czz_history.append(czz)

    curve = model.get_entropy_curve()
    ent_errors = [abs(curve[l-1] - REF_ENTROPY[l]) for l in REF_ENTROPY]
    ent_mae = sum(ent_errors) / len(ent_errors)

    return {
        "config": config,
        "curve": curve,
        "czz": czz_history[-1],
        "ent_mae": ent_mae,
        "history": czz_history
    }

if __name__ == "__main__":
    file_name = f"xxz_10_delta1.00_5000000.npz"
    file_path = data_dir / file_name
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[20_000])
    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=256, shuffle=True, drop_last=True, rng=rng_loader)

    # === THE REGULARIZATION SWEEP ===
    configs = [
        {"name": "A_NoDecay",    "wd": 0.0},    # Expect: High Czz, Low Entropy
        {"name": "B_WeakDecay",  "wd": 1e-5},   # Expect: Transition
        {"name": "C_MidDecay",   "wd": 1e-4},   # Expect: Sweet Spot?
        {"name": "D_StrongDecay","wd": 5e-4},   # Expect: Low Czz, High Entropy
    ]

    results = []
    for c in configs:
        results.append(run_experiment(c, loader))

    print("\n\n=== WEIGHT DECAY SWEEP RESULTS ===")
    print(f"{'Name':<15} | {'Czz (Ref -0.60)':<15} | {'Ent MAE':<15}")
    print("-" * 50)
    for r in results:
        print(f"{r['config']['name']:<15} | {r['czz']:.4f}          | {r['ent_mae']:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    l_axis = list(REF_ENTROPY.keys())
    ax[0].plot(l_axis, list(REF_ENTROPY.values()), 'k--', linewidth=2, label='Reference')
    for r in results:
        ax[0].plot(l_axis, r['curve'], 'o-', label=f"{r['config']['name']} (WD={r['config']['wd']})")
    ax[0].set_title("Entropy S2 (Sensitivity to WD)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    for r in results:
        ax[1].plot([20,40,60,80,100,120,140,160,180,200], r['history'], 'o-', label=r['config']['name'])
    ax[1].axhline(REF_CZZ, color='k', linestyle='--')
    ax[1].set_title("Czz vs Epochs")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()