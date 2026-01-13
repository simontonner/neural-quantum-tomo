import sys
import math
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = Path("models")
H_TARGET = 0.50          # Hardest point
SAMPLES_PER_RUN = 3000
T_END = 1.0

# --- THE 3D GRID ---
START_TEMPS = [4.0, 5.0, 6.0]
STEP_COUNTS = [20, 30, 40, 50, 60, 80, 100]
FALLOFFS =    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

print(f"Running Triple Heatmap Sweep on: {DEVICE}")

# ==============================================================================
# 1. EXACT MODEL CLASS
# ==============================================================================
class Conditioner(nn.Module):
    def __init__(self, num_visible: int, num_hidden: int, cond_dim: int, hidden_width: int):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, 2 * (num_visible + num_hidden))
        self.num_visible = num_visible
        self.num_hidden = num_hidden

    def forward(self, cond: torch.Tensor):
        x = torch.tanh(self.fc1(cond))
        x = self.fc2(x)
        return torch.split(x, [self.num_visible, self.num_visible, self.num_hidden, self.num_hidden], dim=-1)

class ConditionalRBM(nn.Module):
    def __init__(self, num_visible: int, num_hidden: int, cond_dim: int,
                 conditioner_width: int = 64, k: int = 1, T: float = 1.0):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.T = T
        self.W = nn.Parameter(torch.empty(num_visible, num_hidden))
        self.b = nn.Parameter(torch.zeros(num_visible))
        self.c = nn.Parameter(torch.zeros(num_hidden))
        self.conditioner = Conditioner(num_visible, num_hidden, cond_dim, conditioner_width)
        nn.init.normal_(self.W, std=0.01)

    def _compute_effective_biases(self, cond: torch.Tensor):
        gamma_b, beta_b, gamma_c, beta_c = self.conditioner(cond)
        if cond.dim() == 1:
            b_mod = (1.0 + gamma_b) * self.b + beta_b
            c_mod = (1.0 + gamma_c) * self.c + beta_c
        else:
            b_mod = (1.0 + gamma_b) * self.b.unsqueeze(0) + beta_b
            c_mod = (1.0 + gamma_c) * self.c.unsqueeze(0) + beta_c
        return b_mod, c_mod

    @staticmethod
    def _apply_flip(v: torch.Tensor, s0: torch.Tensor) -> torch.Tensor:
        return s0 * v + (1.0 - s0) * (1.0 - v)

    def _gibbs_step_sym_fast(self, v, h, s0, b_mod, c_mod, rng):
        v_eff = self._apply_flip(v, s0)
        p_h = torch.sigmoid((v_eff @ self.W + c_mod) / self.T)
        h = torch.bernoulli(p_h, generator=rng)
        a = h @ self.W.t()
        vb   = (v * b_mod).sum(dim=-1)
        va   = (v * a).sum(dim=-1)
        bsum = b_mod.sum(dim=-1)
        asum = a.sum(dim=-1)
        dE = (-bsum - asum + 2.0 * vb + 2.0 * va)
        p_s0 = torch.sigmoid(dE / self.T)
        s0 = torch.bernoulli(p_s0, generator=rng).to(v.dtype).unsqueeze(-1)
        p_v = torch.sigmoid((a + b_mod) / self.T)
        v_eff = torch.bernoulli(p_v, generator=rng)
        v_next = self._apply_flip(v_eff, s0)
        return v_next, h, s0

    @torch.no_grad()
    def generate_sigmoid_dynamic(self, cond, steps, t_start, falloff, rng):
        if cond.dim() == 1: cond = cond.view(-1, 1)
        cond = cond.to(self.W.device, dtype=self.W.dtype)
        b_mod, c_mod = self._compute_effective_biases(cond)

        B = cond.shape[0]
        v = torch.bernoulli(torch.full((B, self.num_visible), 0.5, device=self.W.device, dtype=self.W.dtype), generator=rng)
        h = torch.zeros((B, self.num_hidden), device=self.W.device, dtype=self.W.dtype)
        s0 = torch.ones((B, 1), device=self.W.device, dtype=self.W.dtype)

        t_indices = torch.arange(steps, device=self.W.device, dtype=self.W.dtype)
        center = steps / 2.0
        s = 1.0 / (1.0 + torch.exp(falloff * (t_indices - center)))
        temps = T_END + (t_start - T_END) * s

        T_orig = self.T
        for i in range(steps):
            self.T = temps[i]
            v, h, s0 = self._gibbs_step_sym_fast(v, h, s0, b_mod, c_mod, rng)

        self.T = T_orig
        return v, temps  # <--- FIXED: Now returns tuple

# ==============================================================================
# 2. RUNNER
# ==============================================================================

def load_exact_model():
    search_path = MODELS_DIR / "crbm_tfim_16_*.pt"
    files = glob.glob(str(search_path))
    latest = max(files, key=os.path.getctime)
    chkpt = torch.load(latest, map_location=DEVICE)
    cfg = chkpt["config"]
    model = ConditionalRBM(
        num_visible=cfg.get("num_visible", 16),
        num_hidden=cfg.get("num_hidden", 64),
        cond_dim=1,
        conditioner_width=64,
        k=cfg.get("k_steps", 20),
        T=1.0
    ).to(DEVICE)
    model.load_state_dict(chkpt["model_state_dict"])
    model.eval()
    return model

def calculate_bias(samples):
    mag = (2.0 * samples - 1.0).mean(dim=1)
    return mag.mean().abs().item()

if __name__ == "__main__":
    model = load_exact_model()
    rng = torch.Generator(device=DEVICE).manual_seed(42)
    model_dtype = next(model.parameters()).dtype
    cond = torch.tensor([[H_TARGET]], device=DEVICE, dtype=model_dtype).expand(SAMPLES_PER_RUN, -1)

    results = []
    total_configs = len(START_TEMPS) * len(STEP_COUNTS) * len(FALLOFFS)
    count = 0

    print(f"Sweeping {total_configs} configurations...")

    for t_start in START_TEMPS:
        for steps in STEP_COUNTS:
            for falloff in FALLOFFS:

                samples, _ = model.generate_sigmoid_dynamic(cond, steps, t_start, falloff, rng)
                bias = calculate_bias(samples)

                viz_bias = min(bias, 0.2)

                results.append({
                    "T_start": t_start,
                    "Steps": steps,
                    "Falloff": falloff,
                    "Bias": viz_bias
                })

                count += 1
                if count % 20 == 0:
                    print(f"  Processed {count}/{total_configs}...")

    # ==========================================================================
    # 3. PLOTTING
    # ==========================================================================
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    vmin, vmax = 0.0, 0.2

    for i, t in enumerate(START_TEMPS):
        ax = axes[i]
        subset = df[df["T_start"] == t]
        matrix = subset.pivot(index="Steps", columns="Falloff", values="Bias")
        sns.heatmap(matrix, ax=ax, cmap="viridis_r", vmin=vmin, vmax=vmax,
                    annot=True, fmt=".2f", cbar=(i == 2))
        ax.set_title(f"Start Temp = {t:.1f}")
        if i == 0: ax.set_ylabel("Annealing Steps")
        else: ax.set_ylabel("")

    plt.suptitle(f"Annealing Symmetry Bias (Lower/Lighter is Better)", fontsize=16)
    plt.tight_layout()
    plt.savefig("triple_annealing_heatmap.png")
    print("\nSaved to triple_annealing_heatmap.png")