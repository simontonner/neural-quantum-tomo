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
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = Path("models")
SEED = 42

# --- YOUR CHOSEN "DEEP YELLOW" PARAMETERS ---
ANNEAL_T_START = 5.0
ANNEAL_STEPS   = 100
ANNEAL_FALLOFF = 0.4
T_END          = 1.0

# Sampling settings
TOTAL_SAMPLES = 50_000
CHUNK_SIZE    = 10_000

print(f"Running Final Paper Evaluation on: {DEVICE}")
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==============================================================================
# 1. EXACT MODEL DEFINITION
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
        # 1. Sample h
        v_eff = self._apply_flip(v, s0)
        p_h = torch.sigmoid((v_eff @ self.W + c_mod) / self.T)
        h = torch.bernoulli(p_h, generator=rng)
        # 2. Sample s
        a = h @ self.W.t()
        vb   = (v * b_mod).sum(dim=-1)
        va   = (v * a).sum(dim=-1)
        bsum = b_mod.sum(dim=-1)
        asum = a.sum(dim=-1)
        dE = (-bsum - asum + 2.0 * vb + 2.0 * va)
        p_s0 = torch.sigmoid(dE / self.T)
        s0 = torch.bernoulli(p_s0, generator=rng).to(v.dtype).unsqueeze(-1)
        # 3. Sample v
        p_v = torch.sigmoid((a + b_mod) / self.T)
        v_eff = torch.bernoulli(p_v, generator=rng)
        v_next = self._apply_flip(v_eff, s0)
        return v_next, h, s0

    # --- Free Energy ---
    def _free_energies_pair(self, v: torch.Tensor, b_mod: torch.Tensor, c_mod: torch.Tensor):
        v = v.to(dtype=self.W.dtype, device=self.W.device)
        v_W = v @ self.W
        W_sum = self.W.sum(dim=0)
        linear_v = v_W + c_mod
        linear_f = W_sum.unsqueeze(0) - v_W + c_mod
        term2_v = F.softplus(linear_v).sum(dim=-1)
        term2_f = F.softplus(linear_f).sum(dim=-1)
        term1_v = -(v * b_mod).sum(dim=-1)
        term1_f = -((1.0 - v) * b_mod).sum(dim=-1)
        return (term1_v - term2_v), (term1_f - term2_f)

    def _free_energy(self, v: torch.Tensor, b_mod: torch.Tensor, c_mod: torch.Tensor) -> torch.Tensor:
        F_v, F_f = self._free_energies_pair(v, b_mod, c_mod)
        stacked = torch.stack([-F_v, -F_f], dim=-1)
        return -self.T * torch.logsumexp(stacked / self.T, dim=-1)

    def log_score(self, v: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b_mod, c_mod = self._compute_effective_biases(cond)
        return -0.5 * self._free_energy(v, b_mod, c_mod) / self.T

    # --- FINAL SIGMOID GENERATION ---
    @torch.no_grad()
    def generate_sigmoid(self, cond: torch.Tensor, steps: int, falloff: float, rng: torch.Generator) -> torch.Tensor:
        if cond.dim() == 1: cond = cond.view(-1, 1)
        cond = cond.to(self.W.device, dtype=self.W.dtype)
        b_mod, c_mod = self._compute_effective_biases(cond)

        B = cond.shape[0]
        v = torch.bernoulli(torch.full((B, self.num_visible), 0.5, device=self.W.device, dtype=self.W.dtype), generator=rng)
        h = torch.zeros((B, self.num_hidden), device=self.W.device, dtype=self.W.dtype)
        s0 = torch.ones((B, 1), device=self.W.device, dtype=self.W.dtype)

        # Sigmoid Schedule
        t_indices = torch.arange(steps, device=self.W.device, dtype=self.W.dtype)
        center = steps / 2.0
        # s goes 1.0 -> 0.0 roughly
        s = 1.0 / (1.0 + torch.exp(falloff * (t_indices - center)))
        temps = T_END + (ANNEAL_T_START - T_END) * s

        T_original = self.T
        for i in range(steps):
            self.T = temps[i]
            v, h, s0 = self._gibbs_step_sym_fast(v, h, s0, b_mod, c_mod, rng)

        self.T = T_original
        return v

# ==============================================================================
# 2. EVALUATION LOGIC
# ==============================================================================
def compute_renyi_large_batch(samples: torch.Tensor, subs_size: int,
                              log_score_fn: callable, chunk_size: int = 10_000) -> tuple:
    n_total = samples.shape[0]
    n_chunks = max(1, n_total // chunk_size)
    scores = log_score_fn(samples)

    chunk_s2_values = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_total)
        if end - start < 2: continue

        c_samples = samples[start:end]
        c_scores = scores[start:end]
        half = (end - start) // 2

        ref_1 = c_samples[:half]; ref_2 = c_samples[half:2*half]
        ref_1_score = c_scores[:half]; ref_2_score = c_scores[half:2*half]

        slice_idx = torch.arange(subs_size, samples.shape[1], device=samples.device)
        swap_1 = ref_1.clone(); swap_1[:, slice_idx] = ref_2[:, slice_idx]
        swap_2 = ref_2.clone(); swap_2[:, slice_idx] = ref_1[:, slice_idx]

        log_ratios = (log_score_fn(swap_1) + log_score_fn(swap_2)) - (ref_1_score + ref_2_score)
        max_val = torch.max(log_ratios)
        log_mean = (torch.log(torch.sum(torch.exp(log_ratios - max_val))) + max_val) - math.log(half)
        chunk_s2_values.append(-log_mean.item())

    vals = np.array(chunk_s2_values)
    return np.mean(vals), (np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)

def load_model():
    search_path = MODELS_DIR / "crbm_tfim_16_*.pt"
    files = glob.glob(str(search_path))
    if not files: raise FileNotFoundError("No models found")
    latest = max(files, key=os.path.getctime)
    print(f"Loading: {latest}")
    chkpt = torch.load(latest, map_location=DEVICE)
    cfg = chkpt["config"]
    model = ConditionalRBM(
        cfg.get("num_visible", 16), cfg.get("num_hidden", 64), 1, 64, cfg.get("k_steps", 20)
    ).to(DEVICE)
    model.load_state_dict(chkpt["model_state_dict"])
    model.eval()
    return model, cfg

# ==============================================================================
# 3. MAIN RUN
# ==============================================================================
model, config = load_model()
h_support = sorted(config.get("h_support", [0.5, 0.8, 0.95, 1.0, 1.05, 1.2, 1.5]))
chain_len = model.num_visible
model_dtype = next(model.parameters()).dtype

# H Grid
dense_res = 0.05
h_novel = np.arange(min(h_support), max(h_support) + 0.01, dense_res)
h_novel = [h for h in h_novel if not any(np.isclose(h, s, atol=1e-3) for s in h_support)]
all_h = sorted(list(set(h_support) | set(h_novel)))

results = []
rng = torch.Generator(device=DEVICE).manual_seed(SEED)
l_axis = list(range(1, chain_len // 2 + 1))

print(f"\n=== Running Final Paper Evaluation ===")
print(f"Schedule: T={ANNEAL_T_START}->{T_END} | Steps={ANNEAL_STEPS} | k={ANNEAL_FALLOFF}")

for i, h_val in enumerate(all_h):
    pt_type = "support" if any(np.isclose(h_val, s, atol=1e-3) for s in h_support) else "interpolated"

    # 1. Generate with Sigmoid
    cond = torch.tensor([[h_val]], device=DEVICE, dtype=model_dtype).expand(TOTAL_SAMPLES, -1)
    with torch.no_grad():
        samples = model.generate_sigmoid(cond, ANNEAL_STEPS, ANNEAL_FALLOFF, rng)

    # 2. Score
    cond_s = torch.tensor([[h_val]], device=DEVICE, dtype=model_dtype)
    scorer = lambda v: model.log_score(v, cond_s)

    # 3. Compute Entropy
    for l in l_axis:
        s2, err = compute_renyi_large_batch(samples, l, scorer, CHUNK_SIZE)
        results.append({"h": float(h_val), "l": int(l), "s2": s2, "s2_err": err, "type": pt_type})

    if (i+1) % 5 == 0 or (i+1) == len(all_h):
        print(f"[{i+1}/{len(all_h)}] h={h_val:.2f} done.")

df_res = pd.DataFrame(results)

# ==============================================================================
# 4. PLOTTING
# ==============================================================================
fig = plt.figure(figsize=(16, 6), dpi=150)
gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 0.8], wspace=-0.15)
pivot = df_res.pivot(index='l', columns='h', values='s2')
X, Y = np.meshgrid(pivot.columns, pivot.index)
Z = pivot.values

# 3D Surface
ax3d = fig.add_subplot(gs[0], projection='3d')
ax3d.set_proj_type('ortho')
surf = ax3d.plot_surface(X, Y, Z, cmap='viridis', rstride=1, cstride=1, shade=True, alpha=0.9)
ax3d.set_xlabel('h'); ax3d.set_ylabel('Subsystem Size'); ax3d.set_zlabel('$S_2$')
ax3d.view_init(elev=30, azim=-40)

# 2D Cuts
ax2d = fig.add_subplot(gs[1])
cmap = plt.get_cmap("tab10")
for i, h in enumerate(h_support):
    sub = df_res[df_res['h'] == h].sort_values('l')
    ax2d.plot(sub['l'], sub['s2'], 'o-', color=cmap(i%10), label=f"h={h:.2f}")
    ax2d.fill_between(sub['l'], sub['s2']-sub['s2_err'], sub['s2']+sub['s2_err'], color=cmap(i%10), alpha=0.2)

ax2d.set_title("Renyi Entropy (Deep Yellow Schedule)")
ax2d.legend(ncol=2, fontsize=8)
ax2d.grid(alpha=0.3)
plt.show()