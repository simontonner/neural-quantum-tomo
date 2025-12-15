import sys
import math
import itertools
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# -----------------------------------------------------------------------------
# 1. SETUP
# -----------------------------------------------------------------------------
sys.path.append(str(Path("..").resolve()))
try:
    from data_handling import load_measurements_npz, load_state_npz, MeasurementDataset, MeasurementLoader
except ImportError:
    print("Error: Could not import 'data_handling'.")
    sys.exit(1)

data_dir = Path("measurements")
state_dir = Path("state_vectors")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on: {device}")

# -----------------------------------------------------------------------------
# 2. MODEL
# -----------------------------------------------------------------------------
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
                 conditioner_width: int = 64, k: int = 1, T: float = 1.0,
                 init_std: float = 0.01):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.T = T

        self.W = nn.Parameter(torch.empty(num_visible, num_hidden))
        self.b = nn.Parameter(torch.zeros(num_visible))
        self.c = nn.Parameter(torch.zeros(num_hidden))
        self.conditioner = Conditioner(num_visible, num_hidden, cond_dim, conditioner_width)

        # Initialize
        nn.init.normal_(self.W, std=init_std)
        nn.init.constant_(self.b, 0.0)
        nn.init.constant_(self.c, 0.0)

    def _free_energy(self, v: torch.Tensor, b_mod: torch.Tensor, c_mod: torch.Tensor) -> torch.Tensor:
        v = v.to(dtype=self.W.dtype, device=self.W.device)
        v_W = v @ self.W
        W_sum = self.W.sum(dim=0)

        linear_v = v_W + c_mod
        term2_v = F.softplus(linear_v).sum(dim=-1)
        term1_v = -(v * b_mod).sum(dim=-1)
        fe_v = term1_v - term2_v

        linear_flip = W_sum.unsqueeze(0) - v_W + c_mod
        term2_f = F.softplus(linear_flip).sum(dim=-1)
        term1_f = -((1.0 - v) * b_mod).sum(dim=-1)
        fe_flipped = term1_f - term2_f

        stacked = torch.stack([-fe_v, -fe_flipped], dim=-1)
        return -self.T * torch.logsumexp(stacked / self.T, dim=-1)

    def _compute_effective_biases(self, cond: torch.Tensor):
        gamma_b, beta_b, gamma_c, beta_c = self.conditioner(cond)
        if cond.dim() == 1:
            b_mod = (1.0 + gamma_b) * self.b + beta_b
            c_mod = (1.0 + gamma_c) * self.c + beta_c
        else:
            b_mod = (1.0 + gamma_b) * self.b.unsqueeze(0) + beta_b
            c_mod = (1.0 + gamma_c) * self.c.unsqueeze(0) + beta_c
        return b_mod, c_mod

    def _gibbs_step(self, v: torch.Tensor, b_mod: torch.Tensor, c_mod: torch.Tensor, rng: torch.Generator):
        p_h = torch.sigmoid((v @ self.W + c_mod) / self.T)
        h = torch.bernoulli(p_h, generator=rng)
        p_v = torch.sigmoid((h @ self.W.t() + b_mod) / self.T)
        return torch.bernoulli(p_v, generator=rng)

    def forward(self, batch: Tuple[torch.Tensor, ...], aux_vars: Dict[str, Any]):
        v_data, _, cond = batch
        v_data = v_data.to(device=self.W.device, dtype=self.W.dtype)
        cond = cond.to(device=self.W.device, dtype=self.W.dtype)
        rng = aux_vars.get("rng")

        b_mod, c_mod = self._compute_effective_biases(cond)
        v_model = v_data.clone()

        noise_frac = aux_vars.get("noise_frac", 0.0)
        if noise_frac > 0:
            n_noise = int(v_data.shape[0] * noise_frac)
            if n_noise > 0:
                mask = torch.rand(v_model[:n_noise].shape, device=v_model.device) < 0.5
                v_model[:n_noise] = torch.where(mask, 1.0 - v_model[:n_noise], v_model[:n_noise])

        for _ in range(self.k):
            v_model = self._gibbs_step(v_model, b_mod, c_mod, rng)

        v_model = v_model.detach()
        fe_data = self._free_energy(v_data, b_mod, c_mod)
        fe_model = self._free_energy(v_model, b_mod, c_mod)
        return (fe_data - fe_model).mean(), {}

    @torch.no_grad()
    def get_log_prob_unnorm(self, cond: torch.Tensor, all_states: torch.Tensor) -> torch.Tensor:
        if cond.dim() == 1: cond = cond.unsqueeze(0)
        cond_exp = cond.expand(all_states.shape[0], -1)
        b_mod, c_mod = self._compute_effective_biases(cond_exp)
        return -self._free_energy(all_states, b_mod, c_mod) / self.T

# -----------------------------------------------------------------------------
# 3. UTILITIES
# -----------------------------------------------------------------------------
def get_sigmoid_curve(high, low, steps, falloff):
    center = steps / 2.0
    def fn(step):
        s = min(step, steps)
        val = falloff * (s - center)
        if val > 50: return low
        if val < -50: return high
        return float(low + (high - low) / (1.0 + math.exp(val)))
    return fn

def generate_all_states(num_qubits: int):
    lst = list(itertools.product([0, 1], repeat=num_qubits))
    return torch.tensor(lst, dtype=torch.float32, device=device)

@torch.no_grad()
def evaluate(model, h_list, all_states):
    model.eval()
    overlaps = []
    for h_val in h_list:
        gt_path = state_dir / f"tfim_16_h{h_val:.2f}.npz"
        if not gt_path.exists(): continue

        psi_np, _ = load_state_npz(gt_path)
        psi_true = torch.from_numpy(psi_np).real.float().to(device)
        psi_true /= torch.norm(psi_true)

        cond = torch.tensor([h_val], device=device, dtype=torch.float32)
        log_prob = model.get_log_prob_unnorm(cond, all_states)
        log_prob -= torch.logsumexp(log_prob, dim=0)
        psi_model = torch.exp(0.5 * log_prob)

        overlaps.append(torch.abs(torch.dot(psi_true, psi_model)).item())

    return min(overlaps), sum(overlaps)/len(overlaps)

def train_trial(config, train_ds, h_eval, all_states):
    bs = config['batch_size']
    lr = config['lr_init']
    n_h = config['n_hidden']
    k = config['cd_k']
    noise = config['noise_frac']
    std = config['init_std']

    # Loader
    rng_cpu = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False, rng=rng_cpu)

    # Model
    model = ConditionalRBM(16, n_h, 1, k=k, init_std=std).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Shorter schedule if batch size is small (more updates)
    epochs = 50
    total_steps = epochs * len(loader)
    schedule = get_sigmoid_curve(lr, 1e-4, total_steps, falloff=0.005)

    rng_gpu = torch.Generator(device=device).manual_seed(42)
    step = 0

    model.train()
    for _ in range(epochs):
        for batch in loader:
            cur_lr = schedule(step)
            for g in optimizer.param_groups: g["lr"] = cur_lr

            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(batch, {"rng": rng_gpu, "noise_frac": noise})
            loss.backward()
            optimizer.step()
            step += 1

    min_ov, avg_ov = evaluate(model, h_eval, all_states)
    return min_ov, avg_ov, model.state_dict()

# -----------------------------------------------------------------------------
# 4. EXPERIMENT: 10k Samples Efficiency
# -----------------------------------------------------------------------------
def run_10k_experiments():
    SAMPLES = 10_000
    h_supp = [0.50, 0.80, 0.95, 1.00, 1.05, 1.20, 1.50]
    h_eval = [0.60, 0.70, 1.00, 1.30, 1.40] # Critical & Novel

    paths = [data_dir / f"tfim_16_h{h:.2f}_5000000.npz" for h in h_supp]
    paths = [p for p in paths if p.exists()]

    ds = MeasurementDataset(paths, load_measurements_npz, ["h"], [SAMPLES]*len(paths))
    print(f"Dataset: {len(ds)} samples (10k subset).")

    all_states = generate_all_states(16)

    # --- SMART GRID ---
    grid = []

    # 1. The "Winner" Baseline (From previous run)
    # 1024 batch, high LR, k=15
    grid.append({'batch_size': 1024, 'n_hidden': 80, 'lr_init': 0.033, 'cd_k': 15, 'noise_frac': 0.02, 'init_std': 0.01})

    # 2. Speed Hypothesis: Can we do k=5 with smaller batches?
    # 256 batch (4x updates), same LR, k=5
    grid.append({'batch_size': 256, 'n_hidden': 80, 'lr_init': 0.025, 'cd_k': 5, 'noise_frac': 0.02, 'init_std': 0.01})

    # 3. Capacity Hypothesis: maybe 80 is too small?
    grid.append({'batch_size': 512, 'n_hidden': 100, 'lr_init': 0.03, 'cd_k': 10, 'noise_frac': 0.02, 'init_std': 0.01})

    # 4. Noise Hypothesis: More noise?
    grid.append({'batch_size': 512, 'n_hidden': 80, 'lr_init': 0.03, 'cd_k': 10, 'noise_frac': 0.06, 'init_std': 0.01})

    # 5. Initialization Hypothesis: Stronger weights
    grid.append({'batch_size': 512, 'n_hidden': 80, 'lr_init': 0.03, 'cd_k': 10, 'noise_frac': 0.02, 'init_std': 0.05})

    # Random fills around the "High Energy" zone
    for _ in range(7):
        grid.append({
            'batch_size': int(np.random.choice([256, 512, 1024])),
            'n_hidden': int(np.random.choice([80, 96, 112])),
            'lr_init': 10**np.random.uniform(-1.7, -1.4), # ~0.02 to 0.04
            'cd_k': int(np.random.choice([5, 8, 10])),
            'noise_frac': np.random.choice([0.02, 0.04, 0.06]),
            'init_std': np.random.choice([0.01, 0.02])
        })

    print(f"\nStarting 10k Sample Experiments ({len(grid)} trials)...")
    print(f"{'#':<2}|{'Btch':<4}|{'Hid':<3}|{'LR':<6}|{'k':<2}|{'Std':<5}|{'Time':<5}|{'MinOv':<7}")
    print("-" * 55)

    results = []
    best_ov = 0.0

    for i, conf in enumerate(grid):
        start = time.time()
        min_ov, avg_ov, state = train_trial(conf, ds, h_eval, all_states)
        dur = time.time() - start

        results.append({**conf, 'min_ov': min_ov, 'avg_ov': avg_ov, 'time': dur})
        print(f"{i+1:<2}|{conf['batch_size']:<4}|{conf['n_hidden']:<3}|{conf['lr_init']:.3f} |{conf['cd_k']:<2}|{conf['init_std']:<5}|{dur:<5.0f}|{min_ov:.5f}")

        if min_ov > best_ov:
            best_ov = min_ov
            torch.save(state, "best_crbm_10k.pt")

    df = pd.DataFrame(results)
    df = df.sort_values(by="min_ov", ascending=False)

    print("\nTop 5 Configs (10k Samples):")
    print(df[['batch_size','n_hidden','lr_init','cd_k','init_std','time','min_ov']].head(5))

if __name__ == "__main__":
    run_10k_experiments()