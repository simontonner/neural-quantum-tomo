import sys
import math
import itertools
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# -----------------------------------------------------------------------------
# 1. SETUP & PATHS
# -----------------------------------------------------------------------------

# Add parent directory to path to find data_handling.py
sys.path.append(str(Path("..").resolve()))
try:
    from data_handling import load_measurements_npz, load_state_npz, MeasurementDataset, MeasurementLoader
except ImportError:
    print("Error: Could not import 'data_handling'. Make sure you are in the correct directory.")
    sys.exit(1)

# Define paths
data_dir = Path("measurements")
state_dir = Path("state_vectors")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on: {device}")

# -----------------------------------------------------------------------------
# 2. MODEL DEFINITIONS (Conditioner & CRBM)
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
        self.initialize_weights()

    def initialize_weights(self):
        # Smaller initialization often helps convergence stability
        nn.init.normal_(self.W, std=0.01)
        nn.init.constant_(self.b, 0.0)
        nn.init.constant_(self.c, 0.0)

    def _free_energy(self, v: torch.Tensor, b_mod: torch.Tensor, c_mod: torch.Tensor) -> torch.Tensor:
        """Symmetrized Free Energy (Z2 symmetry enforcement)"""
        v = v.to(dtype=self.W.dtype, device=self.W.device)
        v_W = v @ self.W
        W_sum = self.W.sum(dim=0)

        # State v
        linear_v = v_W + c_mod
        term2_v = F.softplus(linear_v).sum(dim=-1)
        term1_v = -(v * b_mod).sum(dim=-1)
        fe_v = term1_v - term2_v

        # State 1-v
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
        # Standard Gibbs sampling
        p_h = torch.sigmoid((v @ self.W + c_mod) / self.T)
        h = torch.bernoulli(p_h, generator=rng)
        p_v = torch.sigmoid((h @ self.W.t() + b_mod) / self.T)
        return torch.bernoulli(p_v, generator=rng)

    def log_score(self, v: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b_mod, c_mod = self._compute_effective_biases(cond)
        return -self._free_energy(v, b_mod, c_mod) / self.T

    def forward(self, batch: Tuple[torch.Tensor, ...], aux_vars: Dict[str, Any]):
        v_data, _, cond = batch
        v_data = v_data.to(device=self.W.device, dtype=self.W.dtype)
        cond = cond.to(device=self.W.device, dtype=self.W.dtype)
        rng = aux_vars.get("rng")

        b_mod, c_mod = self._compute_effective_biases(cond)

        # Contrastive Divergence
        # 1. Initialize chain
        v_model = v_data.clone()

        # 2. Add noise if requested (helps mixing)
        noise_frac = aux_vars.get("noise_frac", 0.0)
        if noise_frac > 0:
            n_noise = int(v_data.shape[0] * noise_frac)
            if n_noise > 0:
                noise_mask = torch.rand(v_model[:n_noise].shape, device=v_model.device) < 0.5
                v_model[:n_noise] = torch.where(noise_mask, 1.0 - v_model[:n_noise], v_model[:n_noise])

        # 3. Gibbs Steps
        for _ in range(self.k):
            v_model = self._gibbs_step(v_model, b_mod, c_mod, rng)
        v_model = v_model.detach()

        # 4. Loss
        fe_data = self._free_energy(v_data, b_mod, c_mod)
        fe_model = self._free_energy(v_model, b_mod, c_mod)
        loss = (fe_data - fe_model).mean()

        return loss, {}

    @torch.no_grad()
    def get_normalized_wavefunction(self, cond: torch.Tensor, all_states: torch.Tensor) -> torch.Tensor:
        if cond.dim() == 1: cond = cond.unsqueeze(0)
        cond_exp = cond.expand(all_states.shape[0], -1)
        log_prob_unnorm = self.log_score(all_states, cond_exp)
        log_Z = torch.logsumexp(log_prob_unnorm, dim=0)
        log_prob = log_prob_unnorm - log_Z
        return torch.exp(0.5 * log_prob)

# -----------------------------------------------------------------------------
# 3. UTILITIES & TRAINING
# -----------------------------------------------------------------------------

def get_sigmoid_curve(high, low, steps, falloff):
    center = steps / 2.0
    def fn(step):
        s = min(step, steps)
        # Check for overflow
        val = falloff * (s - center)
        if val > 50: return low
        if val < -50: return high
        return float(low + (high - low) / (1.0 + math.exp(val)))
    return fn

def generate_all_states(num_qubits: int, device: torch.device):
    lst = list(itertools.product([0, 1], repeat=num_qubits))
    return torch.tensor(lst, dtype=torch.float32, device=device)

@torch.no_grad()
def evaluate_overlaps(model, h_values, state_dir, all_states):
    model.eval()
    overlaps = {}
    for h_val in h_values:
        gt_fname = f"tfim_16_h{h_val:.2f}.npz"
        gt_path = state_dir / gt_fname
        if not gt_path.exists(): continue

        psi_np, _ = load_state_npz(gt_path)
        psi_true = torch.from_numpy(psi_np).real.float().to(device)
        psi_true = psi_true / torch.norm(psi_true)

        cond = torch.tensor([h_val], device=device, dtype=torch.float32)
        psi_model = model.get_normalized_wavefunction(cond, all_states)

        ov = torch.abs(torch.dot(psi_true, psi_model)).item()
        overlaps[h_val] = ov
    return overlaps

def train_one_trial(config, train_ds, h_eval_list, all_states, state_dir, epochs=50):
    # Unpack config
    batch_size = config['batch_size']
    lr_init = config['lr_init']
    lr_final = config['lr_final']
    n_hidden = config['n_hidden']
    cond_width = config['cond_width']
    cd_k = config['cd_k']
    noise_frac = config['noise_frac']

    # Init Loader
    # Create a fresh loader with specific batch size
    rng_cpu = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, rng=rng_cpu)

    # Init Model
    model = ConditionalRBM(
        num_visible=16,
        num_hidden=n_hidden,
        cond_dim=1,
        conditioner_width=cond_width,
        k=cd_k
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)

    # Scheduler
    total_steps = epochs * len(loader)
    schedule_fn = get_sigmoid_curve(lr_init, lr_final, total_steps, falloff=0.005) # Slightly sharper falloff

    rng_gpu = torch.Generator(device=device).manual_seed(42)
    global_step = 0

    model.train()
    for epoch in range(epochs):
        for batch in loader:
            current_lr = schedule_fn(global_step)
            for g in optimizer.param_groups: g["lr"] = current_lr

            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(batch, {"rng": rng_gpu, "noise_frac": noise_frac})
            loss.backward()
            optimizer.step()
            global_step += 1

    # Final Evaluation
    overlaps = evaluate_overlaps(model, h_eval_list, state_dir, all_states)

    # Key Metric: Minimum Overlap (Weakest link)
    min_ov = min(overlaps.values()) if overlaps else 0.0
    avg_ov = sum(overlaps.values()) / len(overlaps) if overlaps else 0.0

    return min_ov, avg_ov, overlaps, model.state_dict()

# -----------------------------------------------------------------------------
# 4. HYPERPARAMETER SEARCH LOGIC
# -----------------------------------------------------------------------------

def run_sweep():
    # --- CONFIGURATION ---
    SYSTEM_SIZE = 16
    FILE_SUFFIX = "5000000"

    # Use smaller dataset for the sweep (10k samples)
    SAMPLES_PER_FILE = 10_000

    h_support = [0.50, 0.80, 0.95, 1.00, 1.05, 1.20, 1.50]
    # We evaluate on NOVEL points to ensure generalization
    h_novel   = [0.60, 0.70, 1.30, 1.40]

    # Pre-load Data
    file_paths = []
    print("Loading Training Data (Subset)...")
    for h in h_support:
        fname = f"tfim_{SYSTEM_SIZE}_h{h:.2f}_{FILE_SUFFIX}.npz"
        fpath = data_dir / fname
        if fpath.exists(): file_paths.append(fpath)

    if not file_paths:
        print("No files found!")
        return

    ds = MeasurementDataset(
        file_paths, load_measurements_npz, ["h"],
        samples_per_file=[SAMPLES_PER_FILE]*len(file_paths)
    )
    print(f"Data Loaded. Total Samples: {len(ds)}")

    # Pre-compute States for Eval
    print("Generating Hilbert space...")
    all_states = generate_all_states(SYSTEM_SIZE, device)

    # --- SEARCH SPACE ---
    # Smart constraints based on your problem:
    # 1. High hidden dim needed for Entanglement Entropy
    # 2. Large batch sizes stabilize RBM training
    # 3. K steps don't need to be huge, but >1 helps

    NUM_TRIALS = 10
    EPOCHS_PER_TRIAL = 60 # Enough to see convergence

    trials = []

    best_min_overlap = 0.0
    best_config = None

    print(f"\nStarting Search ({NUM_TRIALS} trials)...")
    print(f"{'#':<3} | {'Batch':<5} | {'Hid':<4} | {'LR_0':<8} | {'Noise':<5} | {'k':<2} | {'Min Ov.':<8} | {'Avg Ov.':<8}")
    print("-" * 75)

    for i in range(NUM_TRIALS):
        # 1. Sample Hyperparameters
        config = {
            'batch_size': int(np.random.choice([2048, 4096])), # Large batches
            'n_hidden': int(np.random.choice([64, 80, 128])),  # Higher capacity
            'cond_width': int(np.random.choice([64, 128])),
            'lr_init': 10 ** np.random.uniform(-2.5, -1.8),    # ~0.003 to 0.015
            'lr_final': 1e-4,
            'cd_k': int(np.random.choice([5, 10, 15])),        # Moderate steps
            'noise_frac': np.random.choice([0.0, 0.02, 0.04])
        }

        # 2. Train
        start_t = time.time()
        min_ov, avg_ov, ov_dict, state_dict = train_one_trial(
            config, ds, h_novel, all_states, state_dir, epochs=EPOCHS_PER_TRIAL
        )
        dur = time.time() - start_t

        # 3. Log
        trials.append({**config, 'min_ov': min_ov, 'avg_ov': avg_ov, 'duration': dur})

        print(f"{i+1:<3} | {config['batch_size']:<5} | {config['n_hidden']:<4} | {config['lr_init']:.1e} | {config['noise_frac']:<5} | {config['cd_k']:<2} | {min_ov:.5f}   | {avg_ov:.5f}")

        # 4. Save Best
        if min_ov > best_min_overlap:
            best_min_overlap = min_ov
            best_config = config
            torch.save(state_dict, "best_crbm_model.pt")
            # print(f"   >>> New Best found! (Saved to best_crbm_model.pt)")

    # --- REPORT ---
    print("\n" + "="*40)
    print("SEARCH COMPLETE")
    print("="*40)

    df = pd.DataFrame(trials)
    df = df.sort_values(by="min_ov", ascending=False)

    print("\nTop 5 Configurations:")
    print(df[['batch_size', 'n_hidden', 'lr_init', 'cd_k', 'noise_frac', 'min_ov', 'avg_ov']].head(5))

    print("\nBest Configuration Detail:")
    print(best_config)
    print(f"Best Minimum Overlap (on Novel H): {best_min_overlap:.5f}")

    print("\nTo use the best model:")
    print("1. Load 'best_crbm_model.pt'")
    print("2. Instantiate ConditionalRBM with the parameters above.")
    print("3. (Optional) Fine-tune on the full 5M dataset with small LR.")

if __name__ == "__main__":
    run_sweep()