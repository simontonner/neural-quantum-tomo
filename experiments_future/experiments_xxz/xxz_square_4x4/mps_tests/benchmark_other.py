import os
import sys
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Adjust path to find your data_handling module
sys.path.append(str(Path("..").resolve()))
from data_handling import load_measurements_npz, MeasurementDataset, MeasurementLoader

# --- DEVICE CONFIGURATION ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Success: Device set to Apple MPS (GPU)")
else:
    device = torch.device("cpu")
    print(f"Warning: Device set to CPU")

data_dir = Path("measurements")

# --- MODEL DEFINITION (SAFE MODE) ---
class Conditioner(nn.Module):
    def __init__(self, num_visible: int, num_hidden: int, cond_dim: int, hidden_width: int):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.fc1 = nn.Linear(cond_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, 2 * (num_visible + num_hidden))

    def forward(self, cond: torch.Tensor):
        x = torch.tanh(self.fc1(cond))
        x = self.fc2(x)
        N, H = self.num_visible, self.num_hidden
        return torch.split(x, [N, N, H, H], dim=-1)

class ConditionalRBM(nn.Module):
    def __init__(self, num_visible: int, num_hidden: int, cond_dim: int,
                 conditioner_width: int = 64, k: int = 1, T: float = 1.0):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.cond_dim = cond_dim
        self.k = k
        self.T = T

        self.W = nn.Parameter(torch.empty(num_visible, num_hidden))
        self.b = nn.Parameter(torch.zeros(num_visible))
        self.c = nn.Parameter(torch.zeros(num_hidden))
        self.conditioner = Conditioner(num_visible, num_hidden, cond_dim, conditioner_width)
        self.initialize_weights()

    def initialize_weights(self, w_mean=0.0, w_std=0.1, bias_val=0.0):
        nn.init.normal_(self.W, mean=w_mean, std=w_std)
        nn.init.constant_(self.b, bias_val)
        nn.init.constant_(self.c, bias_val)

    def _compute_effective_biases(self, cond):
        gamma_b, beta_b, gamma_c, beta_c = self.conditioner(cond)
        b_mod = (1.0 + gamma_b) * self.b.unsqueeze(0) + beta_b
        c_mod = (1.0 + gamma_c) * self.c.unsqueeze(0) + beta_c
        return b_mod, c_mod

    @staticmethod
    def _free_energy(v, W, b, c):
        v = v.to(dtype=W.dtype, device=W.device)
        return -(v * b).sum(dim=-1) - F.softplus(v @ W + c).sum(dim=-1)

    def forward(self, batch, aux_vars):
        values, _, cond = batch
        v_data = values.to(dtype=self.W.dtype, device=self.W.device)
        cond = cond.to(v_data.device, dtype=v_data.dtype)

        l2_strength = float(aux_vars.get("l2_strength", 0.0))

        # 1. Bias Prep
        b_mod, c_mod = self._compute_effective_biases(cond)
        l2_reg = (self.b.unsqueeze(0) - b_mod).pow(2).sum() + (self.c.unsqueeze(0) - c_mod).pow(2).sum()

        # 2. Gibbs Sampling (Python Loop)
        # Init with noise (Global RNG)
        v_model = torch.bernoulli(torch.full_like(v_data, 0.5))

        for _ in range(self.k):
            # Positive
            p_h = torch.sigmoid((v_model @ self.W + c_mod) / self.T)
            h = torch.bernoulli(p_h)
            # Negative
            p_v = torch.sigmoid((h @ self.W.t() + b_mod) / self.T)
            v_model = torch.bernoulli(p_v)

        v_model = v_model.detach()

        # 3. Loss
        fe_data = self._free_energy(v_data, self.W, b_mod, c_mod)
        fe_model = self._free_energy(v_model, self.W, b_mod, c_mod)
        fe_diff = fe_data - fe_model

        loss = fe_diff.mean() + l2_strength * l2_reg
        return loss, {"free_energy_mean": fe_diff.mean().detach(), "free_energy_std": fe_diff.std(unbiased=False).detach()}

    def log_score(self, v, cond):
        b_mod, c_mod = self._compute_effective_biases(cond)
        return -0.5 * self._free_energy(v, self.W, b_mod, c_mod) / self.T

# --- VECTORIZED MONITORING ---
def compute_cxx(samples, pairs, log_score_fn):
    B = samples.shape[0]
    num_pairs = len(pairs)
    device = samples.device

    with torch.no_grad():
        log_scores_orig = log_score_fn(samples)

        # Expand for all flips: (num_pairs * B, N)
        samples_expanded = samples.repeat(num_pairs, 1)

        us = torch.tensor([p[0] for p in pairs], device=device)
        vs = torch.tensor([p[1] for p in pairs], device=device)

        batch_indices = torch.arange(B, device=device).unsqueeze(0).expand(num_pairs, B).flatten()
        pair_offsets = torch.arange(num_pairs, device=device).unsqueeze(1).expand(num_pairs, B).flatten() * B
        flat_indices = pair_offsets + batch_indices

        u_flat = us.unsqueeze(1).expand(num_pairs, B).flatten()
        v_flat = vs.unsqueeze(1).expand(num_pairs, B).flatten()

        # Flip bits
        flipped_samples = samples_expanded.clone()
        flipped_samples[flat_indices, u_flat] = 1.0 - flipped_samples[flat_indices, u_flat]
        flipped_samples[flat_indices, v_flat] = 1.0 - flipped_samples[flat_indices, v_flat]

        # Score
        log_scores_flip = log_score_fn(flipped_samples).view(num_pairs, B)
        ratios = torch.exp(log_scores_flip - log_scores_orig.unsqueeze(0))

        sample_cxx = ratios.mean(dim=0)
        return sample_cxx.mean().item()

def monitor_cxx(model, ds, pair_indices, device, seed):
    model.eval()
    rng = torch.Generator(device='cpu').manual_seed(seed)
    indices = torch.randint(0, len(ds), (min(1000, len(ds)),), generator=rng)

    samples = torch.as_tensor(ds.values[indices], device=device).float()
    cond = torch.as_tensor(ds.system_params[indices], device=device).float()

    scorer = lambda v: model.log_score(v, cond)
    val = compute_cxx(samples, pair_indices, scorer)
    model.train()
    return val

# --- TRAINING LOOPS ---
def train(model, optimizer, loader, num_epochs, ds, pairs, lr_schedule_fn):
    print(f"{'Epoch':<8} | {'Loss':<9} | {'FE Std':<9} | {'Cxx':<9}")
    print("-" * 50)

    steps = 0
    for epoch in range(num_epochs):
        tot_loss = 0
        last_fe_std = 0

        for batch in loader:
            lr = lr_schedule_fn(steps)
            for g in optimizer.param_groups: g['lr'] = lr

            optimizer.zero_grad(set_to_none=True)
            loss, aux = model(batch, {"l2_strength": 1e-4})
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            last_fe_std = aux['free_energy_std'].item()
            steps += 1

        cxx = monitor_cxx(model, ds, pairs, next(model.parameters()).device, 42)
        print(f"{epoch+1:<8} | {tot_loss/len(loader):.4f}    | {last_fe_std:.4f}    | {cxx:.5f}")
    return model

def get_sigmoid_curve(high, low, steps, falloff):
    center = steps / 2.0
    return lambda s: float(low + (high - low) / (1.0 + math.exp(falloff * (min(s, steps) - center))))

# --- MAIN EXECUTION ---
SIDE_LENGTH = 4
TRAIN_SAMPLES = 50_000
delta_support = [0.40, 0.60, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.40, 2.00]
file_paths = [data_dir / f"xxz_{SIDE_LENGTH}x{SIDE_LENGTH}_delta{d:.2f}_5000000.npz" for d in delta_support]

# Load Data
ds = MeasurementDataset(file_paths, load_fn=load_measurements_npz, system_param_keys=["delta"], samples_per_file=[TRAIN_SAMPLES]*len(file_paths))
diag = [k*(SIDE_LENGTH+1) for k in range(SIDE_LENGTH)]
pairs = list(zip(diag, diag[1:]))

# Config
BATCH_SIZE = 1024
K_STEPS = 5  # Standard for RBMs, fast enough for Python loop on MPS
EPOCHS = 10
SEED = 42

torch.manual_seed(SEED)
loader = MeasurementLoader(ds, batch_size=BATCH_SIZE, shuffle=True, rng=torch.Generator(device='cpu').manual_seed(SEED))

model = ConditionalRBM(ds.num_qubits, 64, 1, k=K_STEPS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
lr_fn = get_sigmoid_curve(1e-2, 1e-3, EPOCHS * len(loader), 0.0005)

# Run
train(model, optimizer, loader, EPOCHS, ds, pairs, lr_fn)