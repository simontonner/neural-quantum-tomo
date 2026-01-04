import os
import sys
import math
import itertools
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# --- PATH SETUP ---
# Ensure we can find data_handling.py
sys.path.append(str(Path("..").resolve()))
from data_handling import load_measurements_npz, load_state_npz, MeasurementDataset, MeasurementLoader

# --- CONFIGURATION FOR THE SEARCH ---
DATA_DIR  = Path("measurements")
STATE_DIR = Path("state_vectors")
OUT_FILE  = Path("best_hyperparams.json")

# We optimize on the HARDEST case: Small data size.
# If it works here, it works everywhere.
SEARCH_SAMPLE_SIZE = 2_000

# The Hyperparameter Grid
HPARAM_GRID = {
    "noise_frac":  [0.0, 0.05, 0.1, 0.2],   # The balance between exploring/exploiting
    "l2_strength": [0.0, 1e-5, 1e-4],       # Smoothing the weights
    "init_lr":     [1e-2, 5e-3],            # Learning rate
}

# Fixed params
BATCH_SIZE = 1024
EPOCHS     = 40
K_STEPS    = 10   # Keep this high for quality
COND_WIDTH = 64
HIDDEN     = 16

# Interpolation points to evaluate (The "Novel" points are the test set)
# We focus specifically on the "danger zones" discussed
H_EVAL_NOVEL = [1.50, 2.50, 2.80, 3.20]

# --- MODEL DEFINITION ---
class Conditioner(nn.Module):
    def __init__(self, num_visible, num_hidden, cond_dim, hidden_width):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, 2 * (num_visible + num_hidden))
        self.n, self.h = num_visible, num_hidden

    def forward(self, cond):
        x = torch.tanh(self.fc1(cond))
        x = self.fc2(x)
        return torch.split(x, [self.n, self.n, self.h, self.h], dim=-1)

class ConditionalRBM(nn.Module):
    def __init__(self, num_visible, num_hidden, cond_dim=1, conditioner_width=64, k=5):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_visible, num_hidden))
        self.b = nn.Parameter(torch.zeros(num_visible))
        self.c = nn.Parameter(torch.zeros(num_hidden))
        self.k = k
        nn.init.normal_(self.W, std=0.01)
        self.conditioner = Conditioner(num_visible, num_hidden, cond_dim, conditioner_width)

    def _effective_biases(self, cond):
        gb, bb, gc, bc = self.conditioner(cond)
        return (1+gb)*self.b.unsqueeze(0)+bb, (1+gc)*self.c.unsqueeze(0)+bc

    def forward(self, batch, aux):
        v_data, _, cond = batch
        v_data, cond = v_data.to(self.W.device).float(), cond.to(self.W.device).float()

        b_mod, c_mod = self._effective_biases(cond)

        # CD-k
        v = v_data.clone()
        # Noise Injection
        if aux["noise"] > 0:
            n = int(v.shape[0] * aux["noise"])
            if n > 0: v[:n] = torch.bernoulli(torch.full_like(v[:n], 0.5))

        for _ in range(self.k):
            ph = torch.sigmoid(v @ self.W + c_mod)
            h = torch.bernoulli(ph)
            pv = torch.sigmoid(h @ self.W.t() + b_mod)
            v = torch.bernoulli(pv)
        v = v.detach()

        # Free Energy
        def fe(val):
            return -(val*b_mod).sum(-1) - F.softplus(val@self.W + c_mod).sum(-1)

        loss = (fe(v_data) - fe(v)).mean()

        # L2 Reg
        if aux["l2"] > 0:
            reg = (self.b.unsqueeze(0)-b_mod).pow(2).sum() + (self.c.unsqueeze(0)-c_mod).pow(2).sum()
            loss += aux["l2"] * reg

        return loss

    @torch.no_grad()
    def get_wavefunction(self, h_val, all_states):
        cond = torch.tensor([[h_val]], device=self.W.device).expand(len(all_states), -1)
        b_mod, c_mod = self._effective_biases(cond)
        # Log Psi = -0.5 * FE
        fe = -(all_states*b_mod).sum(-1) - F.softplus(all_states@self.W + c_mod).sum(-1)
        log_psi = -0.5 * fe
        # Normalize
        log_Z = torch.logsumexp(2.0 * log_psi, dim=0)
        return torch.exp(log_psi - 0.5 * log_Z)

# --- UTILS ---
def generate_all_states(n, device):
    return torch.tensor(list(itertools.product([0, 1], repeat=n)), dtype=torch.float32, device=device)

def get_scheduler(lr, steps):
    return lambda s: lr * (1.0 - s/steps) # Simple linear decay for speed

# --- MAIN SEARCH LOOP ---

def run_search():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Grid Search on {device}")

    # 1. Load Data (N=2000)
    h_support = [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00]
    fnames = [f"tfim_4x4_h{h:.2f}_50000.npz" for h in h_support]
    fpaths = [DATA_DIR / fn for fn in fnames]

    ds = MeasurementDataset(fpaths, load_measurements_npz, ["h"], [SEARCH_SAMPLE_SIZE]*len(fpaths))
    loader = MeasurementLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, rng=torch.Generator())
    print(f"Data Loaded: {len(ds)} samples (Stress Test Mode)")

    all_states = generate_all_states(16, device)

    # 2. Generate Grid
    keys, values = zip(*HPARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []

    print(f"\nEvaluating {len(combinations)} configurations...")
    print(f"{'Noise':<6} | {'L2':<8} | {'LR':<8} | {'Avg Overlap':<12} | {'Min Overlap':<12}")
    print("-" * 65)

    for i, config in enumerate(combinations):
        # Reset Seed for Fairness
        torch.manual_seed(42)

        model = ConditionalRBM(16, HIDDEN, cond_dim=1, conditioner_width=COND_WIDTH, k=K_STEPS).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["init_lr"])

        # Train
        steps = len(loader) * EPOCHS
        model.train()
        for epoch in range(EPOCHS):
            lr = config["init_lr"] * (1 - epoch/EPOCHS) # Simple linear decay
            for g in optimizer.param_groups: g['lr'] = lr

            for batch in loader:
                optimizer.zero_grad()
                loss = model(batch, {"noise": config["noise_frac"], "l2": config["l2_strength"]})
                loss.backward()
                optimizer.step()

        # Evaluate on Novel Points (The Test)
        overlaps = []
        for h_val in H_EVAL_NOVEL:
            gt_path = STATE_DIR / f"tfim_4x4_h{h_val:.2f}.npz"
            if not gt_path.exists(): continue

            psi_gt, _ = load_state_npz(gt_path)
            psi_gt = torch.from_numpy(psi_gt).real.float().to(device)
            psi_gt /= torch.norm(psi_gt)

            psi_model = model.get_wavefunction(h_val, all_states)
            ov = torch.abs(torch.dot(psi_gt, psi_model)).item()
            overlaps.append(ov)

        avg_ov = np.mean(overlaps)
        min_ov = np.min(overlaps)

        print(f"{config['noise_frac']:<6.2f} | {config['l2_strength']:<8.0e} | {config['init_lr']:<8.0e} | {avg_ov:.5f}      | {min_ov:.5f}")

        results.append({
            "config": config,
            "avg_overlap": avg_ov,
            "min_overlap": min_ov, # Critical for finding the h=1.5 failure
            "overlaps": overlaps
        })

    # 3. Select Winner
    # Criteria: Best Worst-Case Performance (Maximize Min Overlap)
    # This specifically targets the h=1.5 dip
    best_run = max(results, key=lambda x: x["min_overlap"])

    print("\n" + "="*40)
    print("WINNING CONFIGURATION")
    print("="*40)
    print(json.dumps(best_run["config"], indent=2))
    print(f"Average Overlap (Novel): {best_run['avg_overlap']:.6f}")
    print(f"Worst Overlap (Novel):   {best_run['min_overlap']:.6f}")

    with open(OUT_FILE, "w") as f:
        json.dump(best_run, f, indent=2)
    print(f"Saved to {OUT_FILE}")

if __name__ == "__main__":
    run_search()