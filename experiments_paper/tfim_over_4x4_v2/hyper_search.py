import os
import sys
import itertools
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import math

# --- PATH SETUP ---
sys.path.append(str(Path("..").resolve()))
from data_handling import load_measurements_npz, load_state_npz, MeasurementDataset, MeasurementLoader

# --- CONFIGURATION ---
DATA_DIR  = Path("measurements")
STATE_DIR = Path("state_vectors")
OUT_FILE  = Path("sweep_results.csv")

# We stick to the HARD setting: N=2000
SEARCH_SAMPLE_SIZE = 2_000

# THE ULTIMATE GRID
HPARAM_GRID = {
    "noise_frac":  [0.0, 0.1, 0.2, 0.3],    # Testing heavy noise
    "k_steps":     [10, 20, 30],            # Testing deep chains
    "batch_size":  [256, 1024],             # Gradient dynamics
}

# Fixed Model Architecture
FIXED_PARAMS = {
    "num_hidden": 16,
    "conditioner_width": 64,
    "cond_dim": 1,
    "epochs": 50,
    "init_lr": 1e-2,
}

# Eval Points
H_SUPPORT = [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00]
H_NOVEL   = [1.50, 2.50, 2.80, 3.20]

# --- MODEL DEFINITIONS ---
class Conditioner(nn.Module):
    def __init__(self, n, h, cd, cw):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cd, cw), nn.Tanh(),
            nn.Linear(cw, 2*(n+h))
        )
        self.n, self.h = n, h
    def forward(self, c):
        return torch.split(self.net(c), [self.n, self.n, self.h, self.h], dim=-1)

class ConditionalRBM(nn.Module):
    def __init__(self, n, h, cd, cw, k):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n, h)); nn.init.normal_(self.W, std=0.01)
        self.b = nn.Parameter(torch.zeros(n))
        self.c = nn.Parameter(torch.zeros(h))
        self.k = k
        self.conditioner = Conditioner(n, h, cd, cw)

    def _biases(self, cond):
        gb, bb, gc, bc = self.conditioner(cond)
        return (1+gb)*self.b.unsqueeze(0)+bb, (1+gc)*self.c.unsqueeze(0)+bc

    def forward(self, batch, aux):
        v_data, _, cond = batch
        v_data, cond = v_data.to(self.W.device).float(), cond.to(self.W.device).float()
        b_mod, c_mod = self._biases(cond)

        v = v_data.clone()
        if aux["noise"] > 0:
            n = int(v.shape[0] * aux["noise"])
            if n > 0: v[:n] = torch.bernoulli(torch.full_like(v[:n], 0.5))

        for _ in range(self.k):
            ph = torch.sigmoid(v @ self.W + c_mod)
            h = torch.bernoulli(ph)
            pv = torch.sigmoid(h @ self.W.t() + b_mod)
            v = torch.bernoulli(pv)
        v = v.detach()

        def fe(val): return -(val*b_mod).sum(-1) - F.softplus(val@self.W + c_mod).sum(-1)
        return (fe(v_data) - fe(v)).mean()

    @torch.no_grad()
    def get_wavefunction(self, h_val, all_states):
        cond = torch.tensor([[h_val]], device=self.W.device).expand(len(all_states), -1)
        b_mod, c_mod = self._biases(cond)
        fe = -(all_states*b_mod).sum(-1) - F.softplus(all_states@self.W + c_mod).sum(-1)
        log_psi = -0.5 * fe
        log_Z = torch.logsumexp(2.0 * log_psi, dim=0)
        return torch.exp(log_psi - 0.5 * log_Z)

# --- UTILS ---
def generate_all_states(n, device):
    return torch.tensor(list(itertools.product([0, 1], repeat=n)), dtype=torch.float32, device=device)

def run_sweep():
    device = torch.device("cpu")
    print(f"Running Ultimate Sweep on {device}")

    # Load Data
    fnames = [f"tfim_4x4_h{h:.2f}_50000.npz" for h in H_SUPPORT]
    fpaths = [DATA_DIR / fn for fn in fnames]
    ds_raw = MeasurementDataset(fpaths, load_measurements_npz, ["h"], [SEARCH_SAMPLE_SIZE]*len(fpaths))
    print(f"Data Loaded: {len(ds_raw)} samples")

    all_states = generate_all_states(16, device)

    # Grid
    keys, values = zip(*HPARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []
    print(f"Evaluating {len(combinations)} configurations...")

    for i, cfg in enumerate(combinations):
        torch.manual_seed(42) # Fixed seed for fair comparison

        loader = MeasurementLoader(ds_raw, batch_size=cfg["batch_size"], shuffle=True, drop_last=False, rng=torch.Generator())

        model = ConditionalRBM(16, FIXED_PARAMS["num_hidden"], FIXED_PARAMS["cond_dim"],
                               FIXED_PARAMS["conditioner_width"], cfg["k_steps"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=FIXED_PARAMS["init_lr"])

        # Schedule
        steps = len(loader) * FIXED_PARAMS["epochs"]
        # Fast decay schedule (hardcoded as it won previously)
        def scheduler(s): return FIXED_PARAMS["init_lr"] * (1.0 - s/steps)

        model.train()
        gs = 0
        for epoch in range(FIXED_PARAMS["epochs"]):
            for batch in loader:
                for g in optimizer.param_groups: g['lr'] = scheduler(gs)
                optimizer.zero_grad()
                loss = model(batch, {"noise": cfg["noise_frac"]})
                loss.backward()
                optimizer.step()
                gs += 1

        # --- EVALUATION ---
        # 1. Support Overlap (Training Performance)
        supp_ovs = []
        for h_val in H_SUPPORT:
            gt_path = STATE_DIR / f"tfim_4x4_h{h_val:.2f}.npz"
            psi_gt, _ = load_state_npz(gt_path)
            psi_gt = torch.from_numpy(psi_gt).real.float().to(device); psi_gt /= torch.norm(psi_gt)
            psi_model = model.get_wavefunction(h_val, all_states)
            supp_ovs.append(torch.abs(torch.dot(psi_gt, psi_model)).item())

        # 2. Novel Overlap (Generalization Performance)
        nov_ovs = []
        for h_val in H_NOVEL:
            gt_path = STATE_DIR / f"tfim_4x4_h{h_val:.2f}.npz"
            psi_gt, _ = load_state_npz(gt_path)
            psi_gt = torch.from_numpy(psi_gt).real.float().to(device); psi_gt /= torch.norm(psi_gt)
            psi_model = model.get_wavefunction(h_val, all_states)
            nov_ovs.append(torch.abs(torch.dot(psi_gt, psi_model)).item())

        entry = {
            **cfg,
            "train_overlap_avg": np.mean(supp_ovs),
            "test_overlap_avg":  np.mean(nov_ovs),
            "test_overlap_min":  np.min(nov_ovs) # The h=1.5 check
        }
        results.append(entry)

        print(f"[{i+1}/{len(combinations)}] Noise={cfg['noise_frac']} k={cfg['k_steps']} B={cfg['batch_size']} -> "
              f"Train: {entry['train_overlap_avg']:.4f} | Test: {entry['test_overlap_avg']:.4f}")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUT_FILE, index=False)
    print(f"\nResults saved to {OUT_FILE}")

if __name__ == "__main__":
    run_sweep()