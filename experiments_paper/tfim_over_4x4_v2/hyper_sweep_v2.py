import os
import sys
import itertools
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# --- PATH SETUP ---
sys.path.append(str(Path("..").resolve()))
from data_handling import load_measurements_npz, load_state_npz, MeasurementDataset, MeasurementLoader

# --- CONFIGURATION ---
DATA_DIR  = Path("measurements")
STATE_DIR = Path("state_vectors")
OUT_FILE  = Path("best_arch.json")

SEARCH_SAMPLE_SIZE = 2_000 # Still stressing the data efficiency

# --- THE NEW GRID: ARCHITECTURE FOCUSED ---
HPARAM_GRID = {
    # Testing if we need more capacity to interpolate correctly
    "conditioner_width": [32, 64, 128, 256],
    "num_hidden":        [8, 16, 24], # "alpha" density
}

# FIXED WINNERS FROM EXP 1
FIXED_PARAMS = {
    "noise_frac":  0.05,
    "l2_strength": 0.0,
    "init_lr":     0.01
}

# --- MODEL DEFINITIONS (Standard) ---
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
    def __init__(self, num_visible, num_hidden, cond_dim=1, conditioner_width=64, k=10):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_visible, num_hidden))
        self.b = nn.Parameter(torch.zeros(num_visible))
        self.c = nn.Parameter(torch.zeros(num_hidden))
        self.k = k
        self.conditioner = Conditioner(num_visible, num_hidden, cond_dim, conditioner_width)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.W, std=0.01)
        nn.init.constant_(self.b, 0.0)
        nn.init.constant_(self.c, 0.0)

    def _effective_biases(self, cond):
        gb, bb, gc, bc = self.conditioner(cond)
        return (1+gb)*self.b.unsqueeze(0)+bb, (1+gc)*self.c.unsqueeze(0)+bc

    def forward(self, batch, aux):
        v_data, _, cond = batch
        v_data, cond = v_data.to(self.W.device).float(), cond.to(self.W.device).float()
        b_mod, c_mod = self._effective_biases(cond)

        # CD-k
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
        b_mod, c_mod = self._effective_biases(cond)
        fe = -(all_states*b_mod).sum(-1) - F.softplus(all_states@self.W + c_mod).sum(-1)
        log_psi = -0.5 * fe
        log_Z = torch.logsumexp(2.0 * log_psi, dim=0)
        return torch.exp(log_psi - 0.5 * log_Z)

# --- UTILS ---
def generate_all_states(n, device):
    return torch.tensor(list(itertools.product([0, 1], repeat=n)), dtype=torch.float32, device=device)

def run_search():
    device = torch.device("cpu") # CPU is fine for 4x4

    # Load Data
    h_support = [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00]
    fpaths = [DATA_DIR / f"tfim_4x4_h{h:.2f}_50000.npz" for h in h_support]
    ds = MeasurementDataset(fpaths, load_measurements_npz, ["h"], [SEARCH_SAMPLE_SIZE]*len(fpaths))
    loader = MeasurementLoader(ds, batch_size=1024, shuffle=True, drop_last=False, rng=torch.Generator())

    all_states = generate_all_states(16, device)

    # Grid Setup
    keys, values = zip(*HPARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Evaluating {len(combinations)} architectures using FIXED params: {FIXED_PARAMS}")
    print(f"{'CondWidth':<10} | {'Hidden':<8} | {'Avg Overlap':<12} | {'Min Overlap':<12}")
    print("-" * 50)

    results = []

    for config in combinations:
        torch.manual_seed(42)

        model = ConditionalRBM(16, config["num_hidden"], cond_dim=1,
                               conditioner_width=config["conditioner_width"], k=10).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=FIXED_PARAMS["init_lr"])

        # Train
        model.train()
        EPOCHS = 40
        for epoch in range(EPOCHS):
            lr = FIXED_PARAMS["init_lr"] * (1 - epoch/EPOCHS)
            for g in optimizer.param_groups: g['lr'] = lr
            for batch in loader:
                optimizer.zero_grad()
                loss = model(batch, {"noise": FIXED_PARAMS["noise_frac"]})
                loss.backward()
                optimizer.step()

        # Evaluate (Focus on Generalization)
        overlaps = []
        # Evaluating on NOVEL points (Generalization)
        for h_val in [1.50, 2.50, 2.80, 3.20]:
            gt_path = STATE_DIR / f"tfim_4x4_h{h_val:.2f}.npz"
            if not gt_path.exists(): continue
            psi_gt, _ = load_state_npz(gt_path)
            psi_gt = torch.from_numpy(psi_gt).real.float().to(device)
            psi_gt /= torch.norm(psi_gt)
            psi_model = model.get_wavefunction(h_val, all_states)
            overlaps.append(torch.abs(torch.dot(psi_gt, psi_model)).item())

        avg_ov = np.mean(overlaps)
        min_ov = np.min(overlaps)

        print(f"{config['conditioner_width']:<10} | {config['num_hidden']:<8} | {avg_ov:.5f}      | {min_ov:.5f}")

        results.append({**config, "avg_overlap": avg_ov, "min_overlap": min_ov})

    best = max(results, key=lambda x: x["min_overlap"])
    print("\nBEST ARCHITECTURE:")
    print(json.dumps(best, indent=2))

if __name__ == "__main__":
    run_search()