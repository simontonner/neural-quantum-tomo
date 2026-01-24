import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import itertools

# --- IMPORTS FROM YOUR PROJECT ---
sys.path.append(str(Path("..").resolve()))
from data_handling import load_measurements_npz, MeasurementDataset, MeasurementLoader
from wavefunction_overlap import generate_basis_states, calculate_exact_overlap, load_gt_wavefunction

# --- CONFIGURATION ---
# Representative points: One easy, one medium, one hard
H_POINTS = [2.50, 3.00, 3.50]
N_SAMPLES = 2000
SIDE_LENGTH = 4
EPOCHS = 50
BATCH_SIZE = 1024
HIDDEN_DIM = 64
K_STEPS = 10
SEED = 42

# --- THE TUNING GRID ---
# We test a wide range of LRs and Weight Decay (L2 Reg)
# If the Single RBM is overfitting, Weight Decay should fix it.
GRID = {
    "lr": [0.001, 0.005, 0.01, 0.02, 0.05],
    "weight_decay": [0.0, 1e-4, 1e-3, 1e-2]
}

# The settings you used in your paper/notebook
BASELINE_CONFIG = {"lr": 1e-2, "weight_decay": 0.0}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("measurements")
STATE_DIR = Path("state_vectors")

# --- CORRECTED SYMMETRIC RBM CLASS ---
class SymmetricRBM(nn.Module):
    def __init__(self, num_v: int, num_h: int, k: int = 10):
        super().__init__()
        self.num_v = num_v
        self.num_h = num_h
        self.k = k
        self.T = 1.0
        self.W = nn.Parameter(torch.empty(num_v, num_h))
        self.b = nn.Parameter(torch.zeros(num_v))
        self.c = nn.Parameter(torch.zeros(num_h))
        self.initialize_weights()

    def initialize_weights(self, std: float = 0.01):
        nn.init.normal_(self.W, std=std)
        nn.init.constant_(self.b, 0.0)
        nn.init.constant_(self.c, 0.0)

    def _free_energy(self, v: torch.Tensor) -> torch.Tensor:
        b, c = self.b.unsqueeze(0), self.c.unsqueeze(0)
        vW = v @ self.W
        W_colsum = self.W.sum(dim=0)

        preact_normal  = vW + c
        preact_flipped = (W_colsum.unsqueeze(0) - vW) + c

        negF = torch.stack([
            (v * b).sum(-1) - F.softplus(preact_normal).sum(-1),
            ((1.0 - v) * b).sum(-1) - F.softplus(preact_flipped).sum(-1)
        ], dim=-1)
        return -self.T * torch.logsumexp(negF / self.T, dim=-1)

    def forward(self, batch, aux_vars):
        rng = aux_vars.get("rng")

        # FIX: Explicit cast to float to avoid Byte/Float errors
        v_data = batch[0].to(device=self.W.device, dtype=self.W.dtype)

        v_model = v_data.clone()
        B = v_data.size(0)
        u = torch.bernoulli(torch.full((B, 1), 0.5, device=self.W.device), generator=rng)

        b, c = self.b.unsqueeze(0), self.c.unsqueeze(0)
        for _ in range(self.k):
            v_branch = u * v_model + (1.0 - u) * (1.0 - v_model)
            h = torch.bernoulli(torch.sigmoid((v_branch @ self.W + c)), generator=rng)
            a = h @ self.W.t()

            vb = (v_model * b).sum(dim=-1)
            va = (v_model * a).sum(dim=-1)
            dE = (-b.sum(dim=-1) - a.sum(dim=-1) + 2.0*vb + 2.0*va)
            u = torch.bernoulli(torch.sigmoid(dE), generator=rng).unsqueeze(-1)

            v_new = torch.bernoulli(torch.sigmoid(a + b), generator=rng)
            v_model = u * v_new + (1.0 - u) * (1.0 - v_new)

        loss = self._free_energy(v_data).mean() - self._free_energy(v_model.detach()).mean()
        return loss

    def log_score(self, v, cond=None):
        return -0.5 * self._free_energy(v) / self.T

# --- EXPERIMENT LOGIC ---
def train_and_eval(h_val, config):
    # 1. Setup Data
    file_path = DATA_DIR / f"tfim_{SIDE_LENGTH}x{SIDE_LENGTH}_h{h_val:.2f}_20000.npz"
    gt_path = STATE_DIR / f"tfim_{SIDE_LENGTH}x{SIDE_LENGTH}_h{h_val:.2f}.npz"

    if not file_path.exists() or not gt_path.exists():
        print(f"Skipping h={h_val} (File not found)")
        return 0.0

    dataset = MeasurementDataset([file_path], load_measurements_npz, ["h"], [N_SAMPLES])
    psi_true = load_gt_wavefunction(gt_path, DEVICE)
    basis_states = generate_basis_states(SIDE_LENGTH**2, DEVICE)

    # 2. Reset everything for fairness
    torch.manual_seed(SEED)
    rng = torch.Generator(device=DEVICE).manual_seed(SEED)
    loader = MeasurementLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, rng=rng)

    model = SymmetricRBM(dataset.num_qubits, HIDDEN_DIM, K_STEPS).to(DEVICE)

    # 3. Apply Config
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # 4. Train
    model.train()
    for epoch in range(EPOCHS):
        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch, {"rng": rng})
            loss.backward()
            optimizer.step()

    # 5. Eval
    model.eval()
    overlap = calculate_exact_overlap(model, h_val, psi_true, basis_states)
    return overlap

def run_suite():
    print(f"### ULTIMATE SINGLE RBM 'STEEL MAN' TEST ###")
    print(f"Testing Grid: {GRID}")
    print(f"Points: {H_POINTS}")
    print("-" * 60)

    summary_data = []

    # Cartesian product of all hyperparameters
    keys, values = zip(*GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for h in H_POINTS:
        print(f"\n>> Analyzing h = {h:.2f}")

        # 1. Run Baseline (User's Paper Setting)
        print("   Running Baseline...", end="")
        baseline_ov = train_and_eval(h, BASELINE_CONFIG)
        print(f" Done. Overlap: {baseline_ov:.5f}")

        best_ov = -1.0
        best_cfg = {}

        # 2. Run Grid Search
        print(f"   Running Grid ({len(param_combinations)} configs)...")
        for i, cfg in enumerate(param_combinations):
            ov = train_and_eval(h, cfg)

            # Optional: print every step
            # print(f"     [{i+1}] LR={cfg['lr']}, WD={cfg['weight_decay']} -> {ov:.5f}")

            if ov > best_ov:
                best_ov = ov
                best_cfg = cfg

        print(f"   *** BEST FOUND *** LR={best_cfg['lr']}, WD={best_cfg['weight_decay']} -> Overlap: {best_ov:.5f}")

        gain = best_ov - baseline_ov
        summary_data.append({
            "h": h,
            "Baseline": baseline_ov,
            "Best Tuned": best_ov,
            "Best LR": best_cfg['lr'],
            "Best WD": best_cfg['weight_decay'],
            "Gain": gain
        })

    # --- FINAL VERDICT ---
    print("\n" + "="*60)
    print("FINAL SUMMARY REPORT")
    print("="*60)
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False, float_format="%.5f"))
    print("-" * 60)

    avg_gain = df["Gain"].mean()
    print(f"Average Improvement from Tuning: {avg_gain:.5f}")

    if avg_gain < 0.005:
        print("\n>> VERDICT: USER IS RIGHT.")
        print("   Optimal tuning improved the Single RBM by less than 0.5%.")
        print("   The performance gap is caused by data scarcity (Architecture),")
        print("   not by hyperparameters (Optimization).")
    else:
        print("\n>> VERDICT: COLLEAGUE IS RIGHT.")
        print(f"   Optimal tuning improved the Single RBM by {avg_gain:.4f}.")
        print("   You should update your paper to use the tuned hyperparameters.")

if __name__ == "__main__":
    run_suite()