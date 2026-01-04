import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

# === PATH SETUP ===
sys.path.append(str(Path("..").resolve()))
from data_handling import load_measurements_npz, MeasurementDataset, MeasurementLoader

data_dir = Path("measurements")

# ==========================================
# 1. RBM MODEL (Fixed Architecture)
# ==========================================
class RBM(nn.Module):
    def __init__(self, num_visible: int, num_hidden: int, k: int = 20, T: float = 1.0):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.T = T

        self.W = nn.Parameter(torch.empty(num_visible, num_hidden))
        self.b = nn.Parameter(torch.zeros(num_visible))
        self.c = nn.Parameter(torch.zeros(num_hidden))

        # Standard Init
        self.initialize_weights(w_std=0.05)

    def initialize_weights(self, w_mean: float = 0.0, w_std: float = 0.05, bias_val: float = 0.0):
        nn.init.normal_(self.W, mean=w_mean, std=w_std)
        nn.init.constant_(self.b, bias_val)
        nn.init.constant_(self.c, bias_val)

    @staticmethod
    def _free_energy(v: torch.Tensor, W: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        v = v.to(dtype=W.dtype, device=W.device)
        term1 = -(v * b).sum(dim=-1)
        term2 = F.softplus(v @ W + c).sum(dim=-1)
        return term1 - term2

    def _gibbs_step(self, v: torch.Tensor, T: float, rng: torch.Generator):
        p_h = torch.sigmoid((v @ self.W + self.c) / T)
        h = torch.bernoulli(p_h, generator=rng)
        p_v = torch.sigmoid((h @ self.W.t() + self.b) / T)
        v_next = torch.bernoulli(p_v, generator=rng)
        return v_next

    def log_score(self, v: torch.Tensor) -> torch.Tensor:
        fe = self._free_energy(v, self.W, self.b, self.c)
        return -0.5 * fe / self.T

    def forward(self, batch: Tuple[torch.Tensor, ...], aux_vars: Dict[str, Any]):
        values, _, _ = batch
        v_data = values.to(dtype=self.W.dtype, device=self.W.device)
        rng = aux_vars.get("rng", torch.Generator(device="cpu"))

        # CD-k: Start from Data
        v_model = v_data.clone()
        for _ in range(self.k):
            v_model = self._gibbs_step(v_model, self.T, rng)
        v_model = v_model.detach()

        # Gradients
        fe_data = self._free_energy(v_data, self.W, self.b, self.c)
        fe_model = self._free_energy(v_model, self.W, self.b, self.c)
        loss = (fe_data - fe_model).mean()

        return loss

    @torch.no_grad()
    def generate(self, n_samples: int, T_schedule: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
        device = next(self.parameters()).device
        probs = torch.full((n_samples, self.num_visible), 0.5, device=device, dtype=torch.float32)
        v = torch.bernoulli(probs, generator=rng)

        if T_schedule.dim() == 0: T_schedule = T_schedule.view(1)
        for i in range(int(T_schedule.shape[0])):
            v = self._gibbs_step(v, float(T_schedule[i]), rng)
        return v

# ==========================================
# 2. UTILS
# ==========================================
def compute_nearest_neighbor_corr(samples: torch.Tensor) -> float:
    spins = 2.0 * samples.float() - 1.0
    term = spins[:, :-1] * spins[:, 1:]
    return term.mean().item()

def compute_renyi_entropy_direct(samples: torch.Tensor, subsystem_size: int, score_fn: Callable) -> float:
    n_samples = samples.shape[0]
    half = n_samples // 2
    if half == 0: return 0.0
    batch1 = samples[:half]
    batch2 = samples[half:2 * half]
    idx_B = torch.arange(subsystem_size, samples.shape[1], device=samples.device)

    log_psi_1 = score_fn(batch1)
    log_psi_2 = score_fn(batch2)

    batch1_swap = batch1.clone()
    batch1_swap[:, idx_B] = batch2[:, idx_B]
    batch2_swap = batch2.clone()
    batch2_swap[:, idx_B] = batch1[:, idx_B]

    log_psi_1_swap = score_fn(batch1_swap)
    log_psi_2_swap = score_fn(batch2_swap)

    log_ratio = (log_psi_1_swap + log_psi_2_swap) - (log_psi_1 + log_psi_2)
    log_ratio = torch.clamp(log_ratio, max=30.0)

    swap_expectation = torch.exp(log_ratio).mean().item()
    if swap_expectation <= 1e-12: return 0.0
    return -math.log(swap_expectation)

# ==========================================
# 3. EXPERIMENT RUNNER
# ==========================================
def run_experiment(batch_size: int, ds: MeasurementDataset, config: Dict):

    print(f"\n>>> STARTING EXPERIMENT: Batch Size={batch_size} <<<")

    # 1. Setup Loader with specific batch size
    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=batch_size, shuffle=True, drop_last=True, rng=rng_loader)

    # 2. Setup Model & Optimizer
    SEED = 42
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)

    model = RBM(num_visible=config['N'], num_hidden=config['H'], k=config['k'], T=1.0)

    # Keep LR fixed to see pure effect of batch size
    # (Technically larger batches can handle larger LR, but let's test ceteris paribus first)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    # 3. Training Loop
    start_time = time.time()

    for epoch in range(config['epochs']):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch, {"rng": rng})
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Progress dots
        if (epoch+1) % 20 == 0:
            print(f". (Ep {epoch+1})", end="", flush=True)

    total_time = time.time() - start_time
    print(f"\nDone. Time: {total_time:.2f}s")

    # 4. Final Correlation Check
    with torch.no_grad():
        gen_samples = model.generate(5000, torch.ones(20), rng)
        final_czz = compute_nearest_neighbor_corr(gen_samples)

    return model, total_time, final_czz

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":

    CHAIN_LENGTH = 16
    TARGET_DELTA = 1.00
    TRAIN_SAMPLES = 100_000

    # HYPERPARAMS (Fixed based on previous best results)
    config = {
        'N': CHAIN_LENGTH,
        'H': 32,          # Alpha = 2 (Chosen sweet spot)
        'epochs': 120,
        'lr': 0.05,
        'k': 20
    }

    # LOAD DATA ONCE
    file_name = f"xxz_{CHAIN_LENGTH}_delta{TARGET_DELTA:.2f}_5000000.npz"
    file_path = data_dir / file_name

    print(f"Loading {file_path}...")
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[TRAIN_SAMPLES])

    # SWEEP
    batch_sizes = [128, 256, 512, 1024]

    results = {}
    curves = {}

    for b_size in batch_sizes:
        model, duration, czz = run_experiment(b_size, ds, config)

        print(f"Evaluating S2 for BS={b_size}...")
        T_schedule = torch.cat([torch.linspace(2.0, 1.0, 50), torch.ones(50)])
        with torch.no_grad():
            eval_samples = model.generate(20000, T_schedule, torch.Generator().manual_seed(123))

        s2_curve = []
        scorer = lambda v: model.log_score(v)
        l_axis = list(range(1, CHAIN_LENGTH // 2 + 1))

        for l in l_axis:
            val = compute_renyi_entropy_direct(eval_samples, l, scorer)
            s2_curve.append(val)

        results[b_size] = {'time': duration, 'czz': czz}
        curves[b_size] = s2_curve

    # REPORT
    print("\n" + "="*50)
    print("FINAL COMPARISON REPORT (BATCH SIZE SWEEP)")
    print("="*50)
    print(f"{'Batch':<6} | {'Time (s)':<10} | {'Final Czz':<10} | {'S2(L=8)':<10}")
    print("-" * 50)

    for b in batch_sizes:
        res = results[b]
        s2_end = curves[b][-1]
        print(f"{b:<6} | {res['time']:<10.2f} | {res['czz']:<10.4f} | {s2_end:<10.4f}")

    print("-" * 50)

    # PLOT
    ref_file = Path(f"xxz_{CHAIN_LENGTH}_entropy_ref.csv")
    plt.figure(figsize=(9, 6))

    if ref_file.exists():
        df_ref = pd.read_csv(ref_file)
        l_cols = sorted([c for c in df_ref.columns if c.startswith("l")], key=lambda s: int(s[1:]))
        l_axis_ref = [int(c[1:]) for c in l_cols]
        mask = np.isclose(df_ref["delta"], TARGET_DELTA)
        if mask.any():
            s2_ref = df_ref.loc[mask].iloc[0][l_cols].to_numpy()
            plt.plot(l_axis_ref, s2_ref, 'k--', linewidth=2, label='Exact Reference')

    colors = {128: "cyan", 256: "blue", 512: "orange", 1024: "red"}

    for b in batch_sizes:
        plt.plot(l_axis, curves[b], color=colors[b], marker="o",
                 label=f"BS={b} ({results[b]['time']:.0f}s)")

    plt.title(f"Batch Size Comparison (XXZ N={CHAIN_LENGTH}, $\Delta=1.0$)")
    plt.xlabel(r"Subsystem size $\ell$")
    plt.ylabel(r"Rényi Entropy $S_2$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()