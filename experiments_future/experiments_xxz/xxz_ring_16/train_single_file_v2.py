import os
import sys
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

# === RBM MODEL ===

class RBM(nn.Module):
    def __init__(self, num_visible: int, num_hidden: int, k: int = 1, T: float = 1.0):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.T = T

        self.W = nn.Parameter(torch.empty(num_visible, num_hidden))
        self.b = nn.Parameter(torch.zeros(num_visible))
        self.c = nn.Parameter(torch.zeros(num_hidden))

        # Initialization
        self.initialize_weights(w_std=0.01)

    def initialize_weights(self, w_mean: float = 0.0, w_std: float = 0.01, bias_val: float = 0.0):
        nn.init.normal_(self.W, mean=w_mean, std=w_std)
        nn.init.constant_(self.b, bias_val)
        nn.init.constant_(self.c, bias_val)

    @staticmethod
    def _free_energy(v: torch.Tensor, W: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        v = v.to(dtype=W.dtype, device=W.device)
        # FE = -b*v - sum(log(1 + exp(W*v + c)))
        term1 = -(v * b).sum(dim=-1)
        term2 = F.softplus(v @ W + c).sum(dim=-1)
        return term1 - term2

    def _gibbs_step(self, v: torch.Tensor, T: float, rng: torch.Generator):
        # Sample h | v
        p_h = torch.sigmoid((v @ self.W + self.c) / T)
        h = torch.bernoulli(p_h, generator=rng)
        # Sample v | h
        p_v = torch.sigmoid((h @ self.W.t() + self.b) / T)
        v_next = torch.bernoulli(p_v, generator=rng)
        return v_next

    def log_score(self, v: torch.Tensor) -> torch.Tensor:
        # log(psi) = -0.5 * FreeEnergy
        fe = self._free_energy(v, self.W, self.b, self.c)
        return -0.5 * fe / self.T

    def forward(self, batch: Tuple[torch.Tensor, ...], aux_vars: Dict[str, Any]):
        values, _, _ = batch
        v_data = values.to(dtype=self.W.dtype, device=self.W.device)
        rng = aux_vars.get("rng", torch.Generator(device="cpu"))

        # === CRITICAL FIX IS HERE ===
        # Standard CD-k: Initialize chain at DATA, not noise.
        # This ensures the chain explores the neighborhood of the data distribution.
        v_model = v_data.clone()

        # Run Gibbs Sampling
        for _ in range(self.k):
            v_model = self._gibbs_step(v_model, self.T, rng)

        v_model = v_model.detach()

        # Loss = FE(data) - FE(model)
        fe_data = self._free_energy(v_data, self.W, self.b, self.c)
        fe_model = self._free_energy(v_model, self.W, self.b, self.c)

        loss = (fe_data - fe_model).mean()

        # Metrics
        # Flip rate close to 0 means model stays at data (overfitting/frozen)
        # Flip rate close to 0.5 means model is random noise
        # Healthy flip rate is usually 0.05 - 0.30 for CD training
        n_flips = (v_data != v_model).float().sum()
        flip_rate = n_flips / v_data.numel()

        return loss, {
            "flip_rate": flip_rate,
            "fe_mean": fe_data.mean().detach()
        }

    @torch.no_grad()
    def generate(self, n_samples: int, T_schedule: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
        device = next(self.parameters()).device

        # Start from noise for generation (Evaluation only)
        probs = torch.full((n_samples, self.num_visible), 0.5, device=device, dtype=torch.float32)
        v = torch.bernoulli(probs, generator=rng)

        if T_schedule.dim() == 0:
            T_schedule = T_schedule.view(1)

        for i in range(int(T_schedule.shape[0])):
            v = self._gibbs_step(v, float(T_schedule[i]), rng)
        return v


# === TRAINING LOOP ===

def train_step(model: nn.Module, optimizer: torch.optim.Optimizer,
               batch: Tuple[torch.Tensor, ...], aux_vars: Dict[str, Any]):
    optimizer.zero_grad(set_to_none=True)
    loss, aux = model(batch, aux_vars)
    loss.backward()
    optimizer.step()
    return loss.detach(), aux

def get_sigmoid_curve(high, low, steps, falloff, center_step=None):
    if center_step is None: center_step = steps / 2.0
    def curve_fn(step: int) -> float:
        s = min(step, steps)
        return float(low + (high - low) / (1.0 + math.exp(falloff * (s - center_step))))
    return curve_fn

def train(model: nn.Module, optimizer: torch.optim.Optimizer, loader, num_epochs: int,
          rng: torch.Generator, lr_schedule_fn: Callable[[int], float]):

    global_step = 0
    print(f"{'Epoch':<6} | {'Loss':<9} | {'FE Data':<9} | {'FlipRate':<9} | {'LR':<9}")
    print("-" * 55)

    for epoch in range(num_epochs):
        tot_loss = 0.0
        avg_fe = 0.0
        avg_flip = 0.0

        for batch in loader:
            lr = float(lr_schedule_fn(global_step))
            for g in optimizer.param_groups: g["lr"] = lr

            aux_vars = { "rng": rng }
            loss, aux_out = train_step(model, optimizer, batch, aux_vars)

            tot_loss += float(loss)
            avg_fe += float(aux_out.get("fe_mean", 0.0))
            avg_flip += float(aux_out.get("flip_rate", 0.0))
            global_step += 1

        n = len(loader)
        print(f"{epoch + 1:<6} | {tot_loss/n:+.4f}   | {avg_fe/n:+.4f}    | "
              f"{avg_flip/n:.4f}    | {lr:.5f}")

    return model

# === ENTROPY CALCULATION ===

def compute_renyi_entropy_from_samples(
        samples: torch.Tensor,
        subsystem_size: int,
        score_fn: Callable[[torch.Tensor], torch.Tensor]
) -> float:
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
    ratios = torch.exp(log_ratio)

    swap_expectation = ratios.mean().item()

    if swap_expectation <= 1e-12: return 0.0
    return -math.log(swap_expectation)


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    # 1. CONFIGURATION
    CHAIN_LENGTH = 16
    FILE_SAMPLES = 5_000_000
    TRAIN_SAMPLES = 100_000
    TARGET_DELTA = 1.00

    # 2. DATA LOADING
    file_name = f"xxz_{CHAIN_LENGTH}_delta{TARGET_DELTA:.2f}_{FILE_SAMPLES}.npz"
    file_path = data_dir / file_name

    if not file_path.exists():
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

    print(f"Loading {file_path}...")
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[TRAIN_SAMPLES])

    # 3. SETUP
    batch_size = 1024
    num_visible = CHAIN_LENGTH
    num_hidden = 32
    num_epochs = 100
    k_steps = 15     # Short steps for training (CD-k)
    init_lr = 1e-2
    final_lr = 1e-3

    SEED = 42
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)

    loader = MeasurementLoader(dataset=ds, batch_size=batch_size, shuffle=True, drop_last=False, rng=rng)

    # 4. INIT MODEL
    model = RBM(num_visible=num_visible, num_hidden=num_hidden, k=k_steps, T=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    schedule_steps = num_epochs * len(loader)
    lr_schedule_fn = get_sigmoid_curve(high=init_lr, low=final_lr, steps=schedule_steps, falloff=0.0005)

    # 5. TRAIN
    print("Starting Training (Standard CD-k)...")
    model = train(model, optimizer, loader, num_epochs, rng, lr_schedule_fn)

    # 6. EVALUATION
    print("\nEvaluating Rényi Entropy...")
    eval_samples = 100_000

    # CRITICAL: Use longer chain for evaluation generation to ensure we reach equilibrium
    T_schedule = torch.ones(200, device=next(model.parameters()).device)
    rng_eval = torch.Generator().manual_seed(123)

    with torch.no_grad():
        samples = model.generate(eval_samples, T_schedule, rng_eval)

    l_axis_model = list(range(1, CHAIN_LENGTH // 2 + 1))
    s2_curve_model = []

    scorer = lambda v: model.log_score(v)

    for l in l_axis_model:
        val = compute_renyi_entropy_from_samples(samples, l, scorer)
        s2_curve_model.append(val)
        print(f"l={l}: S2={val:.4f}")

    # 7. PLOTTING
    print("\nPlotting...")
    ref_file = Path(f"xxz_{CHAIN_LENGTH}_entropy_ref.csv")

    plt.figure(figsize=(8, 6))

    if ref_file.exists():
        df_ref = pd.read_csv(ref_file)
        l_cols = sorted([c for c in df_ref.columns if c.startswith("l")], key=lambda s: int(s[1:]))
        l_axis_ref = [int(c[1:]) for c in l_cols]
        mask = np.isclose(df_ref["delta"], TARGET_DELTA)
        if mask.any():
            s2_ref = df_ref.loc[mask].iloc[0][l_cols].to_numpy()
            plt.plot(l_axis_ref, s2_ref, 'k--', label=f'Reference $\Delta={TARGET_DELTA}$')

    plt.plot(l_axis_model, s2_curve_model, 'ro-', label=f'RBM $\Delta={TARGET_DELTA}$')

    plt.title(f"Entanglement Entropy (N={CHAIN_LENGTH})")
    plt.xlabel(r"Subsystem size $\ell$")
    plt.ylabel(r"$S_2$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()