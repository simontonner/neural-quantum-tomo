#!/usr/bin/env python3
"""
Steelman single (symmetric) RBM test.

What it does:
1) Runs two baselines on each h in H_POINTS:
   - "paper_simple": constant LR=1e-2, wd=0
   - "hyper_style":  sigmoid LR schedule 1e-2 -> 1e-4, noise_frac=0.1, wd=0
2) Identifies the WORST baseline point h_worst (lowest overlap).
3) Tunes SINGLE-RBM training on h_worst over:
   - LR schedules
   - init schemes / init std
   - weight decay
   - noise_frac
4) Reports best config (best-of-N restarts, by default),
   then re-evaluates that best config on all H_POINTS.

Run:
  python steelman_single_rbm.py

Optional:
  python steelman_single_rbm.py --device cuda --random-configs 80 --restarts 3 --save-csv
"""

import sys
import math
import time
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- IMPORTS FROM YOUR PROJECT ---
sys.path.append(str(Path("..").resolve()))
from data_handling import load_measurements_npz, MeasurementDataset, MeasurementLoader
from wavefunction_overlap import generate_basis_states, calculate_exact_overlap, load_gt_wavefunction


# -----------------------------
# Config defaults
# -----------------------------
DEFAULT_H_POINTS = [2.50, 3.00, 3.50]
DEFAULT_SIDE_LENGTH = 4
DEFAULT_N_SAMPLES = 2000

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 1024
DEFAULT_HIDDEN_DIM = 64
DEFAULT_K_STEPS = 10

DEFAULT_SEED = 42

DEFAULT_DATA_DIR = Path("measurements")
DEFAULT_STATE_DIR = Path("state_vectors")
DEFAULT_OUT_DIR = Path("models")


# -----------------------------
# Symmetric RBM
# -----------------------------
class SymmetricRBM(nn.Module):
    def __init__(self, num_v: int, num_h: int, k: int = 10, T: float = 1.0):
        super().__init__()
        self.num_v = num_v
        self.num_h = num_h
        self.k = k
        self.T = T

        self.W = nn.Parameter(torch.empty(num_v, num_h))
        self.b = nn.Parameter(torch.zeros(num_v))
        self.c = nn.Parameter(torch.zeros(num_h))

    def initialize_weights(self, mode: str = "normal", std: float = 0.01, scale: float = 1.0):
        nn.init.constant_(self.b, 0.0)
        nn.init.constant_(self.c, 0.0)

        mode = mode.lower()
        if mode == "normal":
            nn.init.normal_(self.W, std=std)
        elif mode == "xavier_normal":
            nn.init.xavier_normal_(self.W)
            with torch.no_grad():
                self.W.mul_(scale)
        elif mode == "xavier_uniform":
            nn.init.xavier_uniform_(self.W)
            with torch.no_grad():
                self.W.mul_(scale)
        elif mode == "kaiming_normal":
            nn.init.kaiming_normal_(self.W, nonlinearity="sigmoid")
            with torch.no_grad():
                self.W.mul_(scale)
        else:
            raise ValueError(f"Unknown init mode: {mode}")

    def _free_energy(self, v: torch.Tensor) -> torch.Tensor:
        b, c = self.b.unsqueeze(0), self.c.unsqueeze(0)
        vW = v @ self.W
        W_colsum = self.W.sum(dim=0)

        preact_normal = vW + c
        preact_flipped = (W_colsum.unsqueeze(0) - vW) + c

        negF = torch.stack(
            [
                (v * b).sum(-1) - F.softplus(preact_normal).sum(-1),
                ((1.0 - v) * b).sum(-1) - F.softplus(preact_flipped).sum(-1),
                ],
            dim=-1,
        )
        return -self.T * torch.logsumexp(negF / self.T, dim=-1)

    def forward(self, batch, aux_vars: Dict[str, Any]):
        rng = aux_vars.get("rng", None)
        noise_frac: float = float(aux_vars.get("noise_frac", 0.0))

        v_data = batch[0].to(device=self.W.device, dtype=self.W.dtype)

        # Optionally add bit-flip noise to the chain start (mixing trick).
        v_model = v_data.clone()
        if noise_frac > 0.0:
            # NOTE: torch.rand_like(..., generator=...) not supported in some torch versions
            mask = torch.rand(
                v_model.shape,
                device=v_model.device,
                dtype=v_model.dtype,
                generator=rng,
            ) < noise_frac
            v_model = torch.where(mask, 1.0 - v_model, v_model)

        B = v_data.size(0)

        # Latent branch selector u in {0,1}
        u = torch.bernoulli(
            torch.full((B, 1), 0.5, device=self.W.device, dtype=self.W.dtype),
            generator=rng,
        )

        b, c = self.b.unsqueeze(0), self.c.unsqueeze(0)
        for _ in range(self.k):
            v_branch = u * v_model + (1.0 - u) * (1.0 - v_model)
            h = torch.bernoulli(torch.sigmoid(v_branch @ self.W + c), generator=rng)
            a = h @ self.W.t()

            vb = (v_model * b).sum(dim=-1)
            va = (v_model * a).sum(dim=-1)

            dE = (-b.sum(dim=-1) - a.sum(dim=-1) + 2.0 * vb + 2.0 * va)
            u = torch.bernoulli(torch.sigmoid(dE), generator=rng).unsqueeze(-1)

            v_new = torch.bernoulli(torch.sigmoid(a + b), generator=rng)
            v_model = u * v_new + (1.0 - u) * (1.0 - v_new)

        loss = self._free_energy(v_data).mean() - self._free_energy(v_model.detach()).mean()
        return loss

    def log_score(self, v, cond=None):
        return -0.5 * self._free_energy(v) / self.T


# -----------------------------
# LR schedules
# -----------------------------
def lr_constant(init_lr: float, final_lr: float, total_steps: int, falloff: float = 0.005):
    def f(step: int) -> float:
        return init_lr
    return f

def lr_linear(init_lr: float, final_lr: float, total_steps: int, falloff: float = 0.005):
    def f(step: int) -> float:
        t = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
        return (1.0 - t) * init_lr + t * final_lr
    return f

def lr_cosine(init_lr: float, final_lr: float, total_steps: int, falloff: float = 0.005):
    def f(step: int) -> float:
        t = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
        c = 0.5 * (1.0 + math.cos(math.pi * t))
        return final_lr + (init_lr - final_lr) * c
    return f

def lr_sigmoid(init_lr: float, final_lr: float, total_steps: int, falloff: float = 0.005):
    center = (total_steps - 1) / 2.0
    def f(step: int) -> float:
        x = step - center
        s = 1.0 / (1.0 + math.exp(falloff * x))
        return final_lr + (init_lr - final_lr) * s
    return f

SCHEDULES = {
    "constant": lr_constant,
    "linear": lr_linear,
    "cosine": lr_cosine,
    "sigmoid": lr_sigmoid,
}


# -----------------------------
# Utils
# -----------------------------
@torch.no_grad()
def _set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------
# Trial config
# -----------------------------
@dataclass(frozen=True)
class TrialConfig:
    init_lr: float
    final_lr: float
    weight_decay: float

    schedule: str
    falloff: float

    noise_frac: float

    init_mode: str
    init_std: float
    init_scale: float


def build_default_grid() -> Dict[str, List[Any]]:
    return {
        "init_lr":      [1e-3, 3e-3, 5e-3, 1e-2, 2e-2],
        "final_lr":     [1e-4, 3e-4, 1e-3],
        "weight_decay": [0.0, 1e-4, 1e-3, 1e-2],
        "schedule":     ["constant", "cosine", "sigmoid", "linear"],
        "falloff":      [0.002, 0.005, 0.01],
        "noise_frac":   [0.0, 0.05, 0.1, 0.2],
        "init_mode":    ["normal", "xavier_normal", "xavier_uniform"],
        "init_std":     [0.005, 0.01, 0.02, 0.05],
        "init_scale":   [0.5, 1.0, 2.0],
    }


def sample_configs(grid: Dict[str, List[Any]], n: int, seed: int) -> List[TrialConfig]:
    rng = random.Random(seed)
    keys = list(grid.keys())
    out: List[TrialConfig] = []
    for _ in range(n):
        cfg = {k: rng.choice(grid[k]) for k in keys}
        out.append(TrialConfig(**cfg))
    return out


# -----------------------------
# Training / eval
# -----------------------------
def prepare_dataset_for_h(
        h_val: float,
        side_length: int,
        n_samples: int,
        data_dir: Path,
) -> Optional[MeasurementDataset]:
    file_path = data_dir / f"tfim_{side_length}x{side_length}_h{h_val:.2f}_20000.npz"
    if not file_path.exists():
        return None
    dataset = MeasurementDataset([file_path], load_measurements_npz, ["h"], [n_samples])
    return dataset


def train_one_model(
        dataset: MeasurementDataset,
        loader: MeasurementLoader,
        device: str,
        cfg: TrialConfig,
        epochs: int,
        hidden_dim: int,
        k_steps: int,
        seed: int,
) -> SymmetricRBM:
    set_all_seeds(seed)
    torch_device = torch.device(device)
    trng = torch.Generator(device=torch_device).manual_seed(seed)

    model = SymmetricRBM(dataset.num_qubits, hidden_dim, k_steps).to(device)

    # init
    if cfg.init_mode.lower() == "normal":
        model.initialize_weights(mode="normal", std=float(cfg.init_std), scale=1.0)
    else:
        model.initialize_weights(mode=cfg.init_mode, std=float(cfg.init_std), scale=float(cfg.init_scale))

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.init_lr), weight_decay=float(cfg.weight_decay))

    steps_per_epoch = len(loader)
    total_steps = max(1, epochs * steps_per_epoch)
    sched_name = cfg.schedule.lower()
    if sched_name not in SCHEDULES:
        raise ValueError(f"Unknown schedule '{cfg.schedule}', allowed: {list(SCHEDULES.keys())}")
    schedule_fn = SCHEDULES[sched_name](float(cfg.init_lr), float(cfg.final_lr), total_steps, float(cfg.falloff))

    model.train()
    step = 0
    for _epoch in range(epochs):
        for batch in loader:
            lr_now = schedule_fn(step)
            _set_lr(optimizer, lr_now)

            optimizer.zero_grad(set_to_none=True)
            loss = model(batch, {"rng": trng, "noise_frac": float(cfg.noise_frac)})
            loss.backward()
            optimizer.step()
            step += 1

    return model


def eval_overlap_for_h(
        h_val: float,
        model: SymmetricRBM,
        device: str,
        side_length: int,
        state_dir: Path,
) -> float:
    gt_path = state_dir / f"tfim_{side_length}x{side_length}_h{h_val:.2f}.npz"
    psi_true = load_gt_wavefunction(gt_path, device)
    basis_states = generate_basis_states(side_length**2, device)
    model.eval()
    ov = calculate_exact_overlap(model, h_val, psi_true, basis_states)
    return float(ov)


# -----------------------------
# Main
# -----------------------------
def run():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--side-length", type=int, default=DEFAULT_SIDE_LENGTH)
    p.add_argument("--h-points", type=float, nargs="+", default=DEFAULT_H_POINTS)
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)

    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    p.add_argument("--k-steps", type=int, default=DEFAULT_K_STEPS)

    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--restarts", type=int, default=2, help="best-of-N restarts per config (steelman)")
    p.add_argument("--random-configs", type=int, default=60, help="number of random configs to try on worst point")
    p.add_argument("--save-csv", action="store_true", help="save results CSV")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    p.add_argument("--state-dir", type=str, default=str(DEFAULT_STATE_DIR))

    args = p.parse_args()

    device = args.device
    side_length = args.side_length
    h_points = list(args.h_points)
    n_samples = args.n_samples

    epochs = args.epochs
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    k_steps = args.k_steps

    seed = args.seed
    restarts = max(1, args.restarts)
    n_random = max(1, args.random_configs)

    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    state_dir = Path(args.state_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n### SINGLE RBM STEELMAN (worst-point tuning) ###")
    print(f"device={device} | side={side_length} | N={n_samples} | epochs={epochs} | bs={batch_size} | hid={hidden_dim} | k={k_steps}")
    print(f"h_points={h_points}")
    print("-" * 80)

    # Baselines
    baseline_paper_simple = TrialConfig(
        init_lr=1e-2, final_lr=1e-2, weight_decay=0.0,
        schedule="constant", falloff=0.005,
        noise_frac=0.0,
        init_mode="normal", init_std=0.01, init_scale=1.0,
    )
    baseline_hyper_style = TrialConfig(
        init_lr=1e-2, final_lr=1e-4, weight_decay=0.0,
        schedule="sigmoid", falloff=0.005,
        noise_frac=0.1,
        init_mode="normal", init_std=0.01, init_scale=1.0,
    )

    def run_cfg_on_h(h_val: float, cfg: TrialConfig, tag: str) -> float:
        dataset = prepare_dataset_for_h(h_val, side_length, n_samples, data_dir)
        if dataset is None:
            print(f"[{tag}] h={h_val:.2f} SKIP (measurement file missing)")
            return float("nan")
        gt_path = state_dir / f"tfim_{side_length}x{side_length}_h{h_val:.2f}.npz"
        if not gt_path.exists():
            print(f"[{tag}] h={h_val:.2f} SKIP (gt state missing)")
            return float("nan")

        set_all_seeds(seed)
        rng = torch.Generator(device=torch.device(device)).manual_seed(seed)
        loader = MeasurementLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, rng=rng)

        model = train_one_model(
            dataset=dataset,
            loader=loader,
            device=device,
            cfg=cfg,
            epochs=epochs,
            hidden_dim=hidden_dim,
            k_steps=k_steps,
            seed=seed,
        )
        ov = eval_overlap_for_h(h_val, model, device, side_length, state_dir)
        return ov

    # 1) Baseline scan to find worst
    baseline_rows = []
    for h in h_points:
        ov_simple = run_cfg_on_h(h, baseline_paper_simple, "paper_simple")
        ov_hstyle = run_cfg_on_h(h, baseline_hyper_style, "hyper_style")
        print(f"h={h:.2f} | baseline_simple={ov_simple:.6f} | baseline_hyper_style={ov_hstyle:.6f}")
        baseline_rows.append({"h": h, "baseline_simple": ov_simple, "baseline_hyper_style": ov_hstyle})

    df_base = pd.DataFrame(baseline_rows).dropna()
    if df_base.empty:
        print("\nNo points could be evaluated (missing files).")
        return

    df_base["baseline_best"] = df_base[["baseline_simple", "baseline_hyper_style"]].max(axis=1)
    worst_idx = df_base["baseline_best"].idxmin()
    h_worst = float(df_base.loc[worst_idx, "h"])
    worst_baseline = float(df_base.loc[worst_idx, "baseline_best"])

    print("-" * 80)
    print(f"WORST POINT (by best baseline): h_worst={h_worst:.2f} with overlap={worst_baseline:.6f}")
    print("-" * 80)

    # 2) Tune on worst point
    dataset_worst = prepare_dataset_for_h(h_worst, side_length, n_samples, data_dir)
    gt_path_worst = state_dir / f"tfim_{side_length}x{side_length}_h{h_worst:.2f}.npz"
    if dataset_worst is None or not gt_path_worst.exists():
        print("Worst point missing files - abort.")
        return

    grid = build_default_grid()
    cfgs = sample_configs(grid, n=n_random, seed=seed + 1337)
    cfgs = [baseline_paper_simple, baseline_hyper_style] + cfgs

    trial_rows: List[Dict[str, Any]] = []
    best_score = -1.0
    best_cfg: Optional[TrialConfig] = None
    best_ov: float = -1.0

    for i, cfg in enumerate(cfgs, start=1):
        ovs = []
        for r in range(restarts):
            run_seed = seed + 1000 * r + i
            set_all_seeds(run_seed)

            rng = torch.Generator(device=torch.device(device)).manual_seed(run_seed)
            loader = MeasurementLoader(dataset_worst, batch_size=batch_size, shuffle=True, drop_last=False, rng=rng)

            model = train_one_model(
                dataset=dataset_worst,
                loader=loader,
                device=device,
                cfg=cfg,
                epochs=epochs,
                hidden_dim=hidden_dim,
                k_steps=k_steps,
                seed=run_seed,
            )
            ov = eval_overlap_for_h(h_worst, model, device, side_length, state_dir)
            ovs.append(ov)

        ov_best = float(np.max(ovs))  # steelman: best restart
        row = asdict(cfg)
        row.update(
            {
                "h": h_worst,
                "trial": i,
                "ov_best_of_restarts": ov_best,
                "ov_mean": float(np.mean(ovs)),
                "ov_std": float(np.std(ovs)),
            }
        )
        trial_rows.append(row)

        if ov_best > best_score:
            best_score = ov_best
            best_cfg = cfg
            best_ov = ov_best

        if i % max(1, len(cfgs) // 10) == 0 or i <= 3:
            print(f"[{i:>4}/{len(cfgs)}] best_of_{restarts}={ov_best:.6f} | cfg={cfg}")

    assert best_cfg is not None
    print("-" * 80)
    print("BEST CONFIG ON WORST POINT")
    print(f"h={h_worst:.2f} | overlap(best-of-{restarts})={best_ov:.6f} | baseline(best)={worst_baseline:.6f} | gain={best_ov - worst_baseline:+.6f}")
    print(best_cfg)
    print("-" * 80)

    # 3) Re-evaluate best config across all points (one canonical restart per h)
    final_rows = []
    for h in h_points:
        dataset_h = prepare_dataset_for_h(h, side_length, n_samples, data_dir)
        gt_path_h = state_dir / f"tfim_{side_length}x{side_length}_h{h:.2f}.npz"
        if dataset_h is None or not gt_path_h.exists():
            final_rows.append({"h": h, "best_cfg_overlap": float("nan")})
            continue

        set_all_seeds(seed)
        rng = torch.Generator(device=torch.device(device)).manual_seed(seed)
        loader = MeasurementLoader(dataset_h, batch_size=batch_size, shuffle=True, drop_last=False, rng=rng)

        model = train_one_model(
            dataset=dataset_h,
            loader=loader,
            device=device,
            cfg=best_cfg,
            epochs=epochs,
            hidden_dim=hidden_dim,
            k_steps=k_steps,
            seed=seed,
        )
        ov = eval_overlap_for_h(h, model, device, side_length, state_dir)
        final_rows.append({"h": h, "best_cfg_overlap": ov})

    df_final = pd.DataFrame(final_rows)
    df_report = df_base.merge(df_final, on="h", how="left")
    df_report["best_baseline"] = df_report[["baseline_simple", "baseline_hyper_style"]].max(axis=1)
    df_report["gain_vs_best_baseline"] = df_report["best_cfg_overlap"] - df_report["best_baseline"]

    print("\nFINAL REPORT (best cfg trained per h)")
    print(df_report.to_string(index=False, float_format="%.6f"))

    worst_baseline_all = float(np.nanmin(df_report["best_baseline"].values))
    worst_bestcfg_all = float(np.nanmin(df_report["best_cfg_overlap"].values))
    print("-" * 80)
    print(f"Worst-case baseline (across points): {worst_baseline_all:.6f}")
    print(f"Worst-case best-cfg (across points):  {worst_bestcfg_all:.6f}")
    print(f"Worst-case gain:                     {worst_bestcfg_all - worst_baseline_all:+.6f}")

    if args.save_csv:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"steelman_single_rbm_hworst{h_worst:.2f}_{ts}.csv"
        out_path2 = out_dir / f"steelman_single_rbm_report_{ts}.csv"
        pd.DataFrame(trial_rows).to_csv(out_path, index=False)
        df_report.to_csv(out_path2, index=False)
        print(f"\nSaved:\n  {out_path}\n  {out_path2}")


if __name__ == "__main__":
    run()
