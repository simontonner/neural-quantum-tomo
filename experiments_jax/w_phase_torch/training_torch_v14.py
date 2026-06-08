#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyperparameter search for pooled-CD ComplexRBM tomography.

Search target:
- maximize the highest fidelity reached at any epoch

Strategy:
1. coarse screen over batch size and constant learning rate
2. local refinement around the best coarse settings
3. final evaluation of a small finalist set with 5 seeds and 100 epochs

Notes:
- this script assumes the base training code lives in a sibling file
  called `project_style_tomography_clean_fixed.py`
- it reuses that code's train_experiment_model(cfg) entrypoint
- schedules are forced to constant here
"""

from __future__ import annotations

import copy
import gc
import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch


BASE_MODULE_PATH = Path(__file__).with_name("training_torch_v13.py")
RESULTS_DIR = Path(__file__).with_name("hp_search_results")
RESULTS_DIR.mkdir(exist_ok=True)

COARSE_BATCH_SIZES = [32, 64, 128, 256, 512]
COARSE_LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

SEEDS_STAGE_1 = [11, 17]
SEEDS_STAGE_2 = [11, 17, 23]
SEEDS_STAGE_3 = [11, 17, 23, 29, 31]

EPOCHS_STAGE_1 = 30
EPOCHS_STAGE_2 = 60
EPOCHS_STAGE_3 = 100

TOPK_STAGE_1 = 6
TOPK_STAGE_2 = 3


@dataclass
class TrialSummary:
    stage: str
    batch_size: int
    learning_rate: float
    seed: int
    epochs: int
    best_fidelity: float
    best_epoch: int
    final_fidelity: float
    final_kl: float


def load_base_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("tomography_base", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_constant_cfg(base_cfg: Dict, *, batch_size: int, learning_rate: float, seed: int, epochs: int) -> Dict:
    cfg = copy.deepcopy(base_cfg)

    cfg["training"]["epochs"] = int(epochs)
    cfg["training"]["batch_size"] = int(batch_size)
    cfg["training"]["neg_batch_size"] = int(batch_size)
    cfg["training"]["log_every"] = 1
    cfg["training"]["seed"] = int(seed)

    cfg["optimizer"]["am_cls"] = torch.optim.Adam
    cfg["optimizer"]["ph_cls"] = torch.optim.Adam
    cfg["optimizer"]["am_kwargs"] = {}
    cfg["optimizer"]["ph_kwargs"] = {}

    cfg["schedule"]["mode"] = "constant"
    cfg["schedule"]["am_init_lr"] = float(learning_rate)
    cfg["schedule"]["am_final_lr"] = float(learning_rate)
    cfg["schedule"]["ph_init_lr"] = float(learning_rate)
    cfg["schedule"]["ph_final_lr"] = float(learning_rate)

    return cfg


def summarize_history(stage: str, batch_size: int, learning_rate: float, seed: int, epochs: int, history: Dict) -> TrialSummary:
    fidelities = list(history["Fidelity"])
    epochs_logged = list(history["epoch"])
    kls = list(history["KL"])

    best_idx = int(np.argmax(fidelities))
    best_fid = float(fidelities[best_idx])
    best_epoch = int(epochs_logged[best_idx])
    final_fid = float(fidelities[-1])
    final_kl = float(kls[-1])

    return TrialSummary(
        stage=stage,
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        seed=int(seed),
        epochs=int(epochs),
        best_fidelity=best_fid,
        best_epoch=best_epoch,
        final_fidelity=final_fid,
        final_kl=final_kl,
    )


def cleanup_after_run(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_single_trial(module, stage: str, batch_size: int, learning_rate: float, seed: int, epochs: int) -> TrialSummary:
    cfg = build_constant_cfg(
        module.CONFIG,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        epochs=epochs,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    model, history, dataset, basis_states = module.train_experiment_model(cfg)
    summary = summarize_history(stage, batch_size, learning_rate, seed, epochs, history)

    # keep peak search lightweight in memory
    del dataset, basis_states, history
    cleanup_after_run(model)
    return summary


def aggregate_summaries(summaries: Iterable[TrialSummary]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(s) for s in summaries])
    grouped = (
        df.groupby(["stage", "batch_size", "learning_rate"], as_index=False)
        .agg(
            mean_peak_fidelity=("best_fidelity", "mean"),
            std_peak_fidelity=("best_fidelity", "std"),
            max_peak_fidelity=("best_fidelity", "max"),
            mean_best_epoch=("best_epoch", "mean"),
            mean_final_fidelity=("final_fidelity", "mean"),
            mean_final_kl=("final_kl", "mean"),
            num_seeds=("seed", "count"),
        )
        .sort_values(
            ["mean_peak_fidelity", "max_peak_fidelity", "mean_final_fidelity"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    return grouped


def print_stage_table(title: str, grouped_df: pd.DataFrame, topk: int = 10):
    print(f"\n=== {title} ===")
    show = grouped_df.head(topk).copy()
    if show.empty:
        print("(no results)")
        return
    cols = [
        "batch_size",
        "learning_rate",
        "mean_peak_fidelity",
        "max_peak_fidelity",
        "mean_best_epoch",
        "mean_final_fidelity",
        "mean_final_kl",
        "num_seeds",
    ]
    print(show[cols].to_string(index=False))


def lr_neighbors(lr: float) -> List[float]:
    factors = [1 / math.sqrt(10.0), 1.0, math.sqrt(10.0)]
    vals = sorted({round(lr * f, 12) for f in factors if 1e-5 <= lr * f <= 1e-1})
    return vals


def batch_neighbors(batch_size: int) -> List[int]:
    grid = COARSE_BATCH_SIZES
    idx = grid.index(batch_size)
    out = {batch_size}
    if idx > 0:
        out.add(grid[idx - 1])
    if idx < len(grid) - 1:
        out.add(grid[idx + 1])
    return sorted(out)


def build_refinement_grid(coarse_grouped: pd.DataFrame, topk: int) -> List[Tuple[int, float]]:
    finalists = coarse_grouped.head(topk)[["batch_size", "learning_rate"]].itertuples(index=False)
    candidates = set()
    for row in finalists:
        for bs in batch_neighbors(int(row.batch_size)):
            for lr in lr_neighbors(float(row.learning_rate)):
                candidates.add((bs, lr))
    return sorted(candidates, key=lambda x: (x[0], x[1]))


def run_stage(module, stage_name: str, candidates: List[Tuple[int, float]], seeds: List[int], epochs: int) -> Tuple[List[TrialSummary], pd.DataFrame]:
    summaries: List[TrialSummary] = []
    total_runs = len(candidates) * len(seeds)
    run_idx = 0

    for batch_size, learning_rate in candidates:
        for seed in seeds:
            run_idx += 1
            print(
                f"[{stage_name}] run {run_idx}/{total_runs} | "
                f"bs={batch_size:<3d} lr={learning_rate:.5g} seed={seed} epochs={epochs}"
            )
            summary = run_single_trial(
                module,
                stage=stage_name,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
                epochs=epochs,
            )
            summaries.append(summary)

            print(
                f"    best_fid={summary.best_fidelity:.6f} at epoch {summary.best_epoch} | "
                f"final_fid={summary.final_fidelity:.6f} | final_kl={summary.final_kl:.6f}"
            )

    grouped = aggregate_summaries(summaries)
    return summaries, grouped


def save_stage_outputs(stage_name: str, summaries: List[TrialSummary], grouped: pd.DataFrame):
    raw_df = pd.DataFrame([asdict(s) for s in summaries])
    raw_path = RESULTS_DIR / f"{stage_name}_raw.csv"
    grouped_path = RESULTS_DIR / f"{stage_name}_grouped.csv"
    raw_df.to_csv(raw_path, index=False)
    grouped.to_csv(grouped_path, index=False)


def main():
    print(f"Loading base module from: {BASE_MODULE_PATH}")
    module = load_base_module(BASE_MODULE_PATH)

    coarse_candidates = [(bs, lr) for bs in COARSE_BATCH_SIZES for lr in COARSE_LRS]

    stage1_summaries, stage1_grouped = run_stage(
        module,
        stage_name="stage1_coarse",
        candidates=coarse_candidates,
        seeds=SEEDS_STAGE_1,
        epochs=EPOCHS_STAGE_1,
    )
    save_stage_outputs("stage1_coarse", stage1_summaries, stage1_grouped)
    print_stage_table("Stage 1 - coarse screen", stage1_grouped, topk=10)

    refine_candidates = build_refinement_grid(stage1_grouped, TOPK_STAGE_1)
    stage2_summaries, stage2_grouped = run_stage(
        module,
        stage_name="stage2_refine",
        candidates=refine_candidates,
        seeds=SEEDS_STAGE_2,
        epochs=EPOCHS_STAGE_2,
    )
    save_stage_outputs("stage2_refine", stage2_summaries, stage2_grouped)
    print_stage_table("Stage 2 - local refinement", stage2_grouped, topk=10)

    final_candidates = [
        (int(row.batch_size), float(row.learning_rate))
        for row in stage2_grouped.head(TOPK_STAGE_2)[["batch_size", "learning_rate"]].itertuples(index=False)
    ]
    stage3_summaries, stage3_grouped = run_stage(
        module,
        stage_name="stage3_final",
        candidates=final_candidates,
        seeds=SEEDS_STAGE_3,
        epochs=EPOCHS_STAGE_3,
    )
    save_stage_outputs("stage3_final", stage3_summaries, stage3_grouped)
    print_stage_table("Stage 3 - final selection", stage3_grouped, topk=TOPK_STAGE_2)

    best_row = stage3_grouped.iloc[0].to_dict()
    best_config = {
        "batch_size": int(best_row["batch_size"]),
        "neg_batch_size": int(best_row["batch_size"]),
        "learning_rate_am": float(best_row["learning_rate"]),
        "learning_rate_ph": float(best_row["learning_rate"]),
        "selection_metric": "mean_peak_fidelity",
        "mean_peak_fidelity": float(best_row["mean_peak_fidelity"]),
        "max_peak_fidelity": float(best_row["max_peak_fidelity"]),
        "mean_best_epoch": float(best_row["mean_best_epoch"]),
        "mean_final_fidelity": float(best_row["mean_final_fidelity"]),
        "mean_final_kl": float(best_row["mean_final_kl"]),
    }

    best_path = RESULTS_DIR / "best_config.json"
    with open(best_path, "w") as f:
        json.dump(best_config, f, indent=2)

    all_raw = pd.concat(
        [
            pd.DataFrame([asdict(s) for s in stage1_summaries]),
            pd.DataFrame([asdict(s) for s in stage2_summaries]),
            pd.DataFrame([asdict(s) for s in stage3_summaries]),
        ],
        ignore_index=True,
    )
    all_raw.to_csv(RESULTS_DIR / "all_trials.csv", index=False)

    print("\n=== Best final setting ===")
    print(json.dumps(best_config, indent=2))
    print(f"\nSaved results in: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
