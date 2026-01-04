import sys
from pathlib import Path
from typing import Callable

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_handling import load_measurements_npz, MeasurementDataset


#### SCORING FUNCTION (EMPIRICAL AMPLITUDE ESTIMATOR) ####

class DatasetLogScore:
    def __init__(self, samples: torch.Tensor):
        self.device = samples.device
        self.num_qubits = samples.shape[1]
        self.num_states = 2 ** self.num_qubits

        powers = 2 ** torch.arange(self.num_qubits, device=self.device)
        indices = (samples.long() * powers).sum(dim=1)

        counts = torch.bincount(indices, minlength=self.num_states).float()
        self.log_counts = torch.log(counts + 1e-10)
        self._powers = powers

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        indices = (states.long() * self._powers).sum(dim=1)
        return 0.5 * self.log_counts[indices]


#### RENYI ENTROPY ESTIMATOR (SWAP OPERATOR) ####

def compute_renyi_entropy(samples: torch.Tensor, subs_size: int,
                          log_score_fn: Callable[[torch.Tensor], torch.Tensor]) -> float:

    n_samples = samples.shape[0]
    half = n_samples // 2
    if half == 0:
        return 0.0

    # split up samples into two replicas and calculate reference scores
    ref_1 = samples[:half]
    ref_2 = samples[half:2 * half]

    ref_1_score = log_score_fn(ref_1)
    ref_2_score = log_score_fn(ref_2)

    # slice replicas into head and tail and swap them between replicas
    slice_idx = torch.arange(subs_size, samples.shape[1], device=samples.device)

    swap_1 = ref_1.clone()
    swap_1[:, slice_idx] = ref_2[:, slice_idx]
    swap_2 = ref_2.clone()
    swap_2[:, slice_idx] = ref_1[:, slice_idx]

    swap_1_score = log_score_fn(swap_1)
    swap_2_score = log_score_fn(swap_2)

    log_swap_ratio = swap_1_score + swap_2_score - ref_1_score - ref_2_score
    swap_ratios = torch.exp(log_swap_ratio)
    swap_exp = swap_ratios.mean().item()    # mean over (half) batch

    if swap_exp <= 1e-12:
        return 0.0

    # finally calculate the 2nd Renyi entropy according to S2 = -log(<Swap>)
    renyi_entropy = -float(np.log(swap_exp))
    return renyi_entropy


#### MAIN EXECUTION ####

if __name__ == "__main__":

    chain_length = 16
    h_values = [0.8, 1.0, 1.2]

    file_samples = 5_000_000
    eval_samples = 5_000_000  # the swap test really needs as many samples as possible

    meas_dir = Path("measurements")
    ref_file = Path(f"tfim_{chain_length}_entropy_ref.csv")

    print("=== RENYI ENTROPY RECONSTRUCTION FROM DATA =====================")
    print(f"System: TFIM N={chain_length}")
    print(f"Target h values: {h_values}")
    print("================================================================")

    if not ref_file.exists():
        print(f"Reference file {ref_file} not found.")
        sys.exit(1)

    ref_df = pd.read_csv(ref_file)
    l_cols = sorted([c for c in ref_df.columns if c.startswith("l")], key=lambda s: int(s[1:]))
    l_axis_ref = [int(c[1:]) for c in l_cols]
    max_l = len(l_axis_ref)

    # iterate over h values and compute entanglement entropy curves from data

    results_data = {}

    for h in h_values:
        print(f"\nProcessing h = {h}...")

        filename = f"tfim_{chain_length}_h{h:.2f}_{file_samples}.npz"
        filepath = meas_dir / filename

        if not filepath.exists():
            print(f"  [Error] File not found: {filepath}")
            continue

        ds = MeasurementDataset([filepath], load_fn=load_measurements_npz,
                                system_param_keys=["h"], samples_per_file=[eval_samples])

        samples = torch.as_tensor(ds.values, dtype=torch.uint8)
        scorer = DatasetLogScore(samples)

        curve = []
        for l in range(1, max_l + 1):
            s2 = compute_renyi_entropy(samples, l, scorer)
            curve.append(s2)
            sys.stdout.write(f"\r  l={l}: S2={s2:.4f}")
            sys.stdout.flush()

        results_data[h] = curve
        print(" -> Done.")

    # PLOTTING

    print("\nPlotting...")

    fig, axes = plt.subplots(1, len(h_values), figsize=(5 * len(h_values), 5), sharey=True, dpi=100)

    if len(h_values) == 1: axes = [axes]

    styles = {
        0.8: {'color': '#5D8AA8', 'label': r'h=0.8'},
        1.0: {'color': '#A0522D', 'label': r'h=1.0'},
        1.2: {'color': '#556B2F', 'label': r'h=1.2'},
    }

    for ax, h in zip(axes, h_values):
        if h not in results_data:
            ax.set_title(f"$h={h}$ (No Data)")
            continue

        color = styles.get(h, {'color': 'black'})['color']
        y_data = results_data[h]
        x_vals = l_axis_ref[:len(y_data)]

        mask = np.isclose(ref_df["h"].values, h, atol=1e-8)
        if mask.any():
            y_ref = ref_df.loc[mask].iloc[0][l_cols].to_numpy()
            ax.plot(x_vals, y_ref[:len(y_data)], ':', color='gray', linewidth=2.0,
                    label=r'$S_2^{\mathrm{reference}}$', zorder=1)

        ax.plot(x_vals, y_data, 'o', markersize=8, color=color, markeredgecolor='black',
                alpha=0.9, label=r'$S_2^{\mathrm{data}}$', zorder=2)

        ax.set_title(f"$h = {h}$", fontsize=14)
        ax.set_xlabel(r"Subsystem size $\ell$", fontsize=12)
        ax.set_xticks(x_vals)
        ax.grid(True, alpha=0.3)

        if ax == axes[0]:
            ax.set_ylabel(r"Rényi Entropy $S_2$", fontsize=14)
            ax.legend(loc='lower right', fontsize=10)

    plt.suptitle(f"Entanglement Entropy ($N={chain_length}$)", fontsize=16)
    plt.tight_layout()
    plt.show()