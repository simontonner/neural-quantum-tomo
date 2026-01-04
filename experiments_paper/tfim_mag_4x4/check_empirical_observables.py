import sys
from pathlib import Path
from typing import Callable

import torch
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_handling import load_measurements_npz, MeasurementDataset



#### SCORING FUNCTION AND OBSERVABLE ESTIMATORS ####

class DatasetLogScore:
    """
    Acts as a statistical stand-in for the free-energy function of an RBM.

    Returns 0.5 * log(count), which corresponds to log(amplitude).
    F_ref - F_flip in RBM terms corresponds to log(amp_ref) - log(amp_flip) here.
    """
    def __init__(self, samples: torch.Tensor):
        self.device = samples.device
        self.num_qubits = samples.shape[1]
        self.num_states = 2 ** self.num_qubits

        # count occurrences by mapping bitstrings to integer indices
        powers = 2 ** torch.arange(self.num_qubits, device=self.device)
        indices = (samples.long() * powers).sum(dim=1)
        counts = torch.bincount(indices, minlength=self.num_states).float()

        # the log-counts are our scoring metric, we add epsilon to avoid log(0)
        self.log_counts = torch.log(counts + 1e-10)
        # we store powers for indexing our samples during scoring
        self._powers = powers

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        indices = (states.long() * self._powers).sum(dim=1)
        # amplitude is like sqrt(count), so log-amp is 0.5 * log(count)
        return 0.5 * self.log_counts[indices]


def calculate_mz(samples: torch.Tensor) -> float:
    spins = 1.0 - 2.0 * samples.float()

    sample_mags = spins.mean(dim=1) # mean magnetization per sample

    # take absolute before averaging over samples
    return sample_mags.abs().mean().item()


def calculate_mx(samples: torch.Tensor, log_score_fn: Callable[[torch.Tensor], torch.Tensor]) -> float:
    # we iterate over sites and apply the ratio trick
    num_qubits = samples.shape[1]

    log_scores_orig = log_score_fn(samples)

    total_mx = 0.0
    for i in range(num_qubits):
        flipped_samples = samples.clone()
        flipped_samples[:, i] = 1 - flipped_samples[:, i]

        log_scores_flip = log_score_fn(flipped_samples)

        log_ratios = log_scores_flip - log_scores_orig
        ratios = torch.exp(log_ratios)

        total_mx += ratios.mean().item()

    return total_mx / num_qubits



if __name__ == "__main__":

    # parameters need to match those used in data generation
    side_length = 4

    h_values = [1.00, 2.00, 2.80, 3.00, 3.30, 3.60, 4.00, 5.00, 6.00, 7.00]
    file_samples = 5_000_000
    eval_samples = 100_000   # this one helps to dial in how many training samples to use for the RBM

    meas_dir = Path("measurements")
    ref_file = Path(f"tfim_{side_length}x{side_length}_magnetizations_ref.csv")

    print("=== CROSSCHECK ANALYSIS ========================================")
    print(f"System: TFIM {side_length}x{side_length}")
    print(f"Estimating Mz and Mx using dataset frequency as amplitude proxy.")
    print("================================================================")


    # load file by file and compute observables
    results_mz, results_mx = [], []

    for h in h_values:
        filename = f"tfim_{side_length}x{side_length}_h{h:.2f}_{file_samples}.npz"
        filepath = meas_dir / filename

        ds = MeasurementDataset([filepath], load_fn=load_measurements_npz,
                                system_param_keys=["h"], samples_per_file=[eval_samples])

        samples = torch.as_tensor(ds.values, dtype=torch.uint8)

        scorer = DatasetLogScore(samples)

        mz = calculate_mz(samples)
        mx = calculate_mx(samples, log_score_fn=scorer)

        results_mz.append(mz)
        results_mx.append(mx)

        print(f"h {h:5.2f} | Mz: {mz:.5f} | Mx: {mx:.5f}")


    #### PLOT AGAINST REFERENCE CURVES ####

    ref_df = pd.read_csv(ref_file)

    plt.figure(figsize=(8, 5))

    plt.plot(ref_df["h"], ref_df["mag_z"], ':', color='gray', linewidth=2.0, label='_nolegend_', zorder=1)
    plt.plot(ref_df["h"], ref_df["mag_x"], ':', color='gray', linewidth=2.0,
             label=r'$\langle \sigma^{z/x} \rangle_{\mathrm{reference}}$', zorder=1)

    plt.plot(h_values, results_mz, 'o', markersize=8,
             label=r'$\langle | \sigma^z | \rangle_{\mathrm{data}}$', zorder=2)
    plt.plot(h_values, results_mx, 'D', markersize=8,
             label=r'$\langle \sigma^x \rangle_{\mathrm{data}}$', zorder=2)

    plt.xlabel(r'$h$', fontsize=14)
    plt.ylabel('Magnetization', fontsize=14)
    plt.title(f"TFIM ({side_length}x{side_length}) - Magnetizations from Training Data", fontsize=16)

    plt.legend(frameon=True, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()