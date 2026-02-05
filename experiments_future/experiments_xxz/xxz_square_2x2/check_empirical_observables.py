import sys
from pathlib import Path
from typing import List, Tuple, Callable

import torch
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_handling import load_measurements_npz, MeasurementDataset



#### SCORING FUNCTION AND TWO-POINT-CORRELATION ESTIMATORS ####

class DatasetLogScore:
    """
    Acts as a statistical stand-in for the free-energy function of our RBM.

    Off-diagonal estimators depend only on the *ratio* of amplitudes between flipped and original states, absolute
    normalization is irrelevant. Because of this we can use the ratio between counts of each bitstring.

    We operate in log-space to maintain numerical stability and compatibility with free-energy scoring.
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



def compute_czz(samples: torch.Tensor, pairs: List[Tuple[int, int]]) -> float:
    spins_pm = 1.0 - 2.0 * samples.float()

    total_czz = 0.0
    for u, v in pairs:
        spin_parities = spins_pm[:, u] * spins_pm[:, v]
        total_czz += spin_parities.mean().item() # batch mean

    return total_czz / len(pairs)


def compute_cxx(samples: torch.Tensor, pairs: List[Tuple[int, int]],
                log_score_fn: Callable[[torch.Tensor], torch.Tensor]) -> float:

    log_scores_orig = log_score_fn(samples)

    total_cxx = 0.0
    for u, v in pairs:
        # we flip spins at positions u and v and compute ratio in log-space
        flipped_samples = samples.clone()
        flipped_samples[:, u] = 1 - flipped_samples[:, u]
        flipped_samples[:, v] = 1 - flipped_samples[:, v]

        log_scores_flip = log_score_fn(flipped_samples)
        log_ratios = log_scores_flip - log_scores_orig
        ratios = torch.exp(log_ratios)

        total_cxx += ratios.mean().item()

    return total_cxx / len(pairs)



if __name__ == "__main__":

    # parameters need to match those used in data generation
    side_length = 2
    periodic_boundaries = False

    delta_values = [0.40, 0.60, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.40, 2.00]
    file_samples = 5_000_000
    eval_samples = 1_000_000    # must be large for large systems, otherwise Cxx underestimates due to missing samples

    meas_dir = Path("measurements")
    ref_file = Path(f"xxz_{side_length}x{side_length}_correlations_ref.csv")


    # derive main diagonal indices with handcrafted formula
    diag_indices = [k * (side_length + 1) for k in range(side_length)]
    targets = diag_indices[1:] + ([diag_indices[0]] if periodic_boundaries else [])
    corr_pairs = list(zip(diag_indices, targets))

    print("=== CROSSCHECK ANALYSIS ========================================")
    print(f"System: {side_length}x{side_length}, PBC={periodic_boundaries}")
    print(f"Correlation pairs (main diagonal): {corr_pairs}")
    print("================================================================")


    # load file by file and compute correlators
    results_czz, results_cxx = [], []
    for delta in delta_values:
        filename = f"xxz_{side_length}x{side_length}_delta{delta:.2f}_{file_samples}.npz"
        filepath = meas_dir / filename

        ds = MeasurementDataset([filepath], load_fn=load_measurements_npz,
                                system_param_keys=["delta"], samples_per_file=[eval_samples])

        samples = torch.as_tensor(ds.values, dtype=torch.uint8)

        scorer = DatasetLogScore(samples)

        czz = compute_czz(samples, corr_pairs)
        cxx = compute_cxx(samples, corr_pairs, log_score_fn=scorer)

        results_czz.append(czz)
        results_cxx.append(cxx)

        print(f"Delta {delta:5.2f} | Czz: {czz:+.5f} | Cxx: {cxx:+.5f}")


    #### PLOT AGAINST REFERENCE CURVES ####

    ref_df = pd.read_csv(ref_file)

    plt.figure(figsize=(8, 5))

    plt.plot(ref_df["delta"], ref_df["czz_diag"], ':', color='gray', linewidth=2.0, label='_nolegend_', zorder=1)
    plt.plot(ref_df["delta"], ref_df["cxx_diag"], ':', color='gray', linewidth=2.0,
             label=r'$\langle \sigma^{z/x} \sigma^{z/x} \rangle_{\mathrm{reference}}$', zorder=1)

    plt.plot(delta_values, results_czz, 'o', markersize=8,
             label=r'$\langle \sigma^z \sigma^z \rangle_{\mathrm{data}}$', zorder=2)
    plt.plot(delta_values, results_cxx, 'D', markersize=8,
             label=r'$\langle \sigma^x \sigma^x \rangle_{\mathrm{data}}$', zorder=2)

    plt.xlabel(r'$\Delta$', fontsize=14)
    plt.ylabel('Two-Point Correlator', fontsize=14)
    plt.title(f"XXZ ({side_length}x{side_length}) - Correlators from Training Data", fontsize=16)

    plt.legend(frameon=True, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()