import math
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from netket.graph import Hypercube
from typing_extensions import TypedDict  # still unused, but left as-is

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

MEASUREMENT_DIR = Path("measurements")

# reference CSV: columns  delta, czz_diag, cxx_diag
REF_CURVE_PATH = Path("xxz_4x4_correlations_ref.csv")

# measurement files (z-basis samples)
FILE_NAMES = [
    "xxz_4x4_delta0.40_1000000.npz",
    "xxz_4x4_delta0.60_1000000.npz",
    "xxz_4x4_delta0.80_1000000.npz",
    "xxz_4x4_delta0.90_1000000.npz",
    "xxz_4x4_delta0.95_1000000.npz",
    "xxz_4x4_delta1.00_1000000.npz",
    "xxz_4x4_delta1.05_1000000.npz",
    "xxz_4x4_delta1.10_1000000.npz",
    "xxz_4x4_delta1.40_1000000.npz",
    "xxz_4x4_delta2.00_1000000.npz",
]

SYSTEM_PARAM_KEY = "delta"

# make local data_handling importable
sys.path.append(str(Path("..").resolve()))
from data_handling import load_measurements_npz, MeasurementDataset  # noqa: E402


# ---------------------------------------------------------------------
# Lattice helper: diagonal pairs as in ED script
# ---------------------------------------------------------------------

def main_diagonal_pairs_from_graph(L: int) -> List[Tuple[int, int]]:
    """
    Construct site index pairs (i, j) along the main diagonal
    using the same convention as in the ED script (NetKet Hypercube).
    """
    graph = Hypercube(length=L, n_dim=2, pbc=True)
    coords = np.asarray(graph.basis_coords, dtype=int)

    Lx = coords[:, 0].max() + 1
    Ly = coords[:, 1].max() + 1
    assert Lx == Ly == L

    xy_to_id = {(c[0], c[1]): i for i, c in enumerate(coords)}

    pairs = [
        (xy_to_id[(x, x)], xy_to_id[((x + 1) % L, (x + 1) % L)])
        for x in range(L)
    ]
    return pairs


# ---------------------------------------------------------------------
# Probability helpers (data -> P(s))
# ---------------------------------------------------------------------

def bitstrings_to_indices(samples: torch.Tensor) -> torch.Tensor:
    """
    Map bitstrings in {0,1}^N to integer labels 0..2^N-1 with
    index = sum_i bit_i * 2^i  (site 0 is least significant bit).
    """
    samples = samples.long()
    B, N = samples.shape
    powers = (2 ** torch.arange(N, device=samples.device)).long()
    idx = (samples * powers).sum(dim=1)
    return idx


def empirical_probs(samples: torch.Tensor) -> torch.Tensor:
    """
    Given B samples in {0,1}^N, estimate P(s) over all 2^N configs.
    Returns probs as (2^N,) tensor (float64), normalized.
    """
    B, N = samples.shape
    idx = bitstrings_to_indices(samples)
    n_states = 2 ** N
    counts = torch.bincount(idx, minlength=n_states).to(torch.float64)
    probs = counts / counts.sum()
    return probs


# ---------------------------------------------------------------------
# Correlators from probabilities (this is where free-energy probs will plug in)
# ---------------------------------------------------------------------

def correlator_zz_from_probs(
        probs: torch.Tensor,
        corr_pairs: List[Tuple[int, int]],
        num_samples: int,
) -> Tuple[float, float]:
    """
    Compute C_zz on given pairs from probabilities P(s) in z-basis.

    probs      : (2^N,)  with sum probs = 1
    corr_pairs : list of (i,j)
    num_samples: how many experimental samples were used to estimate probs
                 (for SEM; pass None if you don't care)
    """
    device = probs.device
    n_states = probs.shape[0]
    N = int(round(math.log2(n_states)))
    assert 2 ** N == n_states, "probs length must be power of two."

    # reconstruct bitstrings for all configs
    indices = torch.arange(n_states, device=device, dtype=torch.long)
    bits = ((indices.unsqueeze(1) >> torch.arange(N, device=device)) & 1).to(torch.float64)
    spins_z = 1.0 - 2.0 * bits  # {0,1} -> {+1,-1}

    # per-config average over pairs
    f = torch.zeros(n_states, device=device, dtype=torch.float64)
    for (i, j) in corr_pairs:
        f += spins_z[:, i] * spins_z[:, j]
    f /= float(len(corr_pairs))

    mean = (probs * f).sum()

    if num_samples is None:
        return float(mean), float("nan")

    # exact variance under probs, SEM for estimator with num_samples shots
    var = (probs * f**2).sum() - mean**2
    sem = math.sqrt(max(var.item(), 0.0) / float(num_samples))
    return float(mean), float(sem)


def correlator_xx_from_probs(
        probs: torch.Tensor,
        corr_pairs: List[Tuple[int, int]],
        num_samples: int,
) -> Tuple[float, float]:
    """
    Compute C_xx on given pairs using only probabilities in z-basis.

    Uses (for real, non-negative ψ):
        <σ_i^x σ_j^x> = sum_s sqrt(P(s) P(s^{ij}))
    implemented as:
        r_ij(s) = sqrt(P(s^{ij}) / P(s))
        C_xx    = E_s[ average_pairs r_ij(s) ].

    probs      : (2^N,)  with sum probs = 1  (P(s) in z-basis)
    corr_pairs : list of (i,j)
    num_samples: number of samples used to estimate probs (for SEM)
    """
    device = probs.device
    n_states = probs.shape[0]
    N = int(round(math.log2(n_states)))
    assert 2 ** N == n_states, "probs length must be power of two."

    indices = torch.arange(n_states, device=device, dtype=torch.long)
    eps = 1e-16

    # per-config average over pairs of amplitude ratios r_ij(s)
    f = torch.zeros(n_states, device=device, dtype=torch.float64)

    for (i, j) in corr_pairs:
        mask = (1 << i) | (1 << j)     # bitmask flipping i and j
        idx_flip = indices ^ mask      # s^{ij}
        r_ij = torch.sqrt(
            (probs[idx_flip] + eps) / (probs[indices] + eps)
        )
        f += r_ij

    f /= float(len(corr_pairs))

    mean = (probs * f).sum()

    if num_samples is None:
        return float(mean), float("nan")

    var = (probs * f**2).sum() - mean**2
    sem = math.sqrt(max(var.item(), 0.0) / float(num_samples))
    return float(mean), float(sem)


# ---------------------------------------------------------------------
# 1) Load dataset and inspect
# ---------------------------------------------------------------------

def load_and_inspect_dataset(samples_per_file: Optional[int] = None) -> MeasurementDataset:
    file_paths = [MEASUREMENT_DIR / fn for fn in FILE_NAMES]

    # turn scalar samples_per_file into a list aligned with file_paths
    if samples_per_file is None:
        spf_list = None
    else:
        spf_list = [samples_per_file] * len(file_paths)

    ds = MeasurementDataset(
        file_paths,
        load_fn=load_measurements_npz,
        system_param_keys=[SYSTEM_PARAM_KEY],
        samples_per_file=spf_list,  # <- uses your new argument
    )

    print(f"implicit_basis      : {ds.implicit_basis}")
    print(f"values shape        : {tuple(ds.values.shape)}")
    print(
        "system_params shape :",
        None if ds.system_params is None else tuple(ds.system_params.shape),
    )

    # inspect unique deltas
    system_params = torch.as_tensor(ds.system_params)
    h_idx = ds.system_param_keys.index(SYSTEM_PARAM_KEY)
    unique_h = torch.unique(system_params[:, h_idx])
    print(
        f"unique {SYSTEM_PARAM_KEY} values :",
        [float(v) for v in unique_h],
    )

    # if you want to see what was actually used per file:
    print("samples_per_file (effective):", ds.samples_per_file)

    return ds


# ---------------------------------------------------------------------
# 2) Compute Czz(Δ), Cxx(Δ) from data via probabilities
# ---------------------------------------------------------------------

def compute_correlators_from_data(
        ds: MeasurementDataset,
        corr_pairs: List[Tuple[int, int]],
):
    # all raw samples (B_total, N)
    values_all = torch.as_tensor(ds.values, dtype=torch.uint8)
    system_params = torch.as_tensor(ds.system_params)

    B_total, N = values_all.shape
    print(f"Total samples: {B_total}, N={N}")

    num_states = 2 ** N
    if num_states > 1_000_000:
        raise ValueError(
            f"Too many states (2^N = {num_states}); "
            "probability-based enumeration is not practical."
        )

    delta_idx = ds.system_param_keys.index(SYSTEM_PARAM_KEY)
    delta_all = system_params[:, delta_idx]
    unique_delta = torch.sort(torch.unique(delta_all)).values

    data_deltas   = []
    data_czz      = []
    data_czz_err  = []
    data_cxx      = []
    data_cxx_err  = []

    for d in unique_delta:
        mask = (delta_all == d)
        samples_d = values_all[mask]         # (B_d, N)
        B_d = samples_d.shape[0]

        probs_d = empirical_probs(samples_d)

        czz, czz_err = correlator_zz_from_probs(
            probs_d, corr_pairs, num_samples=B_d
        )
        cxx, cxx_err = correlator_xx_from_probs(
            probs_d, corr_pairs, num_samples=B_d
        )

        data_deltas.append(float(d))
        data_czz.append(czz)
        data_czz_err.append(czz_err)
        data_cxx.append(cxx)
        data_cxx_err.append(cxx_err)

        print(
            f"delta={float(d):.2f}  "
            f"Czz_data={czz:+.6f} ± {czz_err:.6f}  "
            f"Cxx_data={cxx:+.6f} ± {cxx_err:.6f}"
        )

    return data_deltas, data_czz, data_czz_err, data_cxx, data_cxx_err


# ---------------------------------------------------------------------
# 3) Plot: ED reference vs data (Czz and Cxx)
# ---------------------------------------------------------------------

def plot_correlators_vs_reference(
        ref_df: pd.DataFrame,
        data_deltas,
        data_czz,
        data_czz_err,
        data_cxx,
        data_cxx_err,
):
    fig, ax = plt.subplots(figsize=(7, 4))

    # ED reference curves
    ax.plot(
        ref_df["delta"],
        ref_df["czz_diag"],
        label=r"ED $C_{zz}$ (diag)",
        linewidth=2,
    )
    ax.plot(
        ref_df["delta"],
        ref_df["cxx_diag"],
        label=r"ED $C_{xx}$ (diag)",
        linewidth=2,
        linestyle="--",
    )

    # data points from measurement probabilities
    ax.errorbar(
        data_deltas,
        data_czz,
        yerr=data_czz_err,
        fmt="o",
        capsize=3,
        label=r"Data $C_{zz}$ (prob-based)",
    )
    ax.errorbar(
        data_deltas,
        data_cxx,
        yerr=data_cxx_err,
        fmt="s",
        capsize=3,
        label=r"Data $C_{xx}$ (prob-based)",
    )

    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"Two-Point Correlators")
    ax.set_title(r"Reference vs Data: $C_{zz}$ and $C_{xx}$, XXZ $4 \times 4$")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------

def main(samples_per_file: Optional[int] = None):
    """
    samples_per_file:
        None -> use all shots in each file.
        int  -> keep at most this many shots per file (via MeasurementDataset).
    """
    # 1) load & inspect data
    ds = load_and_inspect_dataset(samples_per_file=samples_per_file)

    # deduce linear system size from number of spins N
    N = ds.values.shape[1]
    side_length = int(math.isqrt(N))
    assert side_length * side_length == N, "num_visible must be a perfect square."
    print(f"Assuming {side_length}x{side_length} lattice (N={N}).")

    # diagonal pairs in the ED convention
    corr_pairs = main_diagonal_pairs_from_graph(side_length)
    print("Diagonal pairs (ED convention):", corr_pairs)

    # load ED reference file
    ref_df = pd.read_csv(REF_CURVE_PATH)
    print(ref_df.head())

    # 2) compute Czz, Cxx from data via P(s)
    (
        data_deltas,
        data_czz,
        data_czz_err,
        data_cxx,
        data_cxx_err,
    ) = compute_correlators_from_data(ds, corr_pairs)

    # 3) plot everything
    plot_correlators_vs_reference(
        ref_df,
        data_deltas,
        data_czz,
        data_czz_err,
        data_cxx,
        data_cxx_err,
    )


if __name__ == "__main__":
    # change this to try different caps, or set to None for full data
    main(samples_per_file=200_000)
