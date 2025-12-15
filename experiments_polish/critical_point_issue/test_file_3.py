#!/usr/bin/env python3
import argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
import matplotlib.pyplot as plt

from netket.graph import Hypercube
from netket.hilbert import Spin
from netket.operator import Ising


def build_tfim(L: int, dim: int, pbc: bool, J: float):
    graph = Hypercube(length=L, n_dim=dim, pbc=pbc)
    hilbert = Spin(s=0.5, N=graph.n_nodes)

    def H(h: float) -> sp.csr_matrix:
        return Ising(hilbert, graph, h=h, J=J).to_sparse().tocsr()

    return graph, hilbert, H


def ground_state(Hs: sp.csr_matrix):
    vals, vecs = eigsh(Hs, k=1, which="SA", tol=1e-10, maxiter=20000)
    psi = vecs[:, 0]
    # stoquastic TFIM -> real eigenvector; fix random global sign flips
    idx = int(np.argmax(np.abs(psi)))
    if psi[idx] != 0.0:
        psi = psi * np.sign(psi[idx])
    return float(vals[0]), psi


def fidelity_susceptibility_from_overlap(psi0: np.ndarray, psi1: np.ndarray, dh: float) -> float:
    # gauge-safe
    F = abs(np.vdot(psi0, psi1))
    return float(2.0 * (1.0 - F) / (dh * dh))


def probs_from_psi(psi: np.ndarray):
    P = np.abs(psi) ** 2
    return P / P.sum()


def shannon_entropy(P: np.ndarray, eps: float = 1e-300) -> float:
    Q = np.clip(P, eps, 1.0)
    return float(-np.sum(Q * np.log(Q)))


def renyi2_entropy(P: np.ndarray, eps: float = 1e-300) -> float:
    return float(-np.log(np.sum(P * P) + eps))


def effective_support(P: np.ndarray) -> float:
    # N_eff = exp(S2) = 1/sum P^2
    return float(1.0 / np.sum(P * P))


def magnetization_moments(P: np.ndarray, N: int):
    """
    Sanity check in z-basis using 0/1 bitstrings:
      w = popcount(state)
      m = (2*w - N)/N
    """
    dimH = P.shape[0]
    states = np.arange(dimH, dtype=np.uint32)  # N=16 fits

    # popcount per state
    if hasattr(np, "bit_count"):
        w = np.bit_count(states).astype(np.float64)
    else:
        bytes_ = states.view(np.uint8).reshape(dimH, states.dtype.itemsize)  # (dimH, 4)
        bits = np.unpackbits(bytes_, axis=1)                                 # (dimH, 32)
        w = bits.sum(axis=1).astype(np.float64)

    m = (2.0 * w - N) / N
    mz = float(np.sum(P * m))
    mz2 = float(np.sum(P * (m * m)))
    abs_mz = float(np.sum(P * np.abs(m)))
    return mz, mz2, abs_mz


def sample_coverage(P: np.ndarray, nshots: int, rng: np.random.Generator):
    """
    Draw nshots samples from P, return:
      - number of unique states seen
      - probability mass covered by those unique states
    """
    idxs = rng.choice(P.shape[0], size=nshots, replace=True, p=P)
    unique = np.unique(idxs)
    mass = float(P[unique].sum())
    return int(unique.size), mass


def main():
    ap = argparse.ArgumentParser(
        description="TFIM hardness diagnostics: compare chi_F peak vs distribution complexity (entropy/support) vs finite-shot coverage."
    )
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--pbc", action="store_true", default=True)
    ap.add_argument("--J", type=float, default=-1.0)
    ap.add_argument("--hmin", type=float, default=1.0)
    ap.add_argument("--hmax", type=float, default=7.0)
    ap.add_argument("--npts", type=int, default=61)
    ap.add_argument("--dh", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, nargs="*", default=[2000, 10000, 50000])
    ap.add_argument("--no_plots", action="store_true", default=False)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    graph, _, H = build_tfim(args.L, args.dim, args.pbc, args.J)
    N = graph.n_nodes
    dimH = 1 << N
    print(f"TFIM: L={args.L} dim={args.dim} pbc={args.pbc} N={N} dimH={dimH} J={args.J}")

    hs = np.linspace(args.hmin, args.hmax, args.npts)

    chiF = np.zeros_like(hs)
    S1 = np.zeros_like(hs)
    S2 = np.zeros_like(hs)
    Neff = np.zeros_like(hs)

    mz = np.zeros_like(hs)
    mz2 = np.zeros_like(hs)
    abs_mz = np.zeros_like(hs)

    uniq = {n: np.zeros_like(hs, dtype=int) for n in args.shots}
    mass = {n: np.zeros_like(hs, dtype=float) for n in args.shots}

    for i, h in enumerate(tqdm(hs, desc="ED scan")):
        _, psi0 = ground_state(H(h))
        _, psi1 = ground_state(H(h + args.dh))
        chiF[i] = fidelity_susceptibility_from_overlap(psi0, psi1, args.dh)

        P = probs_from_psi(psi0)
        S1[i] = shannon_entropy(P)
        S2[i] = renyi2_entropy(P)
        Neff[i] = effective_support(P)

        mz[i], mz2[i], abs_mz[i] = magnetization_moments(P, N)

        for n in args.shots:
            u, m = sample_coverage(P, n, rng)
            uniq[n][i] = u
            mass[n][i] = m

    i_chi = int(np.argmax(chiF))
    i_S2 = int(np.argmax(S2))
    i_Ne = int(np.argmax(Neff))

    print("\n=== Where things peak ===")
    print(f"chi_F peak at h ≈ {hs[i_chi]:.4f}")
    print(f"Renyi-2 entropy S2 peak at h ≈ {hs[i_S2]:.4f}")
    print(f"Effective support Neff peak at h ≈ {hs[i_Ne]:.4f}")
    print("=========================\n")

    # Save CSV for overlay with your overlap curve
    cols = [hs, chiF, S1, S2, Neff, mz, mz2, abs_mz]
    header = ["h", "chiF", "S1_shannon", "S2_renyi2", "Neff_1/sumP2", "mz", "mz2", "abs_mz"]
    for n in args.shots:
        cols.append(uniq[n])
        cols.append(mass[n])
        header.append(f"unique_{n}")
        header.append(f"mass_covered_{n}")

    out = np.column_stack(cols)
    np.savetxt("tfim_hardness_diagnostics.csv", out, delimiter=",", header=",".join(header), comments="")
    print("Saved: tfim_hardness_diagnostics.csv")

    if args.no_plots:
        return

    plt.figure(figsize=(8, 5), dpi=120)
    plt.plot(hs, chiF, marker="o", markersize=3)
    plt.xlabel(r"field $h$")
    plt.ylabel(r"$\chi_F$ (overlap)")
    plt.title(f"TFIM {args.L}x{args.L}: Fidelity susceptibility")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5), dpi=120)
    plt.plot(hs, S1, marker="o", markersize=3, label="Shannon S1")
    plt.plot(hs, S2, marker="o", markersize=3, label="Renyi-2 S2")
    plt.xlabel(r"field $h$")
    plt.ylabel("entropy of $P=|\\psi|^2$")
    plt.title("Distribution hardness proxies (basis-dependent)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5), dpi=120)
    plt.plot(hs, Neff, marker="o", markersize=3)
    plt.xlabel(r"field $h$")
    plt.ylabel(r"$N_{\\rm eff} = 1/\\sum_\\sigma P(\\sigma)^2$")
    plt.title("Effective support size")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5), dpi=120)
    for n in args.shots:
        plt.plot(hs, mass[n], marker="o", markersize=3, label=f"mass covered, shots={n}")
    plt.xlabel(r"field $h$")
    plt.ylabel("probability mass covered by seen states")
    plt.title("Finite-shot coverage: where learning gets data-limited")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
