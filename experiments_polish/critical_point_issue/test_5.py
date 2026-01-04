#!/usr/bin/env python3
# tfim_4x4_fidelity_debug.py
#
# Produces: tfim_4x4_fidelity_debug.csv
# - Fidelity susceptibility (2 ways): FD-projected + overlap (gauge-safe)
# - Gaps: naive (E1-E0), even-sector (E1,+ - E0,+), even-odd splitting (E0,- - E0,+)
#
# Run:
#   python tfim_4x4_fidelity_debug.py
#   python tfim_4x4_fidelity_debug.py --hmin 1 --hmax 7 --npts 61 --dh 1e-3 --k_eigs 8

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


def sign_fix_real(psi: np.ndarray) -> np.ndarray:
    """Fix random global sign: make largest amplitude positive."""
    idx = int(np.argmax(np.abs(psi)))
    if psi[idx] != 0.0:
        psi = psi * np.sign(psi[idx])
    return psi


def align_sign(ref: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Align sign of psi to maximize overlap with ref (real vectors)."""
    ov = float(np.vdot(ref, psi).real)
    s = 1.0 if ov >= 0 else -1.0
    return psi * s


def ground_state_only(Hs: sp.csr_matrix):
    vals, vecs = eigsh(Hs, k=1, which="SA", tol=1e-10, maxiter=20000)
    psi = vecs[:, 0]
    psi = sign_fix_real(psi)
    return float(vals[0]), psi


def low_eigs(Hs: sp.csr_matrix, k: int):
    vals, vecs = eigsh(Hs, k=k, which="SA", tol=1e-10, maxiter=20000)
    # sort
    idx = np.argsort(vals)
    vals = np.array(vals[idx], dtype=np.float64)
    vecs = np.array(vecs[:, idx], dtype=np.float64)
    # fix sign of each vec (not essential, but nice)
    for j in range(vecs.shape[1]):
        vecs[:, j] = sign_fix_real(vecs[:, j])
    return vals, vecs


def parity_resolved_energies(vals: np.ndarray, vecs: np.ndarray, N: int):
    """
    Resolve Z2 parity under P = Π_i σ^x_i, which in z-basis maps |s> -> |~s>.
    Works even when eigsh returns arbitrary mixtures inside (near) degenerate subspaces.
    """
    dimH = vecs.shape[0]
    all_mask = (1 << N) - 1
    idx = np.arange(dimH, dtype=np.uint32)
    flip_idx = (idx ^ np.uint32(all_mask)).astype(np.int64)

    vecs_flip = vecs[flip_idx, :]  # P|psi_j>
    # P matrix in the low-energy subspace: P_ij = <psi_i|P|psi_j>
    Psub = vecs.T @ vecs_flip  # (k,k), real symmetric ideally
    Psub = 0.5 * (Psub + Psub.T)

    p_eigs, U = np.linalg.eigh(Psub)  # columns of U

    # energies in this parity-diagonal basis: E_alpha = sum_i |U_{iα}|^2 * vals_i
    weights = (U * U)  # real
    E_alpha = weights.T @ vals  # (k,)

    # split by parity sign (eigenvalues near ±1)
    plus = [(E_alpha[a], p_eigs[a]) for a in range(len(E_alpha)) if p_eigs[a] >= 0.0]
    minus = [(E_alpha[a], p_eigs[a]) for a in range(len(E_alpha)) if p_eigs[a] < 0.0]
    plus.sort(key=lambda x: x[0])
    minus.sort(key=lambda x: x[0])

    E0p = plus[0][0] if len(plus) >= 1 else np.nan
    E1p = plus[1][0] if len(plus) >= 2 else np.nan
    E0m = minus[0][0] if len(minus) >= 1 else np.nan
    return float(E0p), float(E1p), float(E0m)


def chiF_overlap(psi0: np.ndarray, psi1: np.ndarray, dh: float) -> float:
    F = abs(np.vdot(psi0, psi1))
    return float(2.0 * (1.0 - F) / (dh * dh))


def chiF_fd_projected(psi_minus: np.ndarray, psi0: np.ndarray, psi_plus: np.ndarray, dh: float) -> float:
    # align both sides to psi0 (critical!)
    psi_plus = align_sign(psi0, psi_plus)
    psi_minus = align_sign(psi0, psi_minus)

    dpsi = (psi_plus - psi_minus) / (2.0 * dh)
    term1 = float(np.vdot(dpsi, dpsi).real)
    term2 = float(abs(np.vdot(psi0, dpsi)) ** 2)
    return term1 - term2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--pbc", action="store_true", default=True)
    ap.add_argument("--J", type=float, default=-1.0)
    ap.add_argument("--hmin", type=float, default=1.0)
    ap.add_argument("--hmax", type=float, default=4.0)
    ap.add_argument("--npts", type=int, default=100)
    ap.add_argument("--dh", type=float, default=1e-3)
    ap.add_argument("--k_eigs", type=int, default=8, help="number of low eigenpairs for parity resolution")
    ap.add_argument("--no_plots", action="store_true", default=False)
    args = ap.parse_args()

    graph, _, H = build_tfim(args.L, args.dim, args.pbc, args.J)
    N = graph.n_nodes
    dimH = 1 << N
    print(f"TFIM: L={args.L} dim={args.dim} pbc={args.pbc} N={N} dimH={dimH} J={args.J}")

    hs = np.linspace(args.hmin, args.hmax, args.npts)

    chi_fd = np.zeros_like(hs)
    chi_ov = np.zeros_like(hs)

    gap_naive = np.zeros_like(hs)
    gap_even = np.zeros_like(hs)
    gap_even_odd = np.zeros_like(hs)

    E0p_arr = np.zeros_like(hs)
    E1p_arr = np.zeros_like(hs)
    E0m_arr = np.zeros_like(hs)

    for i, h in enumerate(tqdm(hs, desc="ED scan")):
        # ground states for chi_F
        _, psi0 = ground_state_only(H(h))
        _, psiP = ground_state_only(H(h + args.dh))
        _, psiM = ground_state_only(H(h - args.dh))

        # chi_F
        chi_ov[i] = chiF_overlap(psi0, psiP, args.dh)
        chi_fd[i] = chiF_fd_projected(psiM, psi0, psiP, args.dh)

        # low spectrum once per h
        vals, vecs = low_eigs(H(h), k=max(2, args.k_eigs))
        gap_naive[i] = float(vals[1] - vals[0])

        E0p, E1p, E0m = parity_resolved_energies(vals, vecs, N)
        E0p_arr[i], E1p_arr[i], E0m_arr[i] = E0p, E1p, E0m

        gap_even[i] = (E1p - E0p) if (np.isfinite(E1p) and np.isfinite(E0p)) else np.nan
        gap_even_odd[i] = (E0m - E0p) if (np.isfinite(E0m) and np.isfinite(E0p)) else np.nan

    # peak/min locations
    i_fd = int(np.nanargmax(chi_fd))
    i_ov = int(np.nanargmax(chi_ov))
    i_gap = int(np.nanargmin(gap_naive))

    print("\n==== Peak / minimum locations ====")
    print(f"FD-projected chi_F peak at h ≈ {hs[i_fd]:.4f}")
    print(f"Overlap chi_F peak      at h ≈ {hs[i_ov]:.4f}")
    print(f"Gap minimum             at h ≈ {hs[i_gap]:.4f}")
    print("==================================\n")

    out = np.column_stack([
        hs,
        chi_fd, chi_ov,
        gap_naive, gap_even, gap_even_odd,
        E0p_arr, E1p_arr, E0m_arr
    ])
    header = ",".join([
        "h",
        "chiF_fdproj",
        "chiF_overlap",
        "gap_naive_E1-E0",
        "gap_even_E1p-E0p",
        "gap_even_odd_E0m-E0p",
        "E0_plus",
        "E1_plus",
        "E0_minus"
    ])
    np.savetxt("tfim_4x4_fidelity_debug.csv", out, delimiter=",", header=header, comments="")
    print("Saved: tfim_4x4_fidelity_debug.csv")

    if args.no_plots:
        return

    plt.figure(figsize=(8, 5), dpi=140)
    plt.plot(hs, chi_fd, marker="o", markersize=3, label=r"$\chi_F$ FD-projected (phase-aligned)")
    plt.plot(hs, chi_ov, marker="o", markersize=3, label=r"$\chi_F$ overlap (gauge-safe)")
    plt.xlabel(r"Transverse field $h$")
    plt.ylabel(r"Fidelity susceptibility $\chi_F$")
    plt.title("TFIM 4x4 - Fidelity susceptibility")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5), dpi=140)
    plt.plot(hs, gap_naive, marker="o", markersize=3, label=r"gap naive $E_1-E_0$")
    plt.plot(hs, gap_even, marker="o", markersize=3, label=r"even gap $E_{1,+}-E_{0,+}$")
    plt.plot(hs, gap_even_odd, marker="o", markersize=3, label=r"even-odd $E_{0,-}-E_{0,+}$")
    plt.xlabel(r"Transverse field $h$")
    plt.ylabel("Gap")
    plt.title("Gap diagnostics (parity-resolved)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
