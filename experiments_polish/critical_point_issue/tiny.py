#!/usr/bin/env python3
# tfim_4x4_hardness_ed_only.py
#
# ED-only diagnostics to explain "why learning is hardest around h~3":
#   - chi_F(h) from neighbor overlap
#   - N_eff(h) = 1 / sum_s P(s)^2   (effective support size / Renyi-2 support)
#   - nearest-neighbor connected ZZ correlation:
#       Cnn_conn(h) = <z_i z_j> - <z_i><z_j> averaged over NN edges
#   - (optional) naive gap E1-E0 (warning: for small h it’s dominated by cat-splitting)
#
# Plot at end: chi_F, N_eff, |Cnn_conn| versus h
#
# Run:
#   python tfim_4x4_hardness_ed_only.py
#   python tfim_4x4_hardness_ed_only.py --hmin 1 --hmax 7 --npts 121 --dh 1e-3
#
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from netket.graph import Hypercube
from netket.hilbert import Spin
from netket.operator import Ising


def build_tfim(L: int, dim: int, pbc: bool, J: float):
    graph = Hypercube(length=L, n_dim=dim, pbc=pbc)
    hilbert = Spin(s=0.5, N=graph.n_nodes)

    def H_sparse(h: float) -> sp.csr_matrix:
        return Ising(hilbert, graph, h=h, J=J).to_sparse().tocsr()

    return graph, hilbert, H_sparse


def sign_fix_real(psi: np.ndarray) -> np.ndarray:
    """Fix global sign: largest amplitude positive."""
    psi = np.asarray(psi)
    idx = int(np.argmax(np.abs(psi)))
    if np.iscomplexobj(psi):
        phase = np.angle(psi[idx]) if psi[idx] != 0 else 0.0
        psi = psi * np.exp(-1j * phase)
        psi = np.real_if_close(psi, tol=1e6)
    psi = np.asarray(psi, dtype=np.float64)
    if psi[idx] < 0:
        psi = -psi
    # normalize defensively
    nrm = np.linalg.norm(psi)
    if nrm != 0:
        psi = psi / nrm
    return psi


def ground_state(Hs: sp.csr_matrix):
    vals, vecs = eigsh(Hs, k=1, which="SA", tol=1e-10, maxiter=20000)
    E0 = float(vals[0])
    psi = sign_fix_real(vecs[:, 0])
    return E0, psi


def two_lowest(Hs: sp.csr_matrix):
    vals, _ = eigsh(Hs, k=2, which="SA", tol=1e-10, maxiter=20000)
    vals = np.sort(np.asarray(vals, dtype=np.float64))
    return float(vals[0]), float(vals[1])


def z_eigenvalues_from_states(states: np.ndarray) -> np.ndarray:
    """
    states: (dimH, N) from hilbert.all_states()
    Returns z-eigs in {-1,+1} with same shape.
    """
    uniq = np.unique(states)
    if len(uniq) != 2:
        raise ValueError(f"Expected 2 local states, got uniq={uniq}")

    lo, hi = float(np.min(uniq)), float(np.max(uniq))

    # Common possibilities:
    #   {-1/2, +1/2} -> sigma^z eigenvalues are { -1, +1 } via 2*state
    #   {-1, +1}     -> already sigma^z
    #   {0, 1}       -> map via 2*state - 1
    if np.isclose(lo, -0.5) and np.isclose(hi, 0.5):
        z = 2.0 * states
    elif np.isclose(lo, -1.0) and np.isclose(hi, 1.0):
        z = states
    elif np.isclose(lo, 0.0) and np.isclose(hi, 1.0):
        z = 2.0 * states - 1.0
    else:
        # generic: map lo->-1, hi->+1
        z = np.where(np.isclose(states, hi), 1.0, -1.0)

    return np.asarray(z, dtype=np.float64)


def get_edges_array(graph) -> np.ndarray:
    e = graph.edges()
    e = np.asarray(e, dtype=np.int64)
    if e.ndim != 2 or e.shape[1] != 2:
        raise ValueError(f"graph.edges() unexpected shape: {e.shape}")
    return e


def compute_observables(psi: np.ndarray, z: np.ndarray, edges: np.ndarray):
    """
    psi: (dimH,) real normalized
    z: (dimH, N) sigma^z eigenvalues in {-1,+1}
    edges: (E,2) nearest-neighbor edges
    """
    P = psi * psi  # probability distribution in Z basis
    # effective support / Renyi-2
    Neff = 1.0 / np.sum(P * P)

    # Shannon entropy (optional extra)
    # avoid log(0)
    eps = 1e-300
    H = -np.sum(P * np.log(P + eps))

    # <z_i>
    m = P @ z  # (N,)

    # <z_i z_j> on edges
    i = edges[:, 0]
    j = edges[:, 1]
    prod = z[:, i] * z[:, j]               # (dimH, E)
    zz_edge = (P[:, None] * prod).sum(0)   # (E,)

    conn_edge = zz_edge - m[i] * m[j]      # (E,)
    Cnn_conn = float(np.mean(conn_edge))
    Cnn_abs = float(np.mean(np.abs(conn_edge)))

    # also report the *raw* edge correlator average (not connected)
    Cnn_raw = float(np.mean(zz_edge))

    return {
        "Neff": float(Neff),
        "Neff_norm": float(Neff / P.shape[0]),
        "H_shannon": float(H),
        "mz_mean": float(np.mean(m)),
        "Cnn_raw": Cnn_raw,
        "Cnn_conn": Cnn_conn,
        "Cnn_conn_abs_mean": Cnn_abs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--pbc", type=int, default=1, help="1=PBC, 0=OBC")
    ap.add_argument("--J", type=float, default=-1.0)

    ap.add_argument("--hmin", type=float, default=1.0)
    ap.add_argument("--hmax", type=float, default=4.0)
    ap.add_argument("--npts", type=int, default=100)
    ap.add_argument("--dh", type=float, default=1e-3)

    ap.add_argument("--do_gap", action="store_true", default=False)
    ap.add_argument("--out_csv", type=str, default="tfim_ed_hardness.csv")
    args = ap.parse_args()

    pbc = bool(args.pbc)
    graph, hilbert, H_sparse = build_tfim(args.L, args.dim, pbc, args.J)
    N = graph.n_nodes
    dimH = 1 << N
    edges = get_edges_array(graph)

    print(f"TFIM ED-only: L={args.L} dim={args.dim} pbc={pbc} N={N} dimH={dimH} J={args.J}")
    print(f"NN edges: {len(edges)}")

    # basis states in the SAME order as the Hamiltonian uses
    states = np.asarray(hilbert.all_states())  # (dimH, N)
    z = z_eigenvalues_from_states(states)      # (dimH, N) in {-1,+1}

    hs = np.linspace(args.hmin, args.hmax, args.npts, dtype=np.float64)

    # cache psi(h) because chi_F uses neighbor
    cache_psi = {}

    def get_psi(h: float):
        hk = float(np.round(h, 12))
        if hk in cache_psi:
            return cache_psi[hk]
        _, psi = ground_state(H_sparse(hk))
        cache_psi[hk] = psi
        return psi

    rows = []
    for h in tqdm(hs, desc="ED scan"):
        psi0 = get_psi(float(h))
        psip = get_psi(float(h + args.dh))

        # chi_F from overlap to neighbor
        F_nei = float(abs(np.vdot(psi0, psip)))
        chiF = float(2.0 * (1.0 - F_nei) / (args.dh * args.dh))

        obs = compute_observables(psi0, z, edges)

        gap = np.nan
        if args.do_gap:
            E0, E1 = two_lowest(H_sparse(float(h)))
            gap = E1 - E0

        rows.append({
            "h": float(h),
            "chiF_overlap": chiF,
            "F_neighbor": F_nei,
            "Neff": obs["Neff"],
            "Neff_norm": obs["Neff_norm"],
            "H_shannon": obs["H_shannon"],
            "mz_mean": obs["mz_mean"],
            "Cnn_raw": obs["Cnn_raw"],
            "Cnn_conn": obs["Cnn_conn"],
            "Cnn_conn_abs_mean": obs["Cnn_conn_abs_mean"],
            "gap_E1_E0": float(gap),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")

    # Print key locations
    h_chi_peak = float(df.loc[df["chiF_overlap"].idxmax(), "h"])
    print(f"chi_F peak at h ≈ {h_chi_peak:.6f}")

    # Compare the two points you care about (nearest in grid)
    def nearest_row(h0: float):
        idx = int(np.argmin(np.abs(df["h"].to_numpy() - h0)))
        return df.iloc[idx]

    r25 = nearest_row(2.5454545)
    r30 = nearest_row(3.0)
    print("\n--- Nearest-point comparison ---")
    print(f"h≈{r25['h']:.6f}: chiF={r25['chiF_overlap']:.6f}, Neff_norm={r25['Neff_norm']:.6f}, |Cnn_conn|={abs(r25['Cnn_conn']):.6e}")
    print(f"h≈{r30['h']:.6f}: chiF={r30['chiF_overlap']:.6f}, Neff_norm={r30['Neff_norm']:.6f}, |Cnn_conn|={abs(r30['Cnn_conn']):.6e}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 7), dpi=140)

    ax.plot(df["h"], df["chiF_overlap"], marker="o", markersize=3, linestyle="-", label=r"$\chi_F(h)$ (neighbor overlap)")
    ax.set_xlabel(r"Transverse field $h$")
    ax.set_ylabel(r"$\chi_F$")
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(df["h"], df["Neff_norm"], marker="o", markersize=3, linestyle="--", label=r"$N_{\rm eff}/2^N$")
    ax2.plot(df["h"], np.abs(df["Cnn_conn"]), marker="o", markersize=3, linestyle=":", label=r"$|\langle zz\rangle_c|_{\rm nn}$")
    ax2.set_ylabel(r"$N_{\rm eff}/2^N$ and $|C^{zz}_{\rm nn,conn}|$")

    # vertical markers
    ax.axvline(h_chi_peak, linestyle="--", alpha=0.5)
    ax.text(h_chi_peak, ax.get_ylim()[1]*0.95, "chi_F peak", rotation=90, va="top", ha="right")

    # combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)

    plt.title(f"TFIM ED-only diagnostics (L={args.L}, dim={args.dim}, pbc={pbc})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
