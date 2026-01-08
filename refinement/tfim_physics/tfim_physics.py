#!/usr/bin/env python3
# tfim_ed_hardness_story.py
#
# ED-only "story" for TFIM learning hardness in the Z basis.
#
# Computes:
# - Fidelity susceptibility chi_F(h) from neighbor overlap (gauge-safe)
# - Renyi-2 effective support fraction: Neff/2^N
# - NN connected ZZ correlator magnitude: |C_nn,conn|
# - Total correlation (multi-information) TC(h) and a normalized version TC_norm in [0,1]
#
# Outputs:
# - CSV:   --out_csv     (default: tfim_ed_hardness_story.csv)
# - Report --out_report  (default: tfim_ed_hardness_story_report.md)
# - Plot on screen (+ optional --out_png)
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


# -----------------------------
# TFIM build
# -----------------------------
def build_tfim(L: int, dim: int, pbc: bool, J: float):
    graph = Hypercube(length=L, n_dim=dim, pbc=pbc)
    hilbert = Spin(s=0.5, N=graph.n_nodes)

    def H_sparse(h: float) -> sp.csr_matrix:
        return Ising(hilbert, graph, h=h, J=J).to_sparse().tocsr()

    return graph, hilbert, H_sparse


# -----------------------------
# ED helpers
# -----------------------------
def sign_fix_real(psi: np.ndarray) -> np.ndarray:
    """Fix global sign/phase: largest amplitude positive real. Normalize defensively."""
    psi = np.asarray(psi)
    idx = int(np.argmax(np.abs(psi)))
    if np.iscomplexobj(psi):
        phase = np.angle(psi[idx]) if psi[idx] != 0 else 0.0
        psi = psi * np.exp(-1j * phase)
        psi = np.real_if_close(psi, tol=1e6)
    psi = np.asarray(psi, dtype=np.float64)
    if psi[idx] < 0:
        psi = -psi
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


# -----------------------------
# Z-basis observables
# -----------------------------
def z_eigenvalues_from_states(states: np.ndarray) -> np.ndarray:
    """
    states: (dimH, N) from hilbert.all_states()
    Returns sigma^z eigenvalues in {-1,+1} with same shape.
    """
    uniq = np.unique(states)
    if len(uniq) != 2:
        raise ValueError(f"Expected 2 local states, got uniq={uniq}")

    lo, hi = float(np.min(uniq)), float(np.max(uniq))

    if np.isclose(lo, -0.5) and np.isclose(hi, 0.5):
        z = 2.0 * states
    elif np.isclose(lo, -1.0) and np.isclose(hi, 1.0):
        z = states
    elif np.isclose(lo, 0.0) and np.isclose(hi, 1.0):
        z = 2.0 * states - 1.0
    else:
        z = np.where(np.isclose(states, hi), 1.0, -1.0)

    return np.asarray(z, dtype=np.float64)


def get_edges_array(graph) -> np.ndarray:
    e = np.asarray(graph.edges(), dtype=np.int64)
    if e.ndim != 2 or e.shape[1] != 2:
        raise ValueError(f"graph.edges() unexpected shape: {e.shape}")
    return e


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Binary Shannon entropy in nats."""
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def compute_observables(psi: np.ndarray, z: np.ndarray, edges: np.ndarray):
    """
    psi: (dimH,) real normalized
    z: (dimH, N) sigma^z eigenvalues in {-1,+1}
    edges: (E,2) nearest-neighbor edges

    Returns:
      - Neff, Neff_norm
      - Cnn_conn, Cnn_conn_abs_mean
      - H_joint (nats), H_single_sum (nats), TC (nats)
    """
    P = psi * psi
    dimH, N = z.shape

    # Renyi-2 effective support
    Neff = 1.0 / np.sum(P * P)
    Neff_norm = Neff / dimH

    # single-site means: <z_i>
    m = P @ z  # (N,)

    # NN connected ZZ
    i = edges[:, 0]
    j = edges[:, 1]
    zz_edge = (P[:, None] * (z[:, i] * z[:, j])).sum(0)
    conn_edge = zz_edge - m[i] * m[j]
    Cnn_conn = float(np.mean(conn_edge))
    Cnn_conn_abs_mean = float(np.mean(np.abs(conn_edge)))

    # Entropies (Shannon) and total correlation TC = sum_i H(p_i) - H(P)
    eps = 1e-300
    H_joint = float(-np.sum(P * np.log(P + eps)))  # nats

    # p(z_i=+1) = (1 + <z_i>)/2
    p_up = 0.5 * (1.0 + m)
    H_single_sum = float(np.sum(binary_entropy(p_up)))  # nats

    TC = float(H_single_sum - H_joint)  # nats

    return {
        "Neff": float(Neff),
        "Neff_norm": float(Neff_norm),
        "Cnn_conn": float(Cnn_conn),
        "Cnn_conn_abs_mean": float(Cnn_conn_abs_mean),
        "H_joint_nats": float(H_joint),
        "H_single_sum_nats": float(H_single_sum),
        "TC_nats": float(TC),
    }


# -----------------------------
# Story helpers
# -----------------------------
def nearest_row(df: pd.DataFrame, h0: float) -> pd.Series:
    idx = int(np.argmin(np.abs(df["h"].to_numpy() - float(h0))))
    return df.iloc[idx]


def dh_for_target_infidelity(chiF: float, one_minus_F: float) -> float:
    # 1 - F ≈ 0.5 * chiF * dh^2  -> dh ≈ sqrt(2(1-F)/chiF)
    chiF = float(max(chiF, 1e-15))
    one_minus_F = float(max(one_minus_F, 0.0))
    return float(np.sqrt(2.0 * one_minus_F / chiF))


def make_report(df: pd.DataFrame, N: int, args) -> str:
    ln2 = float(np.log(2.0))

    i_chi = int(df["chiF_overlap"].idxmax())
    h_chi = float(df.loc[i_chi, "h"])
    chi_max = float(df.loc[i_chi, "chiF_overlap"])

    i_tc = int(df["TC_norm"].idxmax())
    h_tc = float(df.loc[i_tc, "h"])
    tc_max = float(df.loc[i_tc, "TC_norm"])

    i_ne = int(df["Neff_norm"].idxmax())
    h_ne = float(df.loc[i_ne, "h"])
    ne_max = float(df.loc[i_ne, "Neff_norm"])

    dh_1pct_at_peak = dh_for_target_infidelity(chi_max, 0.01)

    lines = []
    lines.append("# TFIM ED hardness story report\n")
    lines.append("## Summary\n")
    lines.append(f"- System: L={args.L}, dim={args.dim}, pbc={args.pbc and not args.obc}, N={N}, J={args.J}\n")
    lines.append(f"- Scan: h in [{args.hmin}, {args.hmax}] with npts={args.npts}, dh={args.dh}\n")
    lines.append("\n## Key locations (finite-size / basis-dependent diagnostics)\n")
    lines.append(f"- $\\chi_F$ peak at **h ≈ {h_chi:.6f}**, $\\chi_F \\approx {chi_max:.6f}$\n")
    lines.append(f"  - Sensitivity scale: for 1% infidelity, $\\Delta h \\approx {dh_1pct_at_peak:.3e}$ from $1-F\\approx\\tfrac12\\chi_F\\Delta h^2$\n")
    lines.append(f"- $\\mathrm{{TC}}$ peak at **h ≈ {h_tc:.6f}**, $\\mathrm{{TC}}_\\mathrm{{norm}} \\approx {tc_max:.6f}$\n")
    lines.append(f"- $N_\\mathrm{{eff}}/2^N$ peak at **h ≈ {h_ne:.6f}**, $N_\\mathrm{{eff}}/2^N \\approx {ne_max:.6f}$\n")

    lines.append("\n## Interpretation (what tends to correlate with RBM difficulty near h≈3)\n")
    lines.append("- $\\chi_F(h)$: where the *true* state changes fastest with $h$ (error amplification).\n")
    lines.append("- $N_{\\rm eff}/2^N$: how *broad* $P_h(s)=|\\psi_h(s)|^2$ is in the Z basis (data + support burden).\n")
    lines.append("- $\\mathrm{TC}_{\\rm norm}$: how *dependent* the distribution is beyond single-site marginals (broad **and** structured is worst).\n")
    lines.append("- $|C^{zz}_{{\\rm nn,conn}}|$: a local-structure sanity check.\n")

    lines.append("\n## Probe points\n")
    for h0 in args.h_probe:
        r = nearest_row(df, float(h0))
        dh_1pct = dh_for_target_infidelity(float(r["chiF_overlap"]), 0.01)
        lines.append(f"### h ≈ {float(r['h']):.6f} (requested probe near {h0})\n")
        lines.append(f"- $\\chi_F \\approx {float(r['chiF_overlap']):.6f}$  (1% infidelity at $\\Delta h \\approx {dh_1pct:.3e}$)\n")
        lines.append(f"- $N_\\mathrm{{eff}}/2^N \\approx {float(r['Neff_norm']):.6f}$\n")
        lines.append(f"- $\\mathrm{{TC}}_\\mathrm{{norm}} \\approx {float(r['TC_norm']):.6f}$  (TC normalized by $N\\log 2$)\n")
        lines.append(f"- $|C^{{zz}}_{{\\rm nn,conn}}| \\approx {abs(float(r['Cnn_conn'])):.6f}$\n")
        lines.append("\n**Reading:** if your RBM is worst here, it’s typically because $N_{\\rm eff}$ is already growing while TC and/or correlations remain nontrivial.\n")

    if args.with_gap_naive:
        lines.append("\n## Gap note\n")
        lines.append("- The naive gap $E_1-E_0$ is included, but at small h it can be dominated by even/odd cat-splitting.\n")
        lines.append("  Treat it as a sanity check, not a clean phase diagnostic on small sizes.\n")

    lines.append("\n## Units\n")
    lines.append(f"- Entropies computed with natural logs (nats). Normalized TC uses $N\\log 2 \\approx {N*ln2:.6f}$.\n")

    return "".join(lines)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="TFIM ED-only story: chi_F + support fraction + dependence (TC) + NN connected correlations."
    )
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--dim", type=int, default=2)

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pbc", action="store_true", default=True, help="periodic boundary conditions (default)")
    g.add_argument("--obc", action="store_true", default=False, help="open boundary conditions")

    ap.add_argument("--J", type=float, default=-1.0)

    ap.add_argument("--hmin", type=float, default=1.0)
    ap.add_argument("--hmax", type=float, default=4.0)
    ap.add_argument("--npts", type=int, default=100)
    ap.add_argument("--dh", type=float, default=1e-3)

    ap.add_argument("--h_probe", type=float, nargs="*", default=[3.0],
                    help="fields to print a focused comparison for (default: 3.0)")
    ap.add_argument("--with_gap_naive", action="store_true", default=False,
                    help="also compute naive gap E1-E0 (warning: can be misleading at small h)")

    ap.add_argument("--out_csv", type=str, default="tfim_ed_hardness_story.csv")
    ap.add_argument("--out_report", type=str, default="tfim_ed_hardness_story_report.md")
    ap.add_argument("--out_png", type=str, default="", help="optional: save figure to this PNG path")
    ap.add_argument("--no_plot", action="store_true", default=False)

    args = ap.parse_args()

    pbc = bool(args.pbc) and (not bool(args.obc))
    graph, hilbert, H_sparse = build_tfim(args.L, args.dim, pbc, args.J)
    N = graph.n_nodes
    dimH = 1 << N
    edges = get_edges_array(graph)

    print("\n============================================================")
    print("TFIM ED hardness story scan")
    print("------------------------------------------------------------")
    print(f"L={args.L} dim={args.dim}  pbc={pbc}  N={N}  dimH={dimH}  J={args.J}")
    print(f"h in [{args.hmin}, {args.hmax}] with npts={args.npts}, dh={args.dh}")
    print(f"NN edges: {len(edges)}")
    print("============================================================\n")

    # Basis states in correct order + z eigenvalues
    states = np.asarray(hilbert.all_states())  # (dimH, N)
    z = z_eigenvalues_from_states(states)      # (dimH, N) in {-1,+1}

    hs = np.linspace(args.hmin, args.hmax, args.npts, dtype=np.float64)

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

        F_nei = float(abs(np.vdot(psi0, psip)))
        chiF = float(2.0 * (1.0 - F_nei) / (args.dh * args.dh))

        obs = compute_observables(psi0, z, edges)

        gap = np.nan
        if args.with_gap_naive:
            E0, E1 = two_lowest(H_sparse(float(h)))
            gap = E1 - E0

        rows.append({
            "h": float(h),
            "chiF_overlap": float(chiF),
            "F_neighbor": float(F_nei),
            "Neff": obs["Neff"],
            "Neff_norm": obs["Neff_norm"],
            "Cnn_conn": obs["Cnn_conn"],
            "Cnn_conn_abs_mean": obs["Cnn_conn_abs_mean"],
            "H_joint_nats": obs["H_joint_nats"],
            "H_single_sum_nats": obs["H_single_sum_nats"],
            "TC_nats": obs["TC_nats"],
            "gap_naive_E1_E0": float(gap),
        })

    df = pd.DataFrame(rows)

    # normalized TC in [0,1] (max is N log 2 for binary variables)
    df["TC_norm"] = df["TC_nats"] / (float(N) * float(np.log(2.0)))
    df["TC_norm"] = df["TC_norm"].clip(lower=0.0)

    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved CSV: {args.out_csv}")

    report = make_report(df, N, args)
    with open(args.out_report, "w") as f:
        f.write(report)
    print(f"Saved report: {args.out_report}\n")

    # Print a short console story (also present in report)
    i_chi = int(df["chiF_overlap"].idxmax())
    i_tc = int(df["TC_norm"].idxmax())
    i_ne = int(df["Neff_norm"].idxmax())
    print("=== Console story ===")
    print(f"chi_F peak at h ≈ {float(df.loc[i_chi, 'h']):.6f} with chi_F ≈ {float(df.loc[i_chi, 'chiF_overlap']):.6f}")
    print(f"TC_norm peak at h ≈ {float(df.loc[i_tc, 'h']):.6f} with TC_norm ≈ {float(df.loc[i_tc, 'TC_norm']):.6f}")
    print(f"Neff_norm peak at h ≈ {float(df.loc[i_ne, 'h']):.6f} with Neff_norm ≈ {float(df.loc[i_ne, 'Neff_norm']):.6f}")
    for h0 in args.h_probe:
        r = nearest_row(df, float(h0))
        print(f"Probe h≈{float(r['h']):.6f}: chi_F={float(r['chiF_overlap']):.6f}, Neff/2^N={float(r['Neff_norm']):.6f}, "
              f"TC_norm={float(r['TC_norm']):.6f}, |Cnn_conn|={abs(float(r['Cnn_conn'])):.6f}")
    print("=====================\n")

    if args.no_plot:
        return

    # -----------------------------
    # Story plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 7), dpi=140)

    ax.plot(
        df["h"], df["chiF_overlap"],
        marker="o", markersize=3, linestyle="-",
        label=r"$\chi_F(h)$ (neighbor overlap)"
    )
    ax.set_xlabel(r"Transverse field $h$")
    ax.set_ylabel(r"Fidelity susceptibility $\chi_F$")
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(
        df["h"], df["Neff_norm"],
        marker="o", markersize=3, linestyle="--",
        label=r"$N_{\rm eff}/2^N$ (support fraction)"
    )
    ax2.plot(
        df["h"], df["TC_norm"],
        marker="o", markersize=3, linestyle="-.",
        label=r"$\mathrm{TC}_{\rm norm}$ (broad + dependent)"
    )
    ax2.plot(
        df["h"], np.abs(df["Cnn_conn"]),
        marker="o", markersize=3, linestyle=":",
        label=r"$|C^{zz}_{{\rm nn,conn}}|$"
    )
    ax2.set_ylabel(r"Support / dependence / local structure (all ~[0,1])")

    # vertical markers: chi_F peak + probes
    h_chi = float(df.loc[int(df["chiF_overlap"].idxmax()), "h"])
    ax.axvline(h_chi, linestyle="--", alpha=0.55)
    ax.text(h_chi, ax.get_ylim()[1] * 0.97, r"$\chi_F$ peak", rotation=90, va="top", ha="right")

    for hh in args.h_probe:
        r = nearest_row(df, float(hh))
        hx = float(r["h"])
        ax.axvline(hx, linestyle="--", alpha=0.20)
        ax.text(hx, ax.get_ylim()[1] * 0.97, f"h≈{hx:.2f}", rotation=90, va="top", ha="left")

    # combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)

    plt.title(f"TFIM ED hardness story (L={args.L}, dim={args.dim}, pbc={pbc})")
    plt.tight_layout()

    if args.out_png:
        plt.savefig(args.out_png, dpi=200)
        print(f"Saved figure: {args.out_png}")

    plt.show()


if __name__ == "__main__":
    main()
