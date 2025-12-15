#!/usr/bin/env python3
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
import matplotlib.pyplot as plt

from netket.graph import Hypercube
from netket.hilbert import Spin
from netket.operator import Ising


# -----------------------------
# Settings
# -----------------------------
L = 4
DIM = 2
PBC = True
J = -1.0

H_MIN, H_MAX, NPTS = 1.0, 7.0, 61


# -----------------------------
# Build TFIM
# -----------------------------
graph = Hypercube(length=L, n_dim=DIM, pbc=PBC)
hilbert = Spin(s=0.5, N=graph.n_nodes)
N = graph.n_nodes
DIMH = 1 << N

def H_full(h: float) -> sp.csr_matrix:
    return Ising(hilbert, graph, h=h, J=J).to_sparse().tocsr()

def ground_state(Hs: sp.csr_matrix):
    vals, vecs = eigsh(Hs, k=1, which="SA", tol=1e-10, maxiter=20000)
    psi = vecs[:, 0]
    # real stoquastic: fix random global sign
    idx = int(np.argmax(np.abs(psi)))
    if psi[idx] != 0.0:
        psi = psi * np.sign(psi[idx])
    return float(vals[0]), psi

def probs_from_psi(psi: np.ndarray):
    P = (psi * psi)  # real
    return P / P.sum()


# -----------------------------
# Bit helpers (0/1 -> sigma^z in ±1)
# -----------------------------
states = np.arange(DIMH, dtype=np.uint32)

def popcount_uint32(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bit_count"):
        return np.bit_count(x)
    # fallback
    b = x.view(np.uint8).reshape(x.size, x.dtype.itemsize)  # (n,4)
    return np.unpackbits(b, axis=1).sum(axis=1)

w = popcount_uint32(states).astype(np.float64)  # number of 1s per basis state
M = (2.0 * w - N)  # magnetization M = sum_i sigma_i^z, with sigma^z=+1 for bit=1, -1 for bit=0

# nearest-neighbor bonds (as pairs of site indices)
edges = np.array(graph.edges(), dtype=np.int64)  # shape (nbonds, 2)
nbonds = edges.shape[0]

def exp_sigma_i(P: np.ndarray, i: int) -> float:
    bi = ((states >> np.uint32(i)) & np.uint32(1)).astype(np.float64)
    si = 2.0 * bi - 1.0
    return float(np.sum(P * si))

def exp_sigma_ij(P: np.ndarray, i: int, j: int) -> float:
    bi = ((states >> np.uint32(i)) & np.uint32(1)).astype(np.uint32)
    bj = ((states >> np.uint32(j)) & np.uint32(1)).astype(np.uint32)
    xor = (bi ^ bj).astype(np.float64)  # 1 if different
    sij = 1.0 - 2.0 * xor               # +1 if equal, -1 if different
    return float(np.sum(P * sij))


def main():
    hs = np.linspace(H_MIN, H_MAX, NPTS)

    chi_z = np.zeros_like(hs)          # magnetization susceptibility per site
    corr_nn = np.zeros_like(hs)        # average <s_i s_j> over NN bonds
    corr_nn_conn = np.zeros_like(hs)   # average connected <s_i s_j> - <s_i><s_j>

    # (optional) show where correlation structure changes fastest
    dcorr_dh = np.zeros_like(hs)

    for t, h in enumerate(tqdm(hs, desc="ED scan")):
        _, psi = ground_state(H_full(h))
        P = probs_from_psi(psi)

        # susceptibility from M
        EM = float(np.sum(P * M))
        EM2 = float(np.sum(P * (M * M)))
        chi_z[t] = (EM2 - EM * EM) / N

        # NN correlators
        # compute <s_i> once for all sites (small N=16)
        si_means = np.array([exp_sigma_i(P, i) for i in range(N)], dtype=np.float64)

        c_sum = 0.0
        cconn_sum = 0.0
        for (i, j) in edges:
            cij = exp_sigma_ij(P, int(i), int(j))
            c_sum += cij
            cconn_sum += (cij - si_means[i] * si_means[j])

        corr_nn[t] = c_sum / nbonds
        corr_nn_conn[t] = cconn_sum / nbonds

    # finite-difference slope of NN connected correlator (where correlations change fastest)
    dcorr_dh[1:-1] = (corr_nn_conn[2:] - corr_nn_conn[:-2]) / (hs[2:] - hs[:-2])
    dcorr_dh[0] = dcorr_dh[1]
    dcorr_dh[-1] = dcorr_dh[-2]

    i_chi = int(np.argmax(chi_z))
    i_slope = int(np.argmax(np.abs(dcorr_dh)))

    print("\n=== Peaks / fastest-change ===")
    print(f"chi_z peak at h ≈ {hs[i_chi]:.4f}")
    print(f"max |d corr_nn_conn / dh| at h ≈ {hs[i_slope]:.4f}")
    print("==============================\n")

    plt.figure(figsize=(8,5), dpi=120)
    plt.plot(hs, chi_z, marker="o", markersize=3)
    plt.xlabel(r"field $h$")
    plt.ylabel(r"$\chi_z = (\langle M^2\rangle-\langle M\rangle^2)/N$")
    plt.title("Magnetization susceptibility proxy (correlation hardness)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,5), dpi=120)
    plt.plot(hs, corr_nn, marker="o", markersize=3, label=r"$\langle s_i s_j\rangle_{\rm NN}$")
    plt.plot(hs, corr_nn_conn, marker="o", markersize=3, label=r"connected NN")
    plt.xlabel(r"field $h$")
    plt.ylabel("NN correlator")
    plt.title("Nearest-neighbor correlations")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,5), dpi=120)
    plt.plot(hs, np.abs(dcorr_dh), marker="o", markersize=3)
    plt.xlabel(r"field $h$")
    plt.ylabel(r"$|\mathrm{d}\,\langle s_i s_j\rangle_c / \mathrm{d}h|$")
    plt.title("Where correlation structure changes fastest")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
