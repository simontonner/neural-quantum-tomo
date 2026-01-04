import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
import matplotlib.pyplot as plt

from netket.graph import Hypercube
from netket.hilbert import Spin
from netket.operator import Ising


# -----------------------------
# Config
# -----------------------------
L = 4
DIM = 2
PBC = True

# NetKet Ising convention:
# H = -h * sum_i σ^x_i  +  J * sum_<ij> σ^z_i σ^z_j
# Ferromagnetic TFIM -> J < 0
J = -1.0

H_MIN, H_MAX, NPTS = 1.0, 4.0, 121
DH = 1e-3  # step for chi_F overlap formula


# -----------------------------
# Build graph/hilbert
# -----------------------------
graph = Hypercube(length=L, n_dim=DIM, pbc=PBC)
hilbert = Spin(s=0.5, N=graph.n_nodes)
N = graph.n_nodes
DIMH = 1 << N


def H_full(h: float) -> sp.csr_matrix:
    return Ising(hilbert, graph, h=h, J=J).to_sparse().tocsr()


# -----------------------------
# Parity sector construction
# -----------------------------
# Global spin-flip in σ^z basis maps bitstring s -> ~s (bitwise complement)
COMP = (DIMH - 1) ^ np.arange(DIMH, dtype=np.int64)


def build_parity_U(sector: str) -> sp.csr_matrix:
    """
    Build U_{±} that maps sector basis -> full basis.

    Column basis vectors are:
      |s,±> = (|s> ± |~s>) / sqrt(2), using the representative s < ~s.
    """
    assert sector in {"+", "-"}
    reps = np.where(np.arange(DIMH) < COMP)[0]  # one representative per {s,~s}
    m = reps.size  # = 2^(N-1)

    rows = np.empty(2 * m, dtype=np.int64)
    cols = np.empty(2 * m, dtype=np.int64)
    data = np.empty(2 * m, dtype=np.float64)

    a = 1.0 / np.sqrt(2.0)
    sgn = +1.0 if sector == "+" else -1.0

    rows[0::2] = reps
    rows[1::2] = COMP[reps]
    cols[0::2] = np.arange(m)
    cols[1::2] = np.arange(m)
    data[0::2] = a
    data[1::2] = sgn * a

    U = sp.csr_matrix((data, (rows, cols)), shape=(DIMH, m))
    return U


U_PLUS = build_parity_U("+")
U_MINUS = build_parity_U("-")


def H_sector(H: sp.csr_matrix, sector: str) -> sp.csr_matrix:
    """
    Sector Hamiltonian: H_{±} = U_{±}^T H U_{±}.
    """
    U = U_PLUS if sector == "+" else U_MINUS
    return (U.T @ (H @ U)).tocsr()


def lowest_k_dense_sorted(Hs: sp.csr_matrix, k: int):
    vals, vecs = eigsh(Hs, k=k, which="SA", tol=1e-10, maxiter=20000)
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def chi_overlap(psi0: np.ndarray, psi1: np.ndarray, dh: float) -> float:
    F = np.abs(np.vdot(psi0, psi1))
    return float(2.0 * (1.0 - F) / (dh * dh))


# -----------------------------
# Main scan
# -----------------------------
def main():
    hs = np.linspace(H_MIN, H_MAX, NPTS)

    # Gaps
    gap_naive = np.zeros_like(hs)  # full-space E1 - E0 (dominated by even/odd doublet for small h)
    gap_even  = np.zeros_like(hs)  # even-sector E_{1,+} - E_{0,+}
    gap_eo    = np.zeros_like(hs)  # E_{0,-} - E_{0,+}

    # Fidelity susceptibility (even sector, gauge-safe overlap)
    chi_even = np.zeros_like(hs)

    for i, h in enumerate(tqdm(hs, desc="Scanning h")):
        H0 = H_full(h)

        # ---- naive full-space gap ----
        vals_full, _ = lowest_k_dense_sorted(H0, k=2)
        gap_naive[i] = float(vals_full[1] - vals_full[0])

        # ---- sector spectra ----
        Hp = H_sector(H0, "+")
        Hm = H_sector(H0, "-")

        vals_p, vecs_p = lowest_k_dense_sorted(Hp, k=2)   # need ground + 1st excited in even sector
        vals_m, _      = lowest_k_dense_sorted(Hm, k=1)   # need ground in odd sector

        E0p, E1p = float(vals_p[0]), float(vals_p[1])
        E0m = float(vals_m[0])

        gap_even[i] = E1p - E0p
        gap_eo[i] = E0m - E0p

        # ---- even-sector chi_F via overlap at h and h+DH ----
        H1 = H_full(h + DH)
        Hp1 = H_sector(H1, "+")
        vals_p1, vecs_p1 = lowest_k_dense_sorted(Hp1, k=1)

        psi0 = vecs_p[:, 0]
        psi1 = vecs_p1[:, 0]
        chi_even[i] = chi_overlap(psi0, psi1, DH)

    # Peak/min locations
    h_chi_peak = hs[np.argmax(chi_even)]
    h_gap_even_min = hs[np.argmin(gap_even)]
    print("\n==== Summary ====")
    print(f"even-sector chi_F peak at h ≈ {h_chi_peak:.6f}")
    print(f"even-sector gap min    at h ≈ {h_gap_even_min:.6f}")
    print("=================\n")

    # ---- plots ----
    plt.figure(figsize=(8, 5), dpi=120)
    plt.plot(hs, chi_even, label=r"$\chi_F$ (even sector, overlap)")
    plt.xlabel(r"Transverse field $h$")
    plt.ylabel(r"Fidelity susceptibility $\chi_F$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5), dpi=120)
    plt.plot(hs, gap_naive, label=r"gap naive $E_1-E_0$")
    plt.plot(hs, gap_even,  label=r"even gap $E_{1,+}-E_{0,+}$")
    plt.plot(hs, gap_eo,    label=r"even-odd $E_{0,-}-E_{0,+}$")
    plt.xlabel(r"Transverse field $h$")
    plt.ylabel("Gap")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- save ----
    out = np.column_stack([hs, chi_even, gap_naive, gap_even, gap_eo])
    np.savetxt(
        "tfim_4x4_parity_full.csv",
        out,
        delimiter=",",
        header="h,chi_even_overlap,gap_naive_E1-E0,gap_even_E1p-E0p,gap_even-odd_E0m-E0p",
        comments="",
    )
    print("Saved: tfim_4x4_parity_full.csv")


if __name__ == "__main__":
    main()
