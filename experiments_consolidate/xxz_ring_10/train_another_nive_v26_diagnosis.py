import sys
import time
import numpy as np
import scipy.sparse.linalg
from pathlib import Path
from netket.graph import Hypercube
from netket.hilbert import Spin
from netket.operator import GraphOperator, LocalOperator, spin

# ==========================================
# YOUR EXACT GENERATION CODE
# ==========================================
def build_xxz(L: int, delta: float, *, pbc: bool = True, marshall_sign_rule: bool = True, epsilon: float = 1e-3):
    graph = Hypercube(length=L, n_dim=1, pbc=pbc)
    hilbert = Spin(s=0.5, N=graph.n_nodes)

    sx = spin.sigmax
    sy = spin.sigmay
    sz = spin.sigmaz

    bond_hilbert = Spin(s=0.5, N=2)
    xy_bond_op = sx(bond_hilbert, 0)*sx(bond_hilbert, 1) + sy(bond_hilbert, 0)*sy(bond_hilbert, 1)
    zz_bond_op = sz(bond_hilbert, 0)*sz(bond_hilbert, 1)

    xy_coeff = -delta if marshall_sign_rule else delta
    bond_op = xy_coeff * xy_bond_op + zz_bond_op
    bond_matrix = bond_op.to_dense()

    H_xxz = GraphOperator(hilbert, graph=graph, bond_ops=[bond_matrix])

    H_field = LocalOperator(hilbert, dtype=np.complex128)
    for i in range(graph.n_nodes):
        H_field -= epsilon * sz(hilbert, i)

    return hilbert, graph, H_xxz + H_field

def calculate_groundstate(hamiltonian):
    sp_mat = hamiltonian.to_sparse()
    # Use 'SA' (Smallest Algebraic) to find ground state
    evals, evecs = scipy.sparse.linalg.eigsh(sp_mat, k=1, which="SA")
    psi = evecs[:, 0]
    # Enforce Marshall Sign Rule (Positive Amplitudes) correction if needed
    # But note: The Hamiltonian construction ALREADY handles the Marshall sign flip on the operator.
    # So the resulting eigenvector should effectively be positive.
    first_idx = np.argmax(np.abs(psi))
    phase = np.angle(psi[first_idx])
    psi = psi * np.exp(-1j * phase)
    return float(evals[0]), psi

# ==========================================
# DIAGNOSTIC ROUTINE
# ==========================================
if __name__ == "__main__":
    L = 10
    DELTA = 1.00
    EPSILON = 1e-3

    print(f"--- DIAGNOSING GROUND TRUTH (L={L}, Delta={DELTA}, eps={EPSILON}) ---")

    # 1. Generate Hamiltonian & State
    hilbert, graph, H = build_xxz(L, DELTA, epsilon=EPSILON)
    E, psi = calculate_groundstate(H)

    # Psi is a vector of size 2^L
    # Probabilities
    probs = np.abs(psi)**2

    # 2. Generate all basis states (bitstrings)
    # Shape: [2^L, L]
    indices = np.arange(2**L)
    # Little trick to get bitstrings as (0, 1) integers
    states = ((indices[:, None] & (1 << np.arange(L-1, -1, -1))) > 0).astype(int)

    # 3. Calculate Observables
    # Spin values: 0 -> -1, 1 -> +1
    spins = 2 * states - 1

    # A) Magnetization
    # M = Sum(sigma_z)
    m_per_state = np.sum(spins, axis=1)
    # Expectation value <M>
    avg_mag = np.sum(probs * m_per_state)
    avg_mag_sq = np.sum(probs * (m_per_state**2))

    # B) Correlation Czz
    # Neighbor pairs with PBC
    czz_sum = 0
    for i in range(L):
        j = (i + 1) % L
        corr = spins[:, i] * spins[:, j]
        czz_sum += np.sum(probs * corr)
    avg_czz = czz_sum / L

    # C) Entropy S2 (L=5 partition)
    # Reshape vector into matrix [2^5, 2^5]
    dim_half = 2**(L//2)
    psi_matrix = psi.reshape((dim_half, dim_half))
    # SVD
    U, S, Vh = np.linalg.svd(psi_matrix)
    # S2 = -ln(Sum lambda^4)
    s2 = -np.log(np.sum(S**4))

    print("\n=== EXACT GROUND TRUTH PROPERTIES ===")
    print(f"Energy:      {E:.6f}")
    print(f"Magnetization <M>:   {avg_mag:.6f}  (Should be 0.0 for symmetry)")
    print(f"Fluctuations <M^2>:  {avg_mag_sq:.6f}  (Should be 0.0 for strict conservation)")
    print(f"Correlation C_zz:    {avg_czz:.6f}")
    print(f"Entanglement S2:     {s2:.6f}")

    print("\n--- CHECKING SIGN STRUCTURE ---")
    # If Marshall rule works, all amplitudes should be real and have the same sign (after rotation)
    # Since we rotated psi to make the largest element real positive, let's check the others.
    n_negative = np.sum(psi.real < -1e-10)
    print(f"Negative Amplitudes: {n_negative} / {2**L}")
    if n_negative > 0:
        print("WARNING: Wavefunction is NOT purely positive. Standard RBM cannot learn this.")
    else:
        print("OK: Wavefunction is positive.")