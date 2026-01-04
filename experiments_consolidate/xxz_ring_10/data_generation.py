from __future__ import annotations

from pathlib import Path

import sys
import time
import numpy as np

from netket.operator import LocalOperator, GraphOperator, spin
from netket.hilbert import Spin
from netket.graph import Hypercube
from scipy.sparse.linalg import eigsh


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_handling import MultiQubitMeasurement, save_state_npz, save_measurements_npz


#### STATE CONSTRUCTION VIA NETKET ####

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


def calculate_groundstate(hamiltonian) -> tuple[float, np.ndarray]:
    start = time.time()

    sp_mat = hamiltonian.to_sparse()
    evals, evecs = eigsh(sp_mat, k=1, which="SA")

    psi = evecs[:, 0]
    energy = float(evals[0])

    first_idx = np.argmax(np.abs(psi))
    phase = np.angle(psi[first_idx])
    psi = psi * np.exp(-1j * phase)

    duration = time.time() - start
    print(f"Diagonalization took {duration:.3f}s. Selected ground state at index 0 with energy {energy:.8f}.")

    return energy, psi


#### RUN SCRIPT ####

if __name__ == "__main__":

    # edit parameters here
    rng_seed = 42
    chain_length = 10
    periodic_boundaries = True
    marshall_sign_rule = True
    delta_values = [0.50, 1.00, 2.00]
    num_samples = 5_000_000

    out_meas = Path("measurements")
    out_states = Path("state_vectors")
    out_meas.mkdir(parents=True, exist_ok=True)
    out_states.mkdir(parents=True, exist_ok=True)

    print("=== SUMMARY ====================================================")
    print(f"2D XXZ Hamiltonian of size {chain_length} via NetKet.")
    print(f"PBC: {periodic_boundaries}, Marshall: {marshall_sign_rule}.")
    print(f"Sweeping over {len(delta_values)} values of delta.")
    print("Ground-state is diagonalized exactly using scipy.")
    print("================================================================\n")

    rng = np.random.default_rng(rng_seed)

    print("=== BASIS CONSTRUCTION =========================================")
    nqubits = chain_length
    print("Constructing measurement basis once since all measurements are in computational basis...")
    bases = ['Z'] * nqubits
    basis_name = "computational"
    meas = MultiQubitMeasurement(bases, verbose=True)
    print("================================================================\n")

    for delta in delta_values:
        print(f"=== Creating data for delta={delta:+.2f} ==================================")
        hilbert, graph, H = build_xxz(L=chain_length, delta=delta, pbc=periodic_boundaries, marshall_sign_rule=marshall_sign_rule)
        energy, psi = calculate_groundstate(H)

        system_shape = f"{chain_length}"
        system = f"XXZ_{system_shape}"

        state_header = {"system": system, "delta": float(delta), "nqubits": int(nqubits), "seed": int(rng_seed)}
        state_path = out_states / f"xxz_{system_shape}_delta{delta:.2f}.npz"
        save_state_npz(state_path, psi, {"state": state_header})
        print(f"Saved ground-state to ./{state_path}")

        samples = meas.sample_state(psi, num_samples=num_samples, rng=rng)

        meas_header = {"basis": basis_name, "samples": int(num_samples), "seed": int(rng_seed)}
        meas_path = out_meas / f"xxz_{system_shape}_delta{delta:.2f}_{num_samples}.npz"
        save_measurements_npz(meas_path, samples, bases, {"state": state_header, "measurement": meas_header})

        print(f"Saved measurements to ./{meas_path}")
        print("================================================================\n")