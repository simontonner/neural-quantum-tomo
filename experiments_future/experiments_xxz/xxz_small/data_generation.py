from __future__ import annotations

from pathlib import Path

import sys
import time
import numpy as np

from netket.operator import LocalOperator, spin
from netket.hilbert import Spin
from netket.graph import Hypercube
from scipy.sparse.linalg import eigsh


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_handling import MultiQubitMeasurement, save_state_npz, save_measurements_npz



#### STATE CONSTRUCTION VIA NETKET ####

def build_xxz(L: int, delta: float, J: float, *, pbc: bool = True):
    graph = Hypercube(length=L, n_dim=2, pbc=pbc)
    hilbert = Spin(s=0.5, N=graph.n_nodes)

    # collect XY and ZZ parts separately
    H_xy = LocalOperator(hilbert, dtype=np.complex128)
    H_z  = LocalOperator(hilbert, dtype=np.complex128)

    for u, v in graph.edges():
        xx = spin.sigmax(hilbert, u) * spin.sigmax(hilbert, v)
        yy = spin.sigmay(hilbert, u) * spin.sigmay(hilbert, v)
        zz = spin.sigmaz(hilbert, u) * spin.sigmaz(hilbert, v)

        H_xy += xx + yy
        H_z  += zz

    ham = J * (delta * H_xy + H_z)

    return hilbert, graph, ham


def calculate_groundstate(H) -> tuple[float, np.ndarray]:
    start = time.time()
    evals, evecs = eigsh(H.to_sparse(), k=1, which="SA")  # smallest algebraic
    duration = time.time() - start
    energy = float(evals[0])
    psi = np.array(evecs[:, 0], dtype=np.complex128, copy=False)
    print(f"Diagonalization took {duration:.3f}s. Selected ground state at index 0 with energy {energy:.8f}.")
    return energy, psi


#### RUN SCRIPT ####

def main() -> None:

    # edit parameters here
    rng_seed = 42
    side_length = 2
    J = 1.00
    delta_values = [0.40, 0.60, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.40, 2.00]
    num_samples = 100_000

    out_meas = Path("measurements")
    out_states = Path("state_vectors")
    out_meas.mkdir(parents=True, exist_ok=True)
    out_states.mkdir(parents=True, exist_ok=True)

    print("=== SUMMARY ====================================================")
    print(f"2D XXZ Hamiltonian of size {side_length}x{side_length} via NetKet.")
    print(f"Coupling J is fixed to {J:.2f}; sweeping over {len(delta_values)} values of delta.")
    print("Ground-state is diagonalized exactly using scipy.")
    print("================================================================\n")

    rng = np.random.default_rng(rng_seed)

    print("=== BASIS CONSTRUCTION =========================================")
    nqubits = side_length * side_length
    print("Constructing measurement basis once since all measurements are in computational basis...")
    bases = ['Z'] * nqubits
    basis_name = "computational"
    meas = MultiQubitMeasurement(bases, show_tqdm=True)
    print("================================================================\n")

    for delta in delta_values:
        print(f"=== Creating data for delta={delta:+.2f} ==================================")
        hilbert, graph, H = build_xxz(L=side_length, delta=delta, J=J, pbc=True)
        energy, psi = calculate_groundstate(H)

        system_shape = f"{side_length}x{side_length}"
        system = f"XXZ_{system_shape}"

        state_header = {"system": system, "J": float(J), "delta": float(delta), "nqubits": int(nqubits), "seed": int(rng_seed)}
        state_path = out_states / f"xxz_{system_shape}_delta{delta:.2f}.npz"
        save_state_npz(state_path, psi, {"state": state_header})
        print(f"Saved ground-state to ./{state_path}")

        samples = meas.sample_state(psi, num_samples=num_samples, rng=rng)

        meas_header = {"basis": basis_name, "samples": int(num_samples), "seed": int(rng_seed)}
        meas_path = out_meas / f"xxz_{system_shape}_delta{delta:.2f}_{num_samples}.npz"
        save_measurements_npz(meas_path, samples, bases, {"state": state_header, "measurement": meas_header})

        print(f"Saved measurements to ./{meas_path}")
        print("================================================================\n")


if __name__ == "__main__":
    main()
