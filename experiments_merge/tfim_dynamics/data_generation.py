from __future__ import annotations

from pathlib import Path

import sys
import time
import numpy as np

from netket.operator import Ising
from netket.hilbert import Spin
from netket.graph import Hypercube
from scipy.sparse.linalg import eigsh


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_handling import MultiQubitMeasurement, save_state_txt, save_measurements_txt



#### STATE CONSTRUCTION VIA NETKET ####

def build_tfim(L: int, h: float, J: float, *, pbc: bool = True):
    graph = Hypercube(length=L, n_dim=2, pbc=pbc)
    hilbert = Spin(s=0.5, N=graph.n_nodes)
    H = Ising(hilbert, graph, h=h, J=J)
    return hilbert, graph, H

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
    side_length = 3
    J = -1.00
    h_values = [1.00, 2.00, 2.80, 3.00, 3.30, 3.60, 4.00, 5.00, 6.00, 7.00]
    num_samples = 10_000

    out_meas = Path("measurements")
    out_states = Path("state_vectors")
    out_meas.mkdir(parents=True, exist_ok=True)
    out_states.mkdir(parents=True, exist_ok=True)

    print("=== SUMMARY ====================================================")
    print(f"2D TFIM Hamiltonian of size {side_length}x{side_length} via NetKet.")
    print(f"Fixed J={J:.2f}; sweeping over {len(h_values)} values for h.")
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

    for h in h_values:
        print(f"=== Creating data for h={h:+.2f} ==================================")
        hilbert, graph, H = build_tfim(L=side_length, h=h, J=J, pbc=True)
        energy, psi = calculate_groundstate(H)

        system_shape = f"{side_length}x{side_length}"
        system = f"TFIM_{system_shape}"

        state_header = {"system": system, "J": float(J), "h": float(h), "nqubits": int(nqubits), "seed": int(rng_seed)}
        state_path = out_states / f"tfim_{system_shape}_h{h:.2f}.txt"
        save_state_txt(state_path, psi, {"state": state_header})
        print(f"Saved ground-state to ./{state_path}")

        samples = meas.sample_state(psi, num_samples=num_samples, rng=rng)

        meas_header = {"basis": basis_name, "samples": int(num_samples), "seed": int(rng_seed)}
        meas_path = out_meas / f"tfim_{system_shape}_h{h:.2f}_{num_samples}.txt"
        save_measurements_txt(meas_path, samples, bases, {"state": state_header, "measurement": meas_header})

        print(f"Saved measurements to ./{meas_path}")
        print("================================================================\n")


if __name__ == "__main__":
    main()
