from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List

import time
import numpy as np

from tqdm import tqdm

from netket.operator import Ising
from netket.hilbert import Spin
from netket.graph import Hypercube
from scipy.sparse.linalg import eigsh


#### UTILITIES ####

def save_state_vector_columns(state: np.ndarray, file_path: str, header: str) -> None:
    with open(file_path, "w") as f:
        f.write(header)
        for c in state:
            re = float(np.real(c))
            im = float(np.imag(c))
            f.write(f"{re:.10f} {im:.10f}\n")


#### STATE CONSTRUCTION VIA NETKET ####

def build_tfim(L: int, h: float, J: float, *, pbc: bool = True):
    graph = Hypercube(length=L, n_dim=2, pbc=pbc)
    hilbert = Spin(s=0.5, N=graph.n_nodes)
    H = Ising(hilbert, graph, h=h, J=J)
    return hilbert, graph, H

def calculate_groundstate(H) -> tuple[float, np.ndarray]:
    start = time.time()
    # k=1 tells Lanczos solver stops once the lowest eigenpair has converged
    # SA: smallest algebraic puts ground state first
    evals, evecs = eigsh(H.to_sparse(), k=1, which='SA')
    duration = time.time() - start
    energy = float(evals[0])
    psi = np.array(evecs[:, 0], dtype=np.complex128, copy=False)
    print(f"Diagonalization took {duration:.3f}s. Selected ground state at index 0 with energy {energy:.8f}.")
    return energy, psi


#### RUN SCRIPT ####

def main() -> None:

    # edit parameters here
    rng_seed = 42
    side_length = 5
    J = -1.00
    h_values = [1.00, 2.00, 2.80, 3.00, 3.30, 3.60, 4.00, 5.00, 6.00, 7.00]

    nqubits = side_length * side_length
    out_meas = Path("measurements")
    out_states = Path("state_vectors")

    out_meas.mkdir(parents=True, exist_ok=True)
    out_states.mkdir(parents=True, exist_ok=True)

    print("=== SUMMARY ====================================================")
    print(f"2D TFIM Hamiltonian of size {side_length}x{side_length} via NetKet.")
    print(f"Fixed J={J:+.2f}; sweeping over {len(h_values)} values for h.")
    print(f"Ground-state is diagonalized exactly using scipy.")
    print("================================================================\n")

    for h in h_values:
        print(f"=== Creating data for h={h:+.2f} ==================================")
        hilbert, graph, H = build_tfim(L=side_length, h=h, J=J, pbc=True)
        energy, psi = calculate_groundstate(H)

        system_shape = f"{side_length}x{side_length}"
        system = f"TFIM_{system_shape}"
        state_header = f"STATE | system={system} | J={J:.2f} | h={h:.2f} | nqubits={nqubits} | seed={rng_seed}\n"
        state_path = out_states / f"tfim_{system_shape}_h{h:.2f}.txt"
        save_state_vector_columns(psi, str(state_path), header=state_header)
        print(f"Saved ground-state amplitudes to ./{state_path}")


if __name__ == "__main__":
    main()
