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

from data_handling import MultiQubitMeasurement, save_state_npz, save_measurements_npz


#### STATE CONSTRUCTION VIA NETKET ####

def build_tfim(L: int, h: float, J: float, *, pbc: bool = True):
    graph = Hypercube(length=L, n_dim=2, pbc=pbc)
    hilbert = Spin(s=0.5, N=graph.n_nodes)
    H = Ising(hilbert, graph, h=h, J=J)
    return hilbert, graph, H

def calculate_groundstate(hamiltonian) -> tuple[float, np.ndarray]:
    start = time.time()

    sp_mat = hamiltonian.to_sparse()
    evals, evecs = eigsh(sp_mat, k=1, which="SA")

    duration = time.time() - start

    psi = evecs[:, 0]
    energy = float(evals[0])

    print(f"  Diagonalization took {duration:.3f}s. Energy: {energy:.6f}")

    # Fix phase
    first_idx = np.argmax(np.abs(psi))
    phase = np.angle(psi[first_idx])
    psi = psi * np.exp(-1j * phase)

    return energy, psi


#### RUN SCRIPT ####

if __name__ == "__main__":

    # --- PARAMETERS ---
    rng_seed = 42
    side_length = 4
    periodic_boundaries = True
    J = -1.00
    num_samples = 5_000_000

    # 1. THE "BACKBONE" (Support) -> We need measurements for these to TRAIN
    h_support = [2.00, 2.50, 3.20, 4.00]

    # 2. THE "INTERPOLATION" (Novel) -> We only need states for these to TEST FIDELITY
    h_novel   = [2.20, 2.40, 2.60, 2.80, 3.60]

    # Combine and sort for the loop
    all_h_values = sorted(list(set(h_support + h_novel)))

    # ------------------

    out_meas = Path("measurements")
    out_states = Path("state_vectors")
    out_meas.mkdir(parents=True, exist_ok=True)
    out_states.mkdir(parents=True, exist_ok=True)

    print("=== SUMMARY ====================================================")
    print(f"2D TFIM Hamiltonian of size {side_length}x{side_length} via NetKet.")
    print(f"Support (Training): {h_support}")
    print(f"Novel   (Testing) : {h_novel}")
    print(f"Total Points      : {len(all_h_values)}")
    print("================================================================\n")

    rng = np.random.default_rng(rng_seed)

    print("=== BASIS CONSTRUCTION =========================================")
    nqubits = side_length * side_length
    bases = ['Z'] * nqubits
    basis_name = "computational"
    meas = MultiQubitMeasurement(bases, verbose=False) # Suppress verbose basis init
    print(f"Measurement operator prepared for {nqubits} qubits.")
    print("================================================================\n")

    for h in all_h_values:
        is_support = h in h_support
        role = "SUPPORT" if is_support else "NOVEL"

        print(f"--- Processing h={h:+.2f} [{role}] ---------------------------")

        # 1. Calculate and Save Exact State (Needed for ALL points for Fidelity)
        hilbert, graph, H = build_tfim(L=side_length, h=h, J=J, pbc=periodic_boundaries)
        energy, psi = calculate_groundstate(H)

        system_shape = f"{side_length}x{side_length}"
        system = f"TFIM_{system_shape}"

        state_header = {"system": system, "J": float(J), "h": float(h), "nqubits": int(nqubits), "seed": int(rng_seed)}
        state_path = out_states / f"tfim_{system_shape}_h{h:.2f}.npz"
        save_state_npz(state_path, psi, {"state": state_header})
        print(f"  [State] Saved to {state_path}")

        # 2. Generate Measurements (ONLY for Support points)
        if is_support:
            samples = meas.sample_state(psi, num_samples=num_samples, rng=rng)

            meas_header = {"basis": basis_name, "samples": int(num_samples), "seed": int(rng_seed)}
            meas_path = out_meas / f"tfim_{system_shape}_h{h:.2f}_{num_samples}.npz"
            save_measurements_npz(meas_path, samples, bases, {"state": state_header, "measurement": meas_header})

            print(f"  [Meas]  Generated {num_samples} samples.")
            print(f"  [Meas]  Saved to {meas_path}")
        else:
            print(f"  [Meas]  Skipping measurement generation (Novel point).")

        print("")