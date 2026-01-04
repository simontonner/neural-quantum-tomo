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

# Assuming these exist in your environment as per your previous code
from data_handling import MultiQubitMeasurement, save_state_txt, save_measurements_txt

#### STATE CONSTRUCTION VIA NETKET ####

def build_1d_tfim(N: int, h: float, J: float, pbc: bool = False):
    """
    Constructs 1D TFIM.
    Note: pbc=False corresponds to an Open Boundary Chain.
    """
    # CHANGE 1: n_dim=1 for a 1D Chain
    graph = Hypercube(length=N, n_dim=1, pbc=pbc)
    hilbert = Spin(s=0.5, N=graph.n_nodes)

    # NetKet Ising H = -J * sum(ZZ) - h * sum(X)
    # To match Paper's Ferromagnetic setup, we use positive J here
    # so the interaction term becomes negative (-1 * ZZ).
    H = Ising(hilbert, graph, h=h, J=J)
    return hilbert, graph, H

def calculate_groundstate(H) -> tuple[float, np.ndarray]:
    start = time.time()
    # For N=16, size is 65536. eigsh is fast enough.
    evals, evecs = eigsh(H.to_sparse(), k=1, which="SA")
    duration = time.time() - start
    energy = float(evals[0])
    psi = np.array(evecs[:, 0], dtype=np.complex128, copy=False)
    print(f"Diagonalization took {duration:.3f}s. Energy {energy:.8f}.")
    return energy, psi


#### RUN SCRIPT ####

def main() -> None:
    # === EXPERIMENT PARAMETERS ===
    rng_seed = 42

    # CHANGE 2: System Size N=16
    num_qubits = 16

    # CHANGE 3: Interaction J
    # J=1.0 in NetKet => -1.0 * ZZ interaction (Ferromagnetic)
    J = 1.00

    # CHANGE 4: h values from Figure 3 of the PDF
    # 0.8 (Ordered), 1.0 (Critical), 1.2 (Disordered)
    h_values = [0.8, 1.0, 1.2]

    num_samples = 10_000 # 10k is usually sufficient for N=16

    # Folders
    out_meas = Path("measurements")
    out_states = Path("state_vectors")
    out_meas.mkdir(parents=True, exist_ok=True)
    out_states.mkdir(parents=True, exist_ok=True)

    print("=== SUMMARY ====================================================")
    print(f"1D TFIM Chain with {num_qubits} qubits.")
    print(f"J={J:.2f} (Ferromagnetic); sweeping {h_values}.")
    print("Ground-state is real and positive (Stoquastic).")
    print("================================================================\n")

    rng = np.random.default_rng(rng_seed)

    print("=== BASIS CONSTRUCTION =========================================")
    # The paper states only Sigma^z measurements are needed for ground states
    bases = ['Z'] * num_qubits
    basis_name = "computational"
    meas = MultiQubitMeasurement(bases, show_tqdm=True)
    print("================================================================\n")

    for h in h_values:
        print(f"=== Creating data for h={h:+.2f} ==================================")

        # CHANGE 5: Use Open Boundary Conditions (pbc=False) for a "Chain"
        # Use pbc=True if you want a Ring. Figure 3 usually implies OBC or very large PBC.
        hilbert, graph, H = build_1d_tfim(N=num_qubits, h=h, J=J, pbc=True)


        energy, psi = calculate_groundstate(H)

        # Ensure wavefunction is real/positive (fixes global phase ambiguity from eigsh)
        # The paper relies on psi being real/positive.
        # Generally TFIM eigenvectors are real, but eigsh might add a -1 factor.
        # We force the first non-zero element to be positive to align phases.
        first_nonzero = np.flatnonzero(np.abs(psi) > 1e-10)[0]
        phase_factor = np.angle(psi[first_nonzero])
        psi = psi * np.exp(-1j * phase_factor)
        # Now it should be real, let's enforce strict realness to be clean
        psi = np.abs(psi)

        system_shape = f"1x{num_qubits}"

        state_header = {
            "system": "TFIM_1D_Chain",
            "J": float(J),
            "h": float(h),
            "nqubits": int(num_qubits),
            "seed": int(rng_seed)
        }

        state_path = out_states / f"tfim_1d_N{num_qubits}_h{h:.2f}.txt"
        save_state_txt(state_path, psi, {"state": state_header})
        print(f"Saved ground-state to ./{state_path}")

        # Sampling
        samples = meas.sample_state(psi, num_samples=num_samples, rng=rng)

        meas_header = {
            "basis": basis_name,
            "samples": int(num_samples),
            "seed": int(rng_seed)
        }

        meas_path = out_meas / f"tfim_1d_N{num_qubits}_h{h:.2f}_{num_samples}.txt"
        save_measurements_txt(meas_path, samples, bases, {"state": state_header, "measurement": meas_header})

        print(f"Saved measurements to ./{meas_path}")
        print("================================================================\n")

if __name__ == "__main__":
    main()