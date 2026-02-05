from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_handling import MultiQubitMeasurement, save_state_txt, save_measurements_txt


#### STATE AND BASIS CONSTRUCTION ####

def generate_phase_augmented_w_state(num_qubits: int, rng: np.random.Generator) -> np.ndarray:

    state_dim = 1 << num_qubits

    thetas = rng.uniform(0.0, 2.0 * np.pi, size=(num_qubits,))
    amps = np.exp(1j * thetas) / np.sqrt(num_qubits)

    state = np.zeros(state_dim, dtype=np.complex128)
    one_hot_indices = (1 << np.arange(num_qubits - 1, -1, -1, dtype=np.int64))
    state[one_hot_indices] = amps.astype(np.complex128, copy=False)

    return state


def sliding_window_bases(num_qubits: int, background: str, window: str, stride: int = 1) -> List[List[str]]:
    w = list(window)
    L = len(w)
    if L == 0 or L > num_qubits or stride <= 0:
        return []
    bases: list[list[str]] = []
    for i in range(0, num_qubits - L + 1, stride):
        b = [background] * num_qubits
        b[i:i + L] = w
        bases.append(b)
    return bases


#### RUN SCRIPT ####


if __name__ == "__main__":

    # edit parameters here
    rng_seed = 43
    num_qubits = 4
    samples_per_basis = 5000

    rng = np.random.default_rng(rng_seed)

    out_meas = Path("measurements")
    out_states = Path("state_vectors")
    out_meas.mkdir(parents=True, exist_ok=True)
    out_states.mkdir(parents=True, exist_ok=True)

    psi = generate_phase_augmented_w_state(num_qubits, rng)

    state_header = {"system": "W_phase", "nqubits": int(num_qubits), "seed": int(rng_seed)}
    state_path = out_states / "w_phase_state.txt"
    save_state_txt(state_path, psi, { "state": state_header })
    num_amplitudes = int(psi.shape[0])
    print(f"Saved {num_amplitudes} amplitudes (Re, Im) to {state_path}.\n")

    print("We compute complex amplitudes as inner products between the state and each measurement basis vector.")
    print("Measurements are sampled from the squared magnitudes of these amplitudes.\n")


    # building up bases. first ZZZZ then sliding windows
    background = "Z"
    window_1, window_2 = "XX", "XY"
    measurement_bases = [[background] * num_qubits]
    measurement_bases += sliding_window_bases(num_qubits, background, window_1)
    measurement_bases += sliding_window_bases(num_qubits, background, window_2)

    print(f"Auxiliary bases via sliding windows '{window_1}', '{window_2}' over '{background}' background.")
    print(f"Sampling {samples_per_basis} shots per basis across {len(measurement_bases)} bases...")


    for meas_dirs in measurement_bases:
        basis_code = ''.join(meas_dirs)
        measurement = MultiQubitMeasurement(meas_dirs, show_tqdm=True)

        samples = measurement.sample_state(state_vec=psi, num_samples=samples_per_basis, rng=rng)

        meas_header = { "basis": basis_code, "samples": int(samples_per_basis), "seed": int(rng_seed) }
        meas_path = out_meas / f"w_phase_{basis_code}_{samples_per_basis}.txt"
        save_measurements_txt(meas_path, samples, meas_dirs, { "state": state_header, "measurement": meas_header })

    print(f"Saved {len(measurement_bases)} files to ./{out_meas} as w_phase_<BASIS>_{samples_per_basis}.txt")