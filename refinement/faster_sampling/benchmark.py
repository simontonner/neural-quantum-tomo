import time
import numpy as np
from dataclasses import dataclass
from typing import List

# ==========================================
# 1. FAST APPROACH (Global Probability)
# ==========================================
# This calculates the full probability vector ONCE, then samples from it.

@dataclass
class PauliMeasurement:
    eigenvectors: np.ndarray

norm = 1.0 / np.sqrt(2.0)
pauli_i = PauliMeasurement(eigenvectors=np.eye(2, dtype=np.complex64))
pauli_z = PauliMeasurement(eigenvectors=np.eye(2, dtype=np.complex64))
pauli_x = PauliMeasurement(eigenvectors=norm * np.array([[1.0,  1.0], [1.0, -1.0]], dtype=np.complex64))
pauli_y = PauliMeasurement(eigenvectors=norm * np.array([[1.0,  1.0], [1.0j, -1.0j]], dtype=np.complex64))
PAULI_MAP = {"X": pauli_x, "Y": pauli_y, "Z": pauli_z, "I": pauli_i}

class Method_Global_Calc:
    def __init__(self, meas_dirs: List[str]):
        self.meas_dirs = meas_dirs
        self.N = len(meas_dirs)
        self.pauli_measurements = [PAULI_MAP[c] for c in meas_dirs]
        self._is_computational = all(c in ('Z', 'I') for c in meas_dirs)

    def sample(self, state_vec: np.ndarray, num_samples: int) -> None:
        # 1. Rotate state
        if self._is_computational:
            probs = np.abs(state_vec)**2
        else:
            psi = state_vec.reshape([2] * self.N).astype(np.complex64, copy=True)
            for idx, pauli in enumerate(self.pauli_measurements):
                local_rot = pauli.eigenvectors.conj().T
                psi = np.tensordot(local_rot, psi, axes=([1], [idx]))
                psi = np.moveaxis(psi, 0, idx)
            probs = np.abs(psi.flatten())**2

        # 2. Normalize
        probs /= probs.sum()

        # 3. Sample (Fast NumPy choice)
        rng = np.random.default_rng()
        _ = rng.choice(probs.shape[0], size=num_samples, replace=True, p=probs)
        # Note: We skip converting to bitstrings to keep the benchmark focused on math speed


# ==========================================
# 2. SLOW APPROACH (Sequential Collapse)
# ==========================================
# This simulates the physical collapse of the wavefunction for EVERY sample.

def measure_qubit(plus_slice, minus_slice, op, rng):
    if op == 'I': return 0

    INV_SQRT2 = 1.0 / np.sqrt(2.0)
    plus_slice_tmp = plus_slice.copy()
    minus_slice_tmp = minus_slice.copy()

    # Z basis defaults
    plus_ampl = plus_slice_tmp
    minus_ampl = minus_slice_tmp
    plus_eigvec = (1.0, 0.0)
    minus_eigvec = (0.0, 1.0)

    if op == 'X':
        plus_ampl = (plus_slice_tmp + minus_slice_tmp) * INV_SQRT2
        minus_ampl = (plus_slice_tmp - minus_slice_tmp) * INV_SQRT2
        plus_eigvec = (INV_SQRT2, INV_SQRT2)
        minus_eigvec = (INV_SQRT2, -INV_SQRT2)
    elif op == 'Y':
        plus_ampl = (plus_slice_tmp - 1j * minus_slice_tmp) * INV_SQRT2
        minus_ampl = (plus_slice_tmp + 1j * minus_slice_tmp) * INV_SQRT2
        plus_eigvec = (INV_SQRT2, 1j * INV_SQRT2)
        minus_eigvec = (INV_SQRT2, -1j * INV_SQRT2)

    plus_prob = np.sum(np.abs(plus_ampl)**2)
    plus_prob = np.clip(plus_prob, 0.0, 1.0)
    plus_meas = rng.random() < plus_prob

    chosen_ampl = (plus_ampl if plus_meas else minus_ampl)
    chosen_eigvec = (plus_eigvec if plus_meas else minus_eigvec)

    # Calculate renorm factor
    prob = plus_prob if plus_meas else (1.0 - plus_prob)
    renorm_factor = np.sqrt(prob) if prob > 1e-15 else 1.0

    # COLLAPSE THE STATE (Write back to memory)
    if prob < 1e-15:
        plus_slice[:] = 0.0
        minus_slice[:] = 0.0
    else:
        plus_slice[:] = (chosen_ampl * chosen_eigvec[0]) / renorm_factor
        minus_slice[:] = (chosen_ampl * chosen_eigvec[1]) / renorm_factor

    return 0 if plus_meas else 1

def run_sequential_collapse(state_vec, pauli_ops, num_samples):
    rng = np.random.default_rng()
    num_qubits = len(pauli_ops)

    # LOOP FOR EVERY SAMPLE
    for _ in range(num_samples):
        # COPY THE FULL STATE (Expensive!)
        state_copy = state_vec.copy()
        state_tensor = state_copy.reshape((2,) * num_qubits)

        # Loop over qubits
        for qubit_idx, pauli_op in enumerate(pauli_ops):
            plus_slice  = state_tensor[(slice(None),) * qubit_idx + (0,)]
            minus_slice = state_tensor[(slice(None),) * qubit_idx + (1,)]
            measure_qubit(plus_slice, minus_slice, pauli_op, rng)


# ==========================================
# BENCHMARK
# ==========================================

def run_benchmark():
    print("="*60)
    print("SPEED TEST: Global Calculation vs Sequential Collapse")
    print("="*60)

    # Settings
    N_QUBITS = 12
    SAMPLES = 100  # Keep this low, or the Sequential method will take forever
    MEAS_STR = ["X", "Y", "Z"] * (N_QUBITS // 3)

    print(f"Qubits:  {N_QUBITS}")
    print(f"Samples: {SAMPLES}")
    print("-" * 60)

    # Prepare State
    dim = 2**N_QUBITS
    rng = np.random.default_rng(42)
    state = rng.random(dim) + 1j * rng.random(dim)
    state /= np.linalg.norm(state)

    # --- METHOD 1: GLOBAL ---
    print("Running GLOBAL PROBABILITY method...")
    method_1 = Method_Global_Calc(MEAS_STR)

    t0 = time.perf_counter()
    method_1.sample(state, SAMPLES)
    t1 = time.perf_counter()
    time_global = t1 - t0
    print(f"   Time: {time_global:.5f} s")

    print("-" * 60)

    # --- METHOD 2: SEQUENTIAL ---
    print("Running SEQUENTIAL COLLAPSE method...")

    t0 = time.perf_counter()
    run_sequential_collapse(state, MEAS_STR, SAMPLES)
    t1 = time.perf_counter()
    time_seq = t1 - t0
    print(f"   Time: {time_seq:.5f} s")

    print("-" * 60)
    print(f"Winner: {'GLOBAL' if time_global < time_seq else 'SEQUENTIAL'}")
    print(f"Factor: {time_seq / time_global:.1f}x faster")
    print("="*60)

if __name__ == "__main__":
    run_benchmark()