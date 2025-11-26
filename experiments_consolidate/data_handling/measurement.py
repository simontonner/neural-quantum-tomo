from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


#### UTILITIES ####

def int_to_bitstring(indices: np.ndarray, num_bits: int) -> np.ndarray:
    # integers -> MSB-first bitstrings
    indices = indices.astype(np.int64, copy=False)
    powers = (1 << np.arange(num_bits - 1, -1, -1, dtype=np.int64))
    bits = ((indices[:, None] & powers[None, :]) > 0).astype(np.uint8)
    return bits

def bitstring_to_filestring(bitstring: np.ndarray, measurement_basis: List[str]) -> str:
    out = []
    for bit, op in zip(bitstring.tolist(), measurement_basis):
        out.append(op.upper() if bit == 0 else op.lower() if bit == 1 else '?')
    return ''.join(out)


#### MULTI-QUBIT PAULI MEASUREMENT ####

@dataclass
class PauliMeasurement:
    eigenvectors: np.ndarray  # columns are eigenkets in Z basis

norm = 1.0 / np.sqrt(2.0)
pauli_i = PauliMeasurement(eigenvectors=np.eye(2, dtype=np.complex64))
pauli_z = PauliMeasurement(eigenvectors=np.eye(2, dtype=np.complex64))
pauli_x = PauliMeasurement(eigenvectors=norm * np.array([[1.0,  1.0], [1.0, -1.0]], dtype=np.complex64))
pauli_y = PauliMeasurement(eigenvectors=norm * np.array([[1.0,  1.0], [1.0j, -1.0j]], dtype=np.complex64))

PAULI_MAP = {"X": pauli_x, "Y": pauli_y, "Z": pauli_z, "I": pauli_i}

class MultiQubitMeasurement:
    def __init__(self, meas_dirs: List[str], verbose: bool = True):
        self.meas_dirs = meas_dirs
        self.N = len(meas_dirs)

        self.pauli_measurements: List[PauliMeasurement] = [PAULI_MAP[c] for c in meas_dirs]

        self._is_computational = all(c in ('Z', 'I') for c in meas_dirs)

    def sample_state(self, state_vec: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:

        if self._is_computational:
            probs = np.abs(state_vec)**2
        else:
            # copy the state and reshape to (2, 2, ..., 2) to access individual qubits
            psi = state_vec.reshape([2] * self.N).astype(np.complex64, copy=True)

            # apply local basis rotation on tensor state
            for idx, pauli in enumerate(self.pauli_measurements):
                # tensordot selects rotation matrix columns via axis 1 and applies them to the states local qubit
                local_rot = pauli.eigenvectors.conj().T
                psi = np.tensordot(local_rot, psi, axes=([1], [idx]))

                # tensordot moves the processed axis to index 0, but we want to move it back to idx
                psi = np.moveaxis(psi, 0, idx)

            # flatten tensor back to vector and compute probabilities
            psi = psi.flatten()
            probs = np.abs(psi)**2

        # normalize to handle potential floating point drift
        probs /= probs.sum()

        idx = rng.choice(probs.shape[0], size=num_samples, replace=True, p=probs)
        return int_to_bitstring(idx, self.N)