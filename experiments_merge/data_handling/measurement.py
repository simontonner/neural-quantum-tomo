from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import List

import numpy as np

from tqdm import tqdm


#### UTILITIES ####

def int_to_bitstring(indices: np.ndarray, num_bits: int) -> np.ndarray:
    """Integer indices -> MSB-first bitstrings, shape (len(indices), num_bits)"""
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
    """
    General tensor-product projective measurement in arbitrary local Pauli directions.
    """
    def __init__(self, meas_dirs: List[str], *, show_tqdm: bool = True):
        self.meas_dirs = meas_dirs
        self.N = len(meas_dirs)

        try:
            self.pauli_measurements: List[PauliMeasurement] = [PAULI_MAP[c] for c in meas_dirs]
        except KeyError as e:
            raise ValueError(f"Unknown measurement axis '{e.args[0]}'; use only X/Y/Z/I.") from None

        # Z or I both mean computational-basis eigenkets; fast path if no X/Y present
        self._all_ZI = all(c in ('Z', 'I') for c in meas_dirs)
        self.basis_vecs: List[np.ndarray] | None = None

        if not self._all_ZI:
            self.basis_vecs = self._construct_measurement_basis(show_tqdm=show_tqdm)

    def _construct_measurement_basis(self, *, show_tqdm: bool) -> List[np.ndarray]:
        measurement_basis_vectors: List[np.ndarray] = []
        outcome_bitstrings = list(product([0, 1], repeat=self.N))
        meas_dirs_str = ''.join(self.meas_dirs)
        iterator = tqdm(outcome_bitstrings, desc=f"Constructing basis {meas_dirs_str}", disable=not show_tqdm)
        for outcome_bits in iterator:
            vec = None
            for P, b in zip(self.pauli_measurements, outcome_bits):
                col = P.eigenvectors[:, b]
                vec = col if vec is None else np.kron(vec, col)
            measurement_basis_vectors.append(vec.astype(np.complex64, copy=False))
        return measurement_basis_vectors

    def sample_state(self, state_vec: np.ndarray, num_samples: int = 1000,
                     rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng

        if self._all_ZI:
            probs = np.abs(state_vec)**2
        else:
            assert self.basis_vecs is not None
            probs = np.array([np.abs(np.vdot(v, state_vec))**2 for v in self.basis_vecs], dtype=np.float64)
            probs /= probs.sum()

        idx = rng.choice(probs.shape[0], size=num_samples, replace=True, p=probs)
        return int_to_bitstring(idx, self.N)