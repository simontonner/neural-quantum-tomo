from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List

import time
import json
import numpy as np

from tqdm import tqdm

from netket.operator import Ising
from netket.hilbert import Spin
from netket.graph import Hypercube
from scipy.sparse.linalg import eigsh



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


#### NPZ SAVE HELPERS ####

_BASIS_MAP = {"X": 0, "Y": 1, "Z": 2, "I": 3}

def save_state_npz(file_path: Path, amplitudes: np.ndarray, headers: dict[str, dict]) -> None:

    compact_amplitudes = amplitudes.astype(np.complex64, copy=False)
    header_tuples = {name: json.dumps(payload) for name, payload in headers.items()}

    np.savez_compressed(file_path, amplitudes=compact_amplitudes, **header_tuples)

def save_measurements_npz(file_path: Path, values: np.ndarray, bases: list[str], headers: dict[str, dict]) -> None:

    packed_values = np.packbits(np.asarray(values, dtype=np.uint8), axis=1)
    encoded_bases = np.array([_BASIS_MAP[b] for b in bases], dtype=np.uint8) # only for homogen basis
    header_tuples = {name: json.dumps(payload) for name, payload in headers.items()}

    np.savez_compressed(file_path, values=packed_values, bases=encoded_bases, **header_tuples)


#### TXT SAVE HELPERS ####

def format_header(name: str, fields: dict) -> str:
    label = name.upper()
    parts = [label]
    for k, v in fields.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.2f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts) + "\n"

def save_state_txt(file_path: Path, amplitudes: np.ndarray, headers: dict[str, dict]) -> None:
    with open(file_path, "w") as f:
        for name, fields in headers.items():
            f.write(format_header(name, fields))

        for c in amplitudes:
            re = float(np.real(c))
            im = float(np.imag(c))
            f.write(f"{re:.10f} {im:.10f}\n")

def save_measurements_txt(file_path: Path, values: np.ndarray, bases: list[str], headers: dict[str, dict]) -> None:
    values = np.asarray(values, dtype=np.uint8)
    with open(file_path, "w") as f:
        for name, fields in headers.items():
            f.write(format_header(name, fields))

        for row in values:
            encoded_measurements = [op.upper() if bit == 0 else op.lower() for bit, op in zip(row, bases)]
            f.write("".join(encoded_measurements) + "\n")


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
