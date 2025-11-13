#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-augmented W-state generation + measurement sampling

- Creates:
  1) measurements/w_phase_state.txt
  2) Per-basis files under measurements/: w_phase_<BASIS>_<shots>.txt
- Logs match the requested template (without unique/values/bases files), plus one final summary line.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax.random import PRNGKey

from tqdm import tqdm  # assumed available


# =========================
# General utilities
# =========================

def int_to_bitstring(indices: jnp.ndarray, num_bits: int) -> jnp.ndarray:
    indices = indices.astype(jnp.int32)
    powers = 2 ** jnp.arange(num_bits - 1, -1, -1, dtype=jnp.int32)
    bits = (indices[..., None] & powers) > 0
    return bits.astype(jnp.uint8)

def bitstring_to_filestring(bitstring: jnp.ndarray, measurement_basis: List[str]) -> str:
    out = []
    for bit, op in zip([int(b) for b in bitstring], measurement_basis):
        out.append(op.upper() if bit == 0 else op.lower() if bit == 1 else '?')
    return ''.join(out)

def save_state_vector_columns(state: jnp.ndarray, file_path: str) -> None:
    with open(file_path, "w") as f:
        for c in state:
            re = float(jnp.real(c))
            im = float(jnp.imag(c))
            f.write(f"{re:.10f} {im:.10f}\n")

def sliding_window_bases(
        window: Tuple[str, ...],
        num_qubits: int,
        *,
        background: str = "Z",
        step: int = 1,
) -> List[List[str]]:
    w = list(window)
    L = len(w)
    if L == 0 or L > num_qubits:
        return []
    bases: List[List[str]] = []
    for i in range(0, num_qubits - L + 1, step):
        b = [background] * num_qubits
        b[i : i + L] = w
        bases.append(b)
    return bases

def generate_phase_augmented_w_state(num_qubits: int, rng: PRNGKey) -> jnp.ndarray:
    """|ψ> = (1/√N) Σ_k e^{iθ_k} |1_k> with random site phases."""
    state_dim = 1 << num_qubits
    thetas = random.uniform(rng, shape=(num_qubits,), minval=0.0, maxval=2 * jnp.pi)
    amps = jnp.exp(1j * thetas) / jnp.sqrt(num_qubits)
    one_hot_indices = (1 << jnp.arange(num_qubits - 1, -1, -1))
    state = jnp.zeros(state_dim, dtype=jnp.complex64).at[one_hot_indices].set(amps.astype(jnp.complex64))
    return state


# =========================
# Pauli primitives (inline)
# =========================

@dataclass
class PauliMeasurement:
    eigenvectors: jnp.ndarray  # shape (2, 2); columns are eigenvectors for outcomes {0,1}

pauli_z = PauliMeasurement(
    eigenvectors=jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.complex64)
)
pauli_x = PauliMeasurement(
    eigenvectors=(1 / jnp.sqrt(2))
                 * jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex64)
)
pauli_y = PauliMeasurement(
    eigenvectors=(1 / jnp.sqrt(2))
                 * jnp.array([[1.0, 1.0], [1.0j, -1.0j]], dtype=jnp.complex64)
)
pauli_i = pauli_z

PAULI_MAP = {"X": pauli_x, "Y": pauli_y, "Z": pauli_z, "I": pauli_i}


# =========================
# MultiQubitMeasurement (tqdm enabled)
# =========================

class MultiQubitMeasurement:
    def __init__(self, meas_dirs: List[str]):
        self.meas_dirs = meas_dirs
        self.pauli_measurements: List[PauliMeasurement] = [PAULI_MAP[c] for c in meas_dirs]
        self.basis_vecs: List[jnp.ndarray] = self._construct_measurement_basis()

    def _construct_measurement_basis(self) -> List[jnp.ndarray]:
        measurement_basis_vectors = []
        outcome_bitstrings = list(product([0, 1], repeat=len(self.pauli_measurements)))
        meas_dirs_str = ''.join(self.meas_dirs)
        for outcome_bitstring in tqdm(outcome_bitstrings, desc=f"Constructing basis {meas_dirs_str}", disable=False):
            multi_qubit_eigenvector = None
            for pauli_measurement, outcome_bit in zip(self.pauli_measurements, outcome_bitstring):
                single_qubit_vector = pauli_measurement.eigenvectors[:, outcome_bit]
                multi_qubit_eigenvector = (
                    single_qubit_vector if multi_qubit_eigenvector is None
                    else jnp.kron(multi_qubit_eigenvector, single_qubit_vector)
                )
            measurement_basis_vectors.append(multi_qubit_eigenvector)
        return measurement_basis_vectors

    def sample_state(self, state_vec: jnp.ndarray, num_samples: int = 1000, rng: PRNGKey = None) -> jnp.ndarray:
        rng = PRNGKey(0) if rng is None else rng
        probs = jnp.array([jnp.abs(jnp.vdot(v, state_vec))**2 for v in self.basis_vecs])
        probs /= jnp.sum(probs)
        chosen_indices = jax.random.choice(rng, a=probs.shape[0], shape=(num_samples,), p=probs)
        bitstrings = int_to_bitstring(chosen_indices, len(self.meas_dirs))
        return bitstrings


# =========================
# Main
# =========================

def main() -> None:
    # ---- Parameters (match template numbers) ----
    rng_seed = 42
    num_qubits = 4
    samples_per_basis = 5000

    # Directory for all outputs
    out_dir = Path("measurements")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Generate state ----
    rng = random.PRNGKey(rng_seed)
    state = generate_phase_augmented_w_state(num_qubits, rng)

    # Save state vector under measurements/
    amp_file = out_dir / "w_phase_state.txt"
    save_state_vector_columns(state, str(amp_file))
    num_amplitudes = int(state.shape[0])
    print(f"Saved {num_amplitudes} amplitudes (Re, Im) to {amp_file}.\n")

    print("We compute complex amplitudes as inner products between the state and each measurement basis vector.")
    print("Measurements are sampled from the squared magnitudes of these amplitudes.\n")

    # ---- Measurement bases (order matches earlier examples) ----
    measurement_bases: List[List[str]] = []
    measurement_bases.append(["Z"] * num_qubits)  # ZZZZ
    window_1 = ("X", "X")
    window_2 = ("X", "Y")
    background = "Z"
    measurement_bases += sliding_window_bases(window_1, num_qubits, background=background)
    measurement_bases += sliding_window_bases(window_2, num_qubits, background=background)

    print(f"Auxiliary bases via sliding windows '{''.join(window_1)}', '{''.join(window_2)}' over '{background}' background.")
    print(f"Sampling {samples_per_basis} shots per basis across {len(measurement_bases)} bases...")

    # ---- Sampling per basis to separate files ----
    rng_samples_master = random.PRNGKey(rng_seed)
    for meas_dirs in measurement_bases:
        rng_samples_master, per_basis_rng = random.split(rng_samples_master)
        measurement = MultiQubitMeasurement(meas_dirs)  # tqdm prints "Constructing basis ..."
        samples = measurement.sample_state(state, num_samples=samples_per_basis, rng=per_basis_rng)

        # per-basis file
        basis_code = ''.join(meas_dirs)
        per_file = out_dir / f"w_phase_{basis_code}_{samples_per_basis}.txt"
        with open(per_file, "w") as f_out:
            for row in samples:
                f_out.write(bitstring_to_filestring(row, meas_dirs) + "\n")

    # ---- Final generic statement about per-basis files ----
    print(f"Saved {len(measurement_bases)} per-basis files to {out_dir}/ as w_phase_<BASIS>_{samples_per_basis}.txt")

if __name__ == "__main__":
    main()
