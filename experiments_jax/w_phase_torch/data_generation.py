#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-augmented W-state generation + measurement sampling (single file, minimal logs)

- No NumPy used for saving.
- All-Z basis is created inside main; additional bases come from a sliding-window helper.
- MultiQubitMeasurement kept functionally as-is.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
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
    """Return shape (N, num_bits) of 0/1 as uint8 for given integer indices."""
    indices = indices.astype(jnp.int32)
    # MSB -> LSB (matches your original helper & typical formatter)
    powers = (1 << jnp.arange(num_bits - 1, -1, -1, dtype=jnp.int32))
    bits = (indices[..., None] & powers) > 0
    return bits.astype(jnp.uint8)


def save_state_vector_columns(state: jnp.ndarray, file_path: str) -> None:
    """Save two-column text: Re(state) Im(state) without NumPy."""
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
    """Generate bases by sliding a tuple 'window' across a 'background'."""
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
# MultiQubitMeasurement (kept functionally as-is)
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
        for outcome_bitstring in tqdm(outcome_bitstrings, desc=f"Constructing basis {meas_dirs_str}"):
            multi_qubit_eigenvector = None

            for pauli_measurement, outcome_bit in zip(self.pauli_measurements, outcome_bitstring):
                single_qubit_vector = pauli_measurement.eigenvectors[:, outcome_bit]

                if multi_qubit_eigenvector is None:
                    multi_qubit_eigenvector = single_qubit_vector
                else:
                    multi_qubit_eigenvector = jnp.kron(multi_qubit_eigenvector, single_qubit_vector)

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
    # ---- Parameters ----
    rng_seed = 42
    num_qubits = 4
    samples_per_basis = 5000

    amp_file = "w_phase_state.txt"
    unique_bases_file = "w_phase_unique_bases.txt"
    values_out = "w_phase_meas_values.txt"
    bases_out = "w_phase_meas_bases.txt"

    # ---- Generate state ----
    rng = random.PRNGKey(rng_seed)
    state = generate_phase_augmented_w_state(num_qubits, rng)
    save_state_vector_columns(state, amp_file)
    num_amplitudes = int(state.shape[0])
    print(f"Saved {num_amplitudes} amplitudes (Re, Im) to {amp_file}.")

    print("\nWe compute complex amplitudes as inner products between the state and each measurement basis vector.\n"
          "Measurements are sampled from the squared magnitudes of these amplitudes.\n")

    bases: List[List[str]] = []
    bases.append(["Z"] * num_qubits)  # all-Z inside main
    window_1 = ("X", "X")
    window_2 = ("X", "Y")
    background = "Z"
    bases += sliding_window_bases(window_1, num_qubits, background="Z")
    bases += sliding_window_bases(window_2, num_qubits, background="Z")

    print(f"Auxiliary bases via sliding windows '{''.join(window_1)}', '{''.join(window_2)}' over background.")

    with open(unique_bases_file, "w") as f:
        for b in bases:
            f.write(" ".join(b) + " \n")
    print(f"Saved {len(bases)} unique bases to {unique_bases_file}.")

    print(f"Sampling {samples_per_basis} shots per basis across {len(bases)} bases...")
    with open(values_out, "w") as f_meas, open(bases_out, "w") as f_basis:
        # FIX: use the SAME RNG key for every basis (mirror the notebook)
        rng_samples = random.PRNGKey(rng_seed)

        for i, meas_dirs in enumerate(bases):
            measurement = MultiQubitMeasurement(meas_dirs)
            samples = measurement.sample_state(state, num_samples=samples_per_basis, rng=rng_samples)

            basis_str = " ".join(meas_dirs) + " \n"
            for row in samples:
                f_meas.write(" ".join(str(int(b)) for b in row) + " \n")
                f_basis.write(basis_str)

    print(f"Saved values to: {values_out}")
    print(f"Saved bases to: {bases_out}")


if __name__ == "__main__":
    main()
