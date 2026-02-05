from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm

# Optional JAX backend
try:
    import jax.numpy as jnp
    from jax import random as jrandom
    HAVE_JAX = True
except ImportError:
    HAVE_JAX = False

USE_JAX = HAVE_JAX


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
    Tensor-product projective measurement in arbitrary local Pauli directions.
    Builds the full 2^N basis (tqdm shows progress).
    """
    def __init__(self, meas_dirs: List[str], *, show_tqdm: bool = True):
        self.meas_dirs = meas_dirs
        self.N = len(meas_dirs)

        try:
            self.pauli_measurements: List[PauliMeasurement] = [
                PAULI_MAP[c] for c in meas_dirs
            ]
        except KeyError as e:
            raise ValueError(
                f"Unknown measurement axis '{e.args[0]}'; use only X/Y/Z/I."
            ) from None

        self._all_Z = all(c == "Z" for c in meas_dirs)
        self.basis_vecs: List[np.ndarray] = self._construct_measurement_basis(show_tqdm=show_tqdm)

    def _construct_measurement_basis(self, *, show_tqdm: bool) -> List[np.ndarray]:
        measurement_basis_vectors: List[np.ndarray] = []
        outcome_bitstrings = list(product([0, 1], repeat=self.N))
        meas_dirs_str = ''.join(self.meas_dirs)

        iterator = tqdm(
            outcome_bitstrings,
            desc=f"Constructing basis {meas_dirs_str}",
            disable=not show_tqdm,
        )

        for outcome_bits in iterator:
            vec = None
            for P, b in zip(self.pauli_measurements, outcome_bits):
                col = P.eigenvectors[:, b]
                vec = col if vec is None else np.kron(vec, col)
            measurement_basis_vectors.append(vec.astype(np.complex64, copy=False))

        return measurement_basis_vectors

    def _probs_numpy(self, state_vec: np.ndarray) -> np.ndarray:
        if self._all_Z:
            probs = np.abs(state_vec) ** 2
        else:
            probs = np.array(
                [np.abs(np.vdot(v, state_vec)) ** 2 for v in self.basis_vecs],
                dtype=np.float64,
            )
        s = probs.sum()
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError("Probability sum is non-positive.")
        return probs / s

    def _sample_with_jax(self, state_vec: np.ndarray, num_samples: int, jax_key) -> np.ndarray:
        # Probabilities in JAX for consistency with original JAX-based implementation.
        state_j = jnp.asarray(state_vec)
        if self._all_Z:
            probs_j = jnp.abs(state_j) ** 2
        else:
            basis_j = jnp.asarray(self.basis_vecs)  # (2^N, 2^N)
            amps_j = jnp.conj(basis_j) @ state_j
            probs_j = jnp.abs(amps_j) ** 2

        probs_j = probs_j / jnp.sum(probs_j)
        idx_j = jrandom.choice(jax_key, a=probs_j.shape[0],
                               shape=(num_samples,), p=probs_j)
        idx = np.array(idx_j, dtype=np.int64)
        return int_to_bitstring(idx, self.N)

    def sample_state(
            self,
            state_vec: np.ndarray,
            num_samples: int,
            *,
            seed: Optional[int] = None,
            rng: Optional[np.random.Generator] = None,
            jax_key=None,
    ) -> np.ndarray:

        if USE_JAX and jax_key is not None:
            return self._sample_with_jax(state_vec, num_samples, jax_key)

        if seed is not None and rng is not None:
            raise ValueError("Specify either 'seed' or 'rng', not both.")

        if seed is not None:
            rng = np.random.default_rng(seed)
        if rng is None:
            rng = np.random.default_rng()

        probs = self._probs_numpy(state_vec)
        idx = rng.choice(len(probs), size=num_samples, replace=True, p=probs)
        return int_to_bitstring(idx, self.N)


#### STATE AND BASIS CONSTRUCTION ####

def generate_phase_augmented_w_state(num_qubits: int, rng_seed: int) -> np.ndarray:

    state_dim = 1 << num_qubits

    if USE_JAX:
        key = jrandom.PRNGKey(rng_seed)
        thetas = jrandom.uniform(key, shape=(num_qubits,),
                                 minval=0.0, maxval=2.0 * jnp.pi)
        amps = jnp.exp(1j * thetas) / jnp.sqrt(num_qubits)
        one_hot_indices = (1 << jnp.arange(num_qubits - 1, -1, -1, dtype=jnp.int32))
        state = jnp.zeros(state_dim, dtype=jnp.complex64)
        state = state.at[one_hot_indices].set(amps.astype(jnp.complex64))
        return np.array(state, dtype=np.complex128)

    rng = np.random.default_rng(rng_seed)
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
    bases: List[List[str]] = []
    for i in range(0, num_qubits - L + 1, stride):
        b = [background] * num_qubits
        b[i:i + L] = w
        bases.append(b)
    return bases




def save_state_vector_columns(state: np.ndarray, file_path: str, header: Optional[str] = None) -> None:
    """Save optional HEADER + two-column text: Re(state) Im(state)."""
    with open(file_path, "w") as f:
        if header is not None:
            f.write(header)
        for c in state:
            re = float(np.real(c))
            im = float(np.imag(c))
            f.write(f"{re:.10f} {im:.10f}\n")


# ---- headers ----

def header_line_state(system: str, nqubits: int, seed: int) -> str:
    return f"HEADER | system={system} | nqubits={nqubits} | seed={seed}\n"


def header_line_meas(system: str, nqubits: int,
                     basis: str, samples: int, seed: int) -> str:
    return (f"HEADER | system={system} | nqubits={nqubits} | "
            f"basis={basis} | samples={samples} | seed={seed}\n")




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
    num_qubits = 4
    samples_per_basis = 5000

    out_meas = Path("measurements")
    out_states = Path("state_vectors")
    out_meas.mkdir(parents=True, exist_ok=True)
    out_states.mkdir(parents=True, exist_ok=True)

    psi = generate_phase_augmented_w_state(num_qubits, rng_seed)

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

    # ---- Sampling per basis to separate files ----
    # JAX path: reuse single key object for all bases (original behavior).
    # NumPy path: reuse same seed per basis for deterministic behavior.
    if USE_JAX:
        jax_samples_key = jrandom.PRNGKey(rng_seed)
        rng_samples_seed = None
    else:
        jax_samples_key = None
        rng_samples_seed = rng_seed

    saved_files = 0

    for meas_dirs in measurement_bases:
        basis_code = ''.join(meas_dirs)
        measurement = MultiQubitMeasurement(meas_dirs, show_tqdm=True)

        if USE_JAX:
            samples = measurement.sample_state(
                state_vec=psi,
                num_samples=samples_per_basis,
                jax_key=jax_samples_key,
            )
        else:
            samples = measurement.sample_state(
                state_vec=psi,
                num_samples=samples_per_basis,
                seed=rng_samples_seed,
            )

        meas_header = { "basis": basis_code, "samples": int(samples_per_basis), "seed": int(rng_seed) }
        meas_path = out_meas / f"w_phase_{basis_code}_{samples_per_basis}.txt"
        save_measurements_txt(meas_path, samples, meas_dirs, { "state": state_header, "measurement": meas_header })

        saved_files += 1

    print(f"Saved {saved_files} per-basis files to ./{out_meas} as w_phase_<BASIS>_{samples_per_basis}.txt")


if __name__ == "__main__":
    main()
