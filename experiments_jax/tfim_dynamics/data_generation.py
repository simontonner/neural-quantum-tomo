#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TFIM ground-state generation + measurement sampling (self-contained, multi-h sweep)

Notebook-style: edit parameters in main(), then run. No CLI.

Run structure:
  • Top summary: J fixed; NetKet builds the 2D L×L PBC graph & Ising Hamiltonian; ground states via sparse exact diagonalization (SciPy eigsh).
  • For each h:
      1) Build & diagonalize (log E0).
      2) Save ground-state amplitudes to state_vectors/tfim_<details>.txt (Re, Im per line).
      3) Construct measurement basis (Z^N) — tqdm shows progress.
      4) Draw `shots` measurements and save to measurements/tfim_<details>_<shots>.txt (one filestring per line).
      5) Per-h SUMMARY line (E0, ⟨∑σ^z⟩, ⟨σ^z⟩ per-site).

Final footer: counts of saved state vectors and measurement files.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Tuple

import time
import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from jax.random import PRNGKey

from tqdm import tqdm  # assumed available

from netket.operator import LocalOperator, Ising
from netket.hilbert import Spin
from netket.graph import Hypercube
from scipy.sparse.linalg import eigsh


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

def format_bytes(num: int) -> str:
    n = float(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


# =========================
# Pauli primitives (inline) — same convention as your W-state file
# =========================

@dataclass
class PauliMeasurement:
    eigenvectors: jnp.ndarray  # shape (2, 2); columns are eigenvectors for outcomes {0,1}

pauli_z = PauliMeasurement(  # columns: |0>, |1>
    eigenvectors=jnp.array([[1.0, 0.0],
                            [0.0, 1.0]], dtype=jnp.complex64)
)
pauli_x = PauliMeasurement(
    eigenvectors=(1 / jnp.sqrt(2))
                 * jnp.array([[1.0, 1.0],
                              [1.0, -1.0]], dtype=jnp.complex64)
)
pauli_y = PauliMeasurement(
    eigenvectors=(1 / jnp.sqrt(2))
                 * jnp.array([[1.0, 1.0],
                              [1.0j, -1.0j]], dtype=jnp.complex64)
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

    def sample_state(self, state_vec: jnp.ndarray, num_samples: int = 1000, rng: PRNGKey | None = None) -> jnp.ndarray:
        rng = PRNGKey(0) if rng is None else rng
        probs = jnp.array([jnp.abs(jnp.vdot(v, state_vec))**2 for v in self.basis_vecs])
        probs /= jnp.sum(probs)
        chosen_indices = jax.random.choice(rng, a=probs.shape[0], shape=(num_samples,), p=probs)
        bitstrings = int_to_bitstring(chosen_indices, len(self.meas_dirs))
        return bitstrings


# =========================
# TFIM helpers (NetKet)
# =========================

def build_tfim(L: int, h: float, J: float, *, pbc: bool = True):
    graph = Hypercube(length=L, n_dim=2, pbc=pbc)
    hilbert = Spin(s=0.5, N=graph.n_nodes)
    H = Ising(hilbert, graph, h=h, J=J)
    return hilbert, graph, H

def calculate_groundstate(H: LocalOperator) -> tuple[float, jnp.ndarray]:
    t0 = time.time()
    eigvals, eigvecs = eigsh(H.to_sparse(), k=1, which='SA')  # smallest algebraic
    e0 = float(eigvals[0])
    state_vector = jnp.array(eigvecs[:, 0], dtype=jnp.complex64)
    dt = time.time() - t0
    print(f"[diag] eigsh(which='SA', k=1) finished in {dt:.3f}s  ->  E0 = {e0:.8f}")
    return e0, state_vector

def magnetization_z(hilbert: Spin, psi: jnp.ndarray) -> tuple[float, float]:
    """Compute ⟨∑_i σ^z_i⟩ and per-site value in the σ^z basis."""
    N = getattr(hilbert, 'size', None) or getattr(hilbert, 'N')
    sigmaz = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex64)

    M = None
    for i in range(N):
        op_i = LocalOperator(hilbert, {i: sigmaz})
        M = op_i if M is None else (M + op_i)

    psi_np = np.asarray(psi, dtype=np.complex64)
    M_sparse = M.to_sparse()
    mz = float((psi_np.conj().T @ (M_sparse @ psi_np)).real)
    return mz, mz / N


# =========================
# Main
# =========================

def main() -> None:
    # ---- Parameters (edit like a notebook) ----
    rng_seed    = 42
    side_length = 3          # N = side_length * side_length
    J           = -1.0       # Ising coupling (Z-Z) — fixed for the sweep
    h_values    = [1.0, 2.0, 2.8, 3.0, 3.3, 3.6, 4.0]  # iterate magnetizations here

    shots       = 10_000     # per-h samples
    out_meas    = Path("measurements")
    out_states  = Path("state_vectors")  # separate directory for amplitude dumps

    out_meas.mkdir(parents=True, exist_ok=True)
    out_states.mkdir(parents=True, exist_ok=True)

    # ---- Top summary ----
    print("================================================================")
    print("[summary] TFIM data generation sweep")
    print(f"[summary] Fixed J={J:+.4f}; NetKet constructs 2D L×L PBC graph & Ising Hamiltonian;")
    print("[summary] Ground states via sparse exact diagonalization (SciPy eigsh, k=1).")
    print(f"[summary] L={side_length} (N={side_length*side_length}) | h values: {h_values}")
    print(f"[summary] Output: states → {out_states.resolve()}  |  measurements → {out_meas.resolve()}")
    print("================================================================\n")

    rng_master = random.PRNGKey(rng_seed + 1)

    saved_states = 0
    saved_files  = 0

    for h in h_values:
        print(f"=== [h={h:+.4f}] Build & diagonalize TFIM (NetKet, LxL, PBC) ===")
        hilbert, graph, H = build_tfim(L=side_length, h=h, J=J, pbc=True)
        e0, psi = calculate_groundstate(H)

        # Save state vector (Re, Im) without 'state' in filename
        state_path = out_states / f"tfim_h{h:.2f}_{side_length}x{side_length}.txt"
        save_state_vector_columns(psi, str(state_path))
        saved_states += 1
        print(f"Saved ground-state amplitudes to {state_path}")

        # Magnetization
        mz_total, mz_site = magnetization_z(hilbert, psi)
        print(f"⟨∑ σ^z⟩ = {mz_total:+.6f}  |  per-site ⟨σ^z⟩ = {mz_site:+.6f}")

        # Construct measurement basis (Z^N) with visible tqdm
        basis = ['Z'] * graph.n_nodes
        _ = MultiQubitMeasurement(basis)  # tqdm prints: Constructing basis ZZZ...
        # We don't retain it; re-instantiate below to keep tqdm visible *then* sample
        measurement = MultiQubitMeasurement(basis)

        # Sample all shots in one go (no batch logs)
        rng_master, rng_h = random.split(rng_master)
        samples = measurement.sample_state(psi, num_samples=shots, rng=rng_h)

        meas_path = out_meas / f"tfim_h{h:.2f}_{side_length}x{side_length}_{shots}.txt"
        with open(meas_path, "w") as f:
            for s in samples:
                f.write(bitstring_to_filestring(s, basis) + "\n")
        saved_files += 1
        print(f"Wrote {shots} measurements to {meas_path}")

        # Per-h concise summary
        print(
            "SUMMARY | "
            f"h={h:+.6f} | L={side_length} N={graph.n_nodes} | J={J:+.6f} | "
            f"shots={shots} | E0={e0:.8f} | Mz_total={mz_total:+.6f} Mz_site={mz_site:+.6f}\n"
        )

    # Footer, in the spirit of your W-state log
    print(f"Saved {saved_states} state-vector files to {out_states}/ as tfim_<details>.txt")
    print(f"Saved {saved_files} measurement files to {out_meas}/ as tfim_h<details>_{shots}.txt")

if __name__ == "__main__":
    main()
