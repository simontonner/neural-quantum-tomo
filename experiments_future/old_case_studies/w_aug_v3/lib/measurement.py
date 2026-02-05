from itertools import product
from typing import List

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from tqdm import tqdm

from .pauli import PauliMeasurement, pauli_x, pauli_y, pauli_z, pauli_i
from .formatting import int_to_bitstring


PAULI_MAP = { 'X': pauli_x, 'Y': pauli_y, 'Z': pauli_z, 'I': pauli_i }


class MultiQubitMeasurement:
    def __init__(self, meas_dirs: List[str]):
        self.meas_dirs = meas_dirs
        self.pauli_measurements: List[PauliMeasurement] = [PAULI_MAP[c] for c in meas_dirs]
        self.basis_vecs: List[jnp.ndarray] = self._construct_measurement_basis()

    def _construct_measurement_basis(self) -> List[jnp.ndarray]:
        measurement_basis_vectors = []
        # can be improved with jax
        outcome_bitstrings = list(product([0, 1], repeat=len(self.pauli_measurements)))

        meas_dirs_str = ''.join(self.meas_dirs)
        for outcome_bitstring in tqdm(outcome_bitstrings, desc=f"Constructing basis {meas_dirs_str}"):
            multi_qubit_eigenvector = None

            for pauli_measurement, outcome_bit in zip(self.pauli_measurements, outcome_bitstring):
                # the eigenvectors are stored in the columns
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



