from dataclasses import dataclass

import jax.numpy as jnp

@dataclass(frozen=True)
class PauliMeasurement:
    name: str
    operator: jnp.ndarray           # shape (2, 2)
    eigenvalues: jnp.ndarray        # shape (2,)
    eigenvectors: jnp.ndarray       # shape (2, 2), columns are eigenvectors in Z basis

    @property
    def rotator_bras(self) -> jnp.ndarray:
        return self.eigenvectors.conj().T

    @property
    def positive_eigenvalue(self) -> float:
        return self.eigenvalues[0]

    @property
    def negative_eigenvalue(self) -> float:
        return self.eigenvalues[1]

    @property
    def positive_eigenvector(self) -> jnp.ndarray:
        return self.eigenvectors[:, 0]

    @property
    def negative_eigenvector(self) -> jnp.ndarray:
        return self.eigenvectors[:, 1]



SQRT2 = jnp.sqrt(2.0)

pauli_i = PauliMeasurement(
    name='I',
    operator=jnp.eye(2, dtype=jnp.complex64),
    eigenvalues=jnp.array([1, 1], dtype=jnp.float32),
    eigenvectors=jnp.eye(2, dtype=jnp.complex64)
)

pauli_x = PauliMeasurement(
    name='X',
    operator=jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64),
    eigenvalues=jnp.array([1, -1], dtype=jnp.float32),
    eigenvectors=jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64) / SQRT2
)

pauli_y = PauliMeasurement(
    name='Y',
    operator=jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64),
    eigenvalues=jnp.array([1, -1], dtype=jnp.float32),
    eigenvectors=jnp.array([[1, 1], [1j, -1j]], dtype=jnp.complex64) / SQRT2
)

pauli_z = PauliMeasurement(
    name='Z',
    operator=jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64),
    eigenvalues=jnp.array([1, -1], dtype=jnp.float32),
    eigenvectors=jnp.eye(2, dtype=jnp.complex64)
)
