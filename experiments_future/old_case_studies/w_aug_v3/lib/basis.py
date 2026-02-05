import jax.numpy as jnp

from .pauli import pauli_x, pauli_y, pauli_z


ROTATOR_DICT = { 'Z': pauli_z.rotator_bras, 'X': pauli_x.rotator_bras, 'Y': pauli_y.rotator_bras }


def get_computational_basis_vectors(num_qubits: int) -> jnp.ndarray:
    indices = jnp.arange(2 ** num_qubits, dtype=jnp.uint32)  # shape (2**n,)
    powers = 2 ** jnp.arange(num_qubits - 1, -1, -1, dtype=jnp.uint32)  # shape (n,)
    bits = (indices[:, None] & powers) > 0  # shape (2**n, n), bool
    return bits.astype(jnp.float32)


def construct_rotation_matrix(
        measurement_basis: jnp.ndarray,
        basis_encoding: dict[str, int] = {'Z': 0, 'X': 1, 'Y': 2}) -> jnp.ndarray:

    # we sort the rotators according to the basis encoding
    rotator_array = jnp.stack([ROTATOR_DICT[l] for l, _ in sorted(basis_encoding.items(), key=lambda kv: kv[1])])

    rotation_matrix = jnp.array([1.0 + 0j], dtype=jnp.complex64).reshape((1, 1))
    for basis_idx in measurement_basis:
        rotation_matrix = jnp.kron(rotation_matrix, rotator_array[basis_idx])
    return rotation_matrix



