import jax
import jax.numpy as jnp
from jax import tree


from lib.pauli import pauli_x, pauli_y, pauli_z

import jax.numpy as jnp
jnp.set_printoptions(suppress=True, formatter={'float_kind': '{: .8f}'.format})


def get_computational_basis_vectors(num_qubits: int) -> jnp.ndarray:
    indices = jnp.arange(2 ** num_qubits, dtype=jnp.uint32)  # shape (2**n,)
    powers = 2 ** jnp.arange(num_qubits - 1, -1, -1, dtype=jnp.uint32)  # shape (n,)
    bits = (indices[:, None] & powers) > 0  # shape (2**n, n), bool
    return bits.astype(jnp.float32)


def construct_rotation_matrix(measurement_basis: jnp.ndarray) -> jnp.ndarray:

    # we use our rotators made from stacked bra eigenvalues
    rotator_array = jnp.stack([pauli_z.rotator_bras, pauli_x.rotator_bras, pauli_y.rotator_bras])

    rotation_matrix = jnp.array([1.0 + 0j], dtype=jnp.complex64).reshape((1, 1))
    for basis_idx in measurement_basis:
        rotation_matrix = jnp.kron(rotation_matrix, rotator_array[basis_idx])
    return rotation_matrix


def dummy_free_energy(sigma: jnp.ndarray, params: dict) -> jnp.ndarray:
    return jnp.dot(sigma, params["W"]) + params["b"]  # (2**n,)

def multiply_with_leaf(factor, leaf):
    if leaf.ndim == 1:
        return factor * leaf
    else:
        return factor[:, None] * leaf


def bitstring_to_int(bitstring: jnp.ndarray) -> jnp.ndarray:
    powers = 2 ** jnp.arange(bitstring.shape[-1] - 1, -1, -1)
    return jnp.sum(bitstring * powers, axis=-1).astype(jnp.int32)


########




def rotated_log_prob_vanilla(rotation_weights, free_energy_lambda, free_energy_mu):
    computational_amplitudes = jnp.exp(-0.5 * free_energy_lambda) * jnp.exp(-0.5j * free_energy_mu) # unnormalized
    rotated_amplitude = jnp.vdot(rotation_weights, computational_amplitudes)

    #print(f"rotated amplitude: {rotated_amplitude}")

    rotated_log_prob = jnp.log(jnp.abs(rotated_amplitude) ** 2 + 1e-30)
    return rotated_log_prob


def loss_fn_mine_vanilla(measurements, basis, params_lambda, params_mu):

    # get the free energies for all computational basis vectors to construct the full state vector
    computational_basis_vectors = get_computational_basis_vectors(measurements.shape[1])
    free_energy_lambda = dummy_free_energy(computational_basis_vectors, params_lambda)
    free_energy_mu = dummy_free_energy(computational_basis_vectors, params_mu)

    rotation_matrix = construct_rotation_matrix(basis)          # (2**n, 2**n)

    get_log_prob = lambda m: rotated_log_prob_vanilla(rotation_matrix[bitstring_to_int(m)], free_energy_lambda, free_energy_mu)
    log_probs = jax.vmap(get_log_prob)(measurements)

    loss = -jnp.mean(log_probs)
    return loss


autodiff_grad_fn_mine = jax.grad(loss_fn_mine_vanilla, argnums=3)

##########################

#### GRADIENTS USING THE REDUCED ROTATOR

def loss_fn_local(
        measurement, # jnp.ndarray of shape (n, ) where n is the number of qubits
        basis, # jnp.ndarray of shape (n, ) where n is the number of qubits
        params_lambda, params_mu):

    j, k = jnp.nonzero(basis != 0, size=2, fill_value=-1)[0]
    local_indices = jnp.array([j, k])
    local_basis = basis[local_indices]
    local_measurement = measurement[local_indices]
    local_rotation_matrix = construct_rotation_matrix(local_basis)  # (4, 4)
    local_rotation_weights = local_rotation_matrix[bitstring_to_int(local_measurement)]

    # amplitudes mismatching with our Z measurements are 0. There are only 4 remaining amplitudes with the local variations
    local_measurement_combos = jnp.array([[0,0],[0,1],[1,0],[1,1]], dtype=measurement.dtype)
    local_computational_basis_vectors = jnp.tile(measurement, (4, 1)).at[:, local_indices].set(local_measurement_combos)  # (4, n)

    local_free_energy_lambda = dummy_free_energy(local_computational_basis_vectors, params_lambda)
    local_free_energy_mu = dummy_free_energy(local_computational_basis_vectors, params_mu)

    local_computational_amplitudes = jnp.exp(-0.5 * local_free_energy_lambda -0.5j * local_free_energy_mu)

    rotated_amplitude = jnp.vdot(local_rotation_weights, local_computational_amplitudes)

    #print(f"local amplitude for {local_measurement}: {rotated_amplitude}")


    rotated_log_prob = jnp.log(jnp.abs(rotated_amplitude) ** 2 + 1e-30)
    return rotated_log_prob


autodiff_grad_fn_local = jax.grad(loss_fn_local, argnums=3)


###########################

basis_local = jnp.array([0, 0, 0, 0, 0, 1, 2, 0, 0, 0])  # Z Z Z Z Z X Y Z Z Z Z
measurements_local = jnp.array([
    [0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1],
    [0,1,0,1,0,1,0,1,0,1],
    [1,0,1,0,1,0,1,0,1,0],
    [0,0,1,1,0,0,1,1,0,0],
    [1,1,0,0,1,1,0,0,1,1],
], dtype=jnp.int32)

params_lambda = {
    "W": jnp.linspace(0.05, 0.15, 10, dtype=jnp.float32),
    "b": jnp.array(0.3, dtype=jnp.float32)
}

params_mu = {
    "W": jnp.linspace(0.05, 0.15, 10, dtype=jnp.float32),
    "b": jnp.array(0.3, dtype=jnp.float32)
}

#######


autodiff_grads_local = jax.vmap(autodiff_grad_fn_local, in_axes=(0, None, None, None))(measurements_local, basis_local, params_lambda, params_mu)

# since we get a bunch of pytrees, we need to use the tree map to calculate the mean over the batch
autodiff_grad_local = tree.map(lambda x: -jnp.mean(x, axis=0), autodiff_grads_local)
print(autodiff_grad_local)


######


autodiff_grad_mine = autodiff_grad_fn_mine(measurements_local, basis_local, params_lambda, params_mu)
print(autodiff_grad_mine)