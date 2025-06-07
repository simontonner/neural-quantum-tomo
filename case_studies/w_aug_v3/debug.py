import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Sequence

import numpy as np

import jax
import jax.lax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.training import checkpoints
from flax import linen as nn

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings(
    "ignore",
    message=(
        "Couldn't find sharding info under RestoreArgs.*"
    ),
    category=UserWarning,
    module="orbax.checkpoint.type_handlers"
)

data_dir = "data"
model_dir = "./models"
model_prefix = "rbm_amp_202506032030_0"


class MultiBasisDataLoader:
    char2id = {'Z': 0, 'X': 1, 'Y': 2}

    def __init__(self,
                 data_dict: dict[str, jnp.ndarray],
                 batch_size: int = 128,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 seed: int = 0):

        lengths = [v.shape[0] for v in data_dict.values()]
        if len(set(lengths)) != 1:
            raise ValueError(f"All arrays must have equal length, got: {lengths}")

        self._bases = list(data_dict.keys())
        self._arrays = list(data_dict.values())

        self.n_visible = data_dict[self._bases[0]].shape[1]
        self.total_samples = lengths[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

        self._encoded_bases = []
        for basis_str in self._bases:
            if len(basis_str) != self.n_visible:
                raise ValueError(f"All basis strings must have length {self.n_visible}. "
                                 f"Got '{basis_str}' length {len(basis_str)}.")
            enc = np.array([MultiBasisDataLoader.char2id[c] for c in basis_str],
                           dtype=np.int8)
            self._encoded_bases.append(enc)
        # Now self._encoded_bases is a list of np.int8 arrays, each shape (n_visible,)

        self.idx_slices = [
            (i, i + batch_size)
            for i in range(0, self.total_samples, batch_size)
            if not drop_last or (i + batch_size <= self.total_samples)
        ]

    def __iter__(self):
        self._order = np.arange(self.total_samples)
        if self.shuffle:
            self.rng.shuffle(self._order)
        self._slice_idx = 0
        return self

    def __next__(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self._slice_idx >= len(self.idx_slices):
            raise StopIteration

        s, e = self.idx_slices[self._slice_idx]
        self._slice_idx += 1

        batch_arrays = []
        for arr in self._arrays:
            # arr has shape (total_samples, n_visible)
            sub = arr[self._order[s:e], :]
            batch_arrays.append(sub)   # (batch_size, n_visible)

        data_array = jnp.stack(batch_arrays, axis=0)

        # basis_ids_array: shape (N_B, n_visible), dtype=int8
        basis_ids_array = jnp.stack(self._encoded_bases, axis=0) # (N_B, n_visible)

        return basis_ids_array, data_array  # ready for model.apply

    def __len__(self):
        return len(self.idx_slices)


def load_measurements(folder: str, file_pattern: str = "w_*.txt") -> dict[str, jnp.ndarray]:
    out: dict[str, jnp.ndarray] = {}

    for fp in Path(folder).glob(file_pattern):
        basis = fp.stem.split("_")[2]

        bitstrings = []
        with fp.open() as f:
            for line in f:
                bitstring = np.fromiter((c.islower() for c in line.strip()), dtype=np.float32)
                bitstrings.append(bitstring)

        arr = jnp.asarray(np.stack(bitstrings))
        if basis in out:
            out[basis] = jnp.concatenate([out[basis], arr], axis=0)
        else:
            out[basis] = arr

    return out



#### SOME HELPER FUNCTIONS

def get_computational_basis_vectors(num_qubits: int) -> jnp.ndarray:
    indices = jnp.arange(2 ** num_qubits, dtype=jnp.uint32)  # shape (2**n,)
    powers = 2 ** jnp.arange(num_qubits - 1, -1, -1, dtype=jnp.uint32)  # shape (n,)
    bits = (indices[:, None] & powers) > 0  # shape (2**n, n), bool
    return bits.astype(jnp.float32)

from functools import reduce

def construct_rotation_matrix(measurement_basis: jnp.ndarray) -> jnp.ndarray:
    SQRT2 = jnp.sqrt(2.0)
    single_qubit_rotation_matrices = jnp.stack([
        jnp.array([[1, 0], [0, 1]], dtype=jnp.complex64),               # Z
        jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64) / SQRT2,      # X
        jnp.array([[1, -1j], [1j, -1]], dtype=jnp.complex64) / SQRT2    # Y
    ])  # (3, 2, 2)

    gates = single_qubit_rotation_matrices[measurement_basis]  # (n, 2, 2)

    def kron_all(a, b):
        return jnp.kron(a, b)

    return reduce(kron_all, gates)

def bitstring_to_int(bitstring: jnp.ndarray) -> jnp.ndarray:
    powers = 2 ** jnp.arange(bitstring.shape[-1] - 1, -1, -1)
    return jnp.sum(bitstring * powers, axis=-1).astype(jnp.int32)





class PairPhaseRBM(nn.Module):
    n_visible: int
    n_hidden: int

    def setup(self):
        zeros = lambda shape: jnp.zeros(shape, dtype=jnp.float32)
        self.W_amp = self.variable('amp', 'W_amp', zeros, (self.n_visible, self.n_hidden))
        self.b_amp = self.variable('amp', 'b_amp', zeros, (self.n_visible,))
        self.c_amp = self.variable('amp', 'c_amp', zeros, (self.n_hidden,))

        self.W_pha = self.param('W_pha', nn.initializers.normal(0.01), (self.n_visible, self.n_hidden))
        self.b_pha = self.param('b_pha', nn.initializers.zeros, (self.n_visible,))
        self.c_pha = self.param('c_pha', nn.initializers.zeros, (self.n_hidden,))

    def _free_energy_amp(self, v):
        return -(v @ self.b_amp.value) - jnp.sum(jax.nn.softplus(v @ self.W_amp.value + self.c_amp.value), -1)

    def _free_energy_pha(self, v):
        return -(v @ self.b_pha) - jnp.sum(jax.nn.softplus(v @ self.W_pha + self.c_pha), -1)

    def compute_phase(self, v: jnp.ndarray) -> jnp.ndarray:
        return -self._free_energy_pha(v)

    @staticmethod
    def rotated_log_prob(rotation_amplitudes: jnp.ndarray, free_energy_amp: jnp.ndarray, free_energy_pha: jnp.ndarray) -> jnp.ndarray:
        exponent = -0.5 * free_energy_amp - 0.5j * free_energy_pha                     # (2**n,)
        values = rotation_amplitudes * jnp.exp(exponent)                          # Complex vector

        abs_vals = jnp.abs(values)
        max_log = jnp.max(jnp.log(abs_vals + 1e-30))                # Scalar for stability
        scaled_values = values * jnp.exp(-max_log)                  # Scale down before summing

        return 2 * (max_log + jnp.log(jnp.abs(jnp.sum(scaled_values)) + 1e-30))

    def _single_sample_loss(self, measurement: jnp.ndarray, basis: jnp.ndarray) -> jnp.ndarray:
        # r_rotated_log_weight
        # the following three can be moved outside. the first two don't change ever per measurement, the third one only changes between gradient steps
        computational_basis_vectors = get_computational_basis_vectors(measurement.shape[0])  # (2**n, n)
        free_energy_amp = jax.vmap(self._free_energy_amp, (0,))(computational_basis_vectors)  # (2**n,)
        free_energy_pha = jax.vmap(self._free_energy_pha, (0,))(computational_basis_vectors)  # (2**n,)

        rotation_matrix = construct_rotation_matrix(basis)  # (2**n, 2**n)

        measurement_basis_amplitude_idx = bitstring_to_int(measurement)  # (B,)
        computational_basis_amplitudes = rotation_matrix[measurement_basis_amplitude_idx]  # (B, 2**n)

        rotated_log_prob = PairPhaseRBM.rotated_log_prob(computational_basis_amplitudes, free_energy_amp, free_energy_pha)
        return rotated_log_prob  # (B,)

    def _single_basis_loss(self, measurements: jnp.ndarray, basis: jnp.ndarray) -> jnp.ndarray:
        rotated_log_weights = jax.vmap(lambda meas: self._single_sample_loss(meas, basis))(measurements)  # (B,)
        return jnp.mean(rotated_log_weights)

    def _multi_basis_loss(self, data_tuple: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        basis_measurements, bases = data_tuple  # (N_B, B, n_visible), (N_B, n_visible)

        mean_per_basis = jax.vmap(self._single_basis_loss)(basis_measurements, bases)  # (N_B,)
        return -jnp.sum(mean_per_basis)




from jax.flatten_util import ravel_pytree
# replace with your actual import

@jax.jit
def train_step_natural(
        state: TrainState,
        amp_vars: Dict[str, jnp.ndarray],
        data_tuple: Tuple[jnp.ndarray, jnp.ndarray],
        lr: float = 1e-3,
        eps: float = 1e-5,
) -> Tuple[TrainState, jnp.ndarray]:
    """
    One step of diagonal‐Fisher natural gradient descent for PairPhaseRBM.
    data_tuple = (basis_measurements, bases), with shapes
      basis_measurements: (N_B, B, n_visible)
      bases             : (N_B, n_visible)
    """
    bases, bms  = data_tuple   # per‐basis minibatch and basis

    # 1) Build per‐basis loss vector ℓ_i(θ)
    def per_basis_loss_vec(params):
        vars = {'params': params, 'amp': amp_vars}
        # returns shape (N_B,), negative mean log‐weight per basis
        return jax.vmap(
            lambda meas, bas: -state.apply_fn(vars, meas, bas, method=PairPhaseRBM._single_basis_loss)
        )(bms, bases)

    # 2) Compute scalar loss and its ordinary gradient
    loss_vec     = per_basis_loss_vec(state.params)       # (N_B,)
    loss_scalar  = jnp.sum(loss_vec)                     # scalar
    loss_val, full_grads = jax.value_and_grad(lambda p: jnp.sum(per_basis_loss_vec(p)))(state.params)

    # 3) Compute per‐basis gradients via jacrev → pytree with leaves shaped (N_B, …)
    per_example_grads = jax.jacrev(
        lambda p: per_basis_loss_vec(p)
    )(state.params)

    # 4) Flatten per‐example grads into an array of shape (N_B, D)
    flat_grads, unravel_fn = ravel_pytree(per_example_grads)  # (N_B, D)

    # 5) Diagonal Fisher: F_diag_j = (1/N_B) Σ_i flat_grads[i, j]^2
    fisher_diag = jnp.mean(flat_grads ** 2, axis=0)            # (D,)

    # 6) Flatten the full ∇_θ L into vector (D,)
    flat_full_grad, _ = ravel_pytree(full_grads)

    # 7) Natural gradient in flat space: inv(F_diag + εI) · ∇L
    inv_fdiag = 1.0 / (fisher_diag + eps)  # (D,)
    nat_flat  = inv_fdiag * flat_full_grad # (D,)
    nat_grads = unravel_fn(nat_flat)       # pytree like state.params

    # 8) Update parameters: θ ← θ − lr · natural_gradient
    new_params = jax.tree_map(lambda p, ng: p - lr * ng, state.params, nat_grads)
    new_state  = state.replace(params=new_params)

    return new_state, loss_val



def train_rbm_pha(
        state:   TrainState,
        amp_vars: Dict[str, jnp.ndarray],
        loader:  "MultiBasisDataLoader",
        num_epochs: int) -> Tuple[TrainState, Dict[int, float]]:

    metrics: Dict[str, float] = {}

    for epoch in range(num_epochs):
        tot_loss  = 0.0
        n_batches = 0

        for data_tuple in loader:                       # (basis_ids, data_array)
            state, loss = train_step_natural(state, amp_vars, data_tuple)
            tot_loss  += loss
            n_batches += 1

        avg_loss = float(tot_loss / n_batches)

        metrics[epoch] = dict(loss=avg_loss)

        print(f"Epoch {epoch+1}/{num_epochs} │ Loss: {avg_loss:.4f}")

    return state, metrics


###### HERE IT BEGINS


data_dict = load_measurements(data_dir, "w_*.txt")

keys_pha = [k for k in data_dict if 'Z' in k and re.search(r'[XY]', k)]
dict_pha = {k: data_dict[k] for k in keys_pha}

params_amp = checkpoints.restore_checkpoint(ckpt_dir=str(Path(model_dir).resolve()), target=None, prefix=model_prefix)
amp_vars = {"W_amp": params_amp["W"], "b_amp": params_amp["b"], "c_amp": params_amp["c"]}



batch_size    = 6400
visible_units = 10
hidden_units  = 20
num_epochs    = 50
init_lr       = 1e-2
weight_decay  = 0.5

key = jax.random.PRNGKey(0)

model_pha = PairPhaseRBM(n_visible=visible_units, n_hidden=hidden_units)

dummy_basis = jnp.zeros((1, visible_units), dtype=jnp.int8)
dummy_data  = jnp.zeros((1, batch_size, visible_units), dtype=jnp.float32)

# FIXED: pass method explicitly
vars_pha = model_pha.init(key, (dummy_data, dummy_basis), method=PairPhaseRBM._multi_basis_loss)

optim = optax.sgd(init_lr)

state_pha = TrainState.create(apply_fn=model_pha.apply, params=vars_pha['params'], tx=optim)

loader_pha = MultiBasisDataLoader(dict_pha, batch_size=batch_size, shuffle=True)

state_pha, metrics_pha = train_rbm_pha(state_pha, amp_vars, loader_pha, num_epochs)
