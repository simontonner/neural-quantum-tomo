from typing import Dict, Tuple, List
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from ..config import DTYPE, CDTYPE
from ..models.rbm import init_rbm_params, effective_energy
from .pauli import create_dict, as_complex_unitary
from .measurement import basis_meta

class ComplexWaveFunction:
    """psi(sigma) = exp(-F_lambda/2) * exp(-i F_mu/2)."""

    def __init__(self, num_visible: int, num_hidden: int = None,
                 unitary_dict: Dict[str, jnp.ndarray] = None,
                 rng_seed: int = 0):
        self.rng = jax.random.PRNGKey(rng_seed)
        self.num_visible = int(num_visible)
        self.num_hidden  = int(num_hidden) if num_hidden else self.num_visible
        raw = unitary_dict if unitary_dict is not None else create_dict()
        self.U = {k: as_complex_unitary(v) for k, v in raw.items()}

        self.params = {
            "am": init_rbm_params(self.num_visible, self.num_hidden, key=jax.random.PRNGKey(rng_seed+1)),
            "ph": init_rbm_params(self.num_visible, self.num_hidden, key=jax.random.PRNGKey(rng_seed+2)),
        }
        self._basis_cache: Dict[Tuple[str, ...], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = {}

    def psi_complex(self, v: jnp.ndarray) -> jnp.ndarray:
        E_lam = effective_energy(self.params["am"], v)
        E_mu  = effective_energy(self.params["ph"], v)
        amp = jnp.exp(jnp.clip(-0.5 * E_lam, -100.0, 100.0)).astype(DTYPE)
        ph  = jnp.clip(-0.5 * E_mu, -1e6, 1e6).astype(DTYPE)
        return amp.astype(CDTYPE) * jnp.exp(1j * ph.astype(CDTYPE))

    def psi_complex_normalized(self, v: jnp.ndarray) -> jnp.ndarray:
        E = effective_energy(self.params["am"], v)
        ph = (-0.5 * effective_energy(self.params["ph"], v)).astype(DTYPE)
        logZ = logsumexp(-E, axis=0)
        return jnp.exp(((-0.5 * E) - 0.5 * logZ).astype(CDTYPE) + 1j * ph.astype(CDTYPE))

    def generate_hilbert_space(self, size: int = None) -> jnp.ndarray:
        size = self.num_visible if size is None else int(size)
        n = 1 << size
        ar = jnp.arange(n, dtype=jnp.int64)
        shifts = jnp.arange(size - 1, -1, -1, dtype=jnp.int64)
        return ((ar[:, None] >> shifts[None, :]) & 1).astype(DTYPE)

    def get_basis_meta(self, basis_tuple: Tuple[str, ...]):
        if basis_tuple not in self._basis_cache:
            self._basis_cache[basis_tuple] = basis_meta(self.U, basis_tuple)
        return self._basis_cache[basis_tuple]
