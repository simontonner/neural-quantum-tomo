import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import optax

from ..config import DTYPE, CDTYPE
from ..models.rbm import init_rbm_params, effective_energy, gibbs_steps
from .pauli import create_dict, as_complex_unitary

class PositiveWaveFunction:
    """Phase-free RBM: psi(v) = exp(-E_am/2) ∈ ℝ_{≥0}."""

    def __init__(self, num_visible: int, num_hidden: int = None,
                 unitary_dict=None, rng_seed: int = 0):
        self.rng = jax.random.PRNGKey(rng_seed)
        self.num_visible = int(num_visible)
        self.num_hidden  = int(num_hidden) if num_hidden else self.num_visible
        raw = unitary_dict if unitary_dict is not None else create_dict()
        self.U = {k: as_complex_unitary(v) for k, v in raw.items()}
        self.params = {"am": init_rbm_params(self.num_visible, self.num_hidden, key=jax.random.PRNGKey(rng_seed+1))}

    def psi_complex(self, v):  # complex cast for API parity
        E = effective_energy(self.params["am"], v)
        amp = jnp.exp(jnp.clip(-0.5 * E, -100.0, 100.0)).astype(DTYPE)
        return amp.astype(CDTYPE)

    def psi_complex_normalized(self, v):
        E = effective_energy(self.params["am"], v)
        logZ = logsumexp(-E, axis=0)
        return jnp.exp(((-0.5 * E) - 0.5 * logZ).astype(CDTYPE))

    def generate_hilbert_space(self, size=None):
        size = self.num_visible if size is None else int(size)
        n = 1 << size
        ar = jnp.arange(n, dtype=jnp.int64)
        shifts = jnp.arange(size - 1, -1, -1, dtype=jnp.int64)
        return ((ar[:, None] >> shifts[None, :]) & 1).astype(DTYPE)

    # simple CD-k trainer (data-seeded negatives), meant for Z-only datasets
    def fit(self, loader, epochs=50, k=10, lr=1e-1):
        opt = optax.sgd(learning_rate=lr, momentum=0.9)
        opt_state = opt.init(self.params)

        @jax.jit
        def step(params, pos_batch, neg_seed, key, opt_state):
            vk, key = gibbs_steps(params["am"], k, neg_seed, key)
            Bp = pos_batch.shape[0]; Bm = vk.shape[0]

            def loss_fn(p):
                pos = jnp.sum(effective_energy(p["am"], pos_batch)) / Bp
                neg = jnp.sum(effective_energy(p["am"], vk)) / Bm
                return pos - neg

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, key, opt_state, loss

        for _ in range(epochs):
            for pos_batch, _, _ in loader.iter_epoch():
                self.params, self.rng, opt_state, _ = step(self.params, pos_batch, pos_batch, self.rng, opt_state)
        return self.params
