import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.random import PRNGKey


class RBM(nn.Module):
    n_visible: int
    n_hidden: int
    k: int = 1  # CD-k / PCD-k

    # ─────────────────────── model forward ────────────────────────
    @nn.compact
    def __call__(self,
                 data_batch: jnp.ndarray,
                 v_persistent: jnp.ndarray,
                 rng: PRNGKey) -> tuple[jnp.ndarray, dict]:
        W = self.param("W", nn.initializers.normal(0.01), (self.n_visible, self.n_hidden))
        b = self.param("b", nn.initializers.zeros,         (self.n_visible,))
        c = self.param("c", nn.initializers.zeros,         (self.n_hidden,))
        params = {"W": W, "b": b, "c": c}

        # ── positive & negative phases ────────────────────────────
        v_k, key = self._gibbs_sample(params, v_persistent, rng, k=self.k)
        v_k = jax.lax.stop_gradient(v_k)

        free_e_data  = self._free_energy(params, data_batch)
        free_e_model = self._free_energy(params, v_k)

        pcd_loss = jnp.mean(free_e_data) - jnp.mean(free_e_model)
        aux      = {"v_persistent": v_k, "key": key}
        return pcd_loss, aux

    # ─────────────────────── statics ──────────────────────────────
    @staticmethod
    def _free_energy(params, v):
        W, b, c   = params["W"], params["b"], params["c"]
        v_term    = jnp.dot(v, b)
        h_term    = jnp.sum(jax.nn.softplus(v @ W + c), axis=-1)
        return -(v_term + h_term)

    @staticmethod
    def _gibbs_step(_, state, params, T=1.0):
        v, key      = state
        W, b, c     = params["W"], params["b"], params["c"]
        key, hk, vk = jax.random.split(key, 3)

        h_prob = jax.nn.sigmoid((v @ W + c) / T)
        h      = jax.random.bernoulli(hk, h_prob).astype(jnp.float32)

        v_prob = jax.nn.sigmoid((h @ W.T + b) / T)
        v      = jax.random.bernoulli(vk, v_prob).astype(jnp.float32)
        return v, key

    @staticmethod
    def _gibbs_sample(params, v0, rng, k=1, T=1.0):
        body = lambda i, st: RBM._gibbs_step(i, st, params, T)
        v_k, key = jax.lax.fori_loop(0, k, body, (v0, rng))
        return v_k, key

    # ────────────────────── sampling util ─────────────────────────
    @nn.nowrap
    def generate(self,
                 params: dict,
                 n_samples: int,
                 T_schedule: jnp.ndarray,
                 rng: PRNGKey) -> jnp.ndarray:
        rng, key = jax.random.split(rng)
        v = jax.random.bernoulli(key, p=0.5,
                                 shape=(n_samples, self.n_visible)).astype(jnp.float32)
        state = (v, rng)

        step = lambda i, st: RBM._gibbs_step(i, st, params, T_schedule[i])
        v_fin, _ = jax.lax.fori_loop(0, len(T_schedule), step, state)
        return v_fin