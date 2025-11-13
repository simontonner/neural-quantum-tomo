from typing import Dict, Tuple
import jax
import jax.numpy as jnp
import optax
from jax import tree_util as jtu

from ..models.rbm import gibbs_steps, effective_energy
from ..neural_states.measurement import basis_meta
from ..config import DTYPE

tmap = jtu.tree_map

@jax.jit
def _pos_loss_z(params_am, samples):
    return jnp.sum(effective_energy(params_am, samples).astype(DTYPE))

@jax.jit
def _pos_loss_rot(params_am, params_ph, samples, Uc_flat, sites, combos):
    from ..neural_states.measurement import stable_log_overlap_amp2_with_meta
    log_amp2 = stable_log_overlap_amp2_with_meta(params_am, params_ph, samples, Uc_flat, sites, combos)
    return -jnp.sum(log_amp2.astype(DTYPE))

@jax.jit
def _sync_barrier(x):
    return jnp.sum(x)

def _grads_only(params, pos_batch, neg_batch, rng, Uc_flat, sites, combos, k:int, is_z:bool):
    """Not jitted here because k / is_z vary; caller jit-compiles the closure."""
    vk, rng = gibbs_steps(params["am"], k, neg_batch, rng)
    B_neg = neg_batch.shape[0]
    B_pos = pos_batch.shape[0]

    def loss_fn(p):
        L_pos = _pos_loss_z(p["am"], pos_batch) if is_z else _pos_loss_rot(p["am"], p["ph"], pos_batch, Uc_flat, sites, combos)
        L_neg = jnp.sum(effective_energy(p["am"], vk).astype(DTYPE))
        return (L_pos / B_pos) - (L_neg / B_neg)

    return jax.value_and_grad(loss_fn)(params), rng

class Trainer:
    def __init__(self, nn_state, base_lr=1e-1, phase_lr_scale=0.3, momentum=0.9, accum_steps=4):
        self.nn = nn_state
        labels = {
            "am": tmap(lambda _: "am", self.nn.params["am"]),
            "ph": tmap(lambda _: "ph", self.nn.params["ph"]),
        }
        transforms = {
            "am": optax.sgd(learning_rate=base_lr, momentum=momentum, nesterov=False),
            "ph": optax.sgd(learning_rate=base_lr * phase_lr_scale, momentum=momentum, nesterov=False),
        }
        self.opt = optax.chain(optax.clip_by_global_norm(10.0), optax.multi_transform(transforms, labels))
        self.opt_state = self.opt.init(self.nn.params)
        self.accum_steps = int(accum_steps)

    def _zero_like_params(self):
        return tmap(jnp.zeros_like, self.nn.params)

    def fit(self, loader, epochs=70, k=10, log_every=5, target=None, bases=None, space=None,
            print_metrics=True, metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}"):
        from .metrics import fidelity, KL
        history: Dict[str, list] = {"epoch": [], "Fidelity": [], "KL": []}

        @jax.jit
        def apply_updates(params, grads, opt_state):
            updates, opt_state = self.opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        for ep in range(1, epochs + 1):
            grads_accum = self._zero_like_params()
            micro = 0

            for pos_batch, neg_batch, basis_tuple, is_z in loader.iter_epoch():
                Uc_flat, sites, combos = self.nn.get_basis_meta(basis_tuple)

                (loss_val, grads), self.nn.rng = _grads_only(
                    self.nn.params, pos_batch, neg_batch, self.nn.rng,
                    Uc_flat, sites, combos, k=k, is_z=is_z
                )
                grads_accum = tmap(lambda a, b: a + b, grads_accum, grads)
                micro += 1

                if micro == self.accum_steps:
                    grads_avg = tmap(lambda g: g / self.accum_steps, grads_accum)
                    self.nn.params, self.opt_state = apply_updates(self.nn.params, grads_avg, self.opt_state)
                    grads_accum = self._zero_like_params(); micro = 0

            if micro > 0:
                grads_avg = tmap(lambda g: g / micro, grads_accum)
                self.nn.params, self.opt_state = apply_updates(self.nn.params, grads_avg, self.opt_state)

            _sync_barrier(self.nn.params["am"]["W"]).block_until_ready()

            if (target is not None) and (bases is not None) and (ep % log_every == 0):
                fid_val = fidelity(self.nn, target, space=space, bases=bases)
                kl_val  = KL(self.nn, target, space=space, bases=bases)
                history["epoch"].append(ep)
                history["Fidelity"].append(fid_val)
                history["KL"].append(kl_val)
                if print_metrics:
                    print(metric_fmt.format(ep=ep, fid=fid_val, kl=kl_val))
        return history
