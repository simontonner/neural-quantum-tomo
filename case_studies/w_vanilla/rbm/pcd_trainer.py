import functools
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax.random import PRNGKey


class RBMTrainState(train_state.TrainState):
    """Bundles params and opt-state (Flax API)."""
    pass


@functools.partial(jax.jit)
def train_step(state: RBMTrainState,
               data_batch: jnp.ndarray,
               v_persistent: jnp.ndarray,
               key: PRNGKey):
    pcd_loss_fn = lambda params: state.apply_fn(
        {"params": params}, data_batch, v_persistent, key)

    (loss, aux), grads = jax.value_and_grad(pcd_loss_fn,
                                            has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, aux["v_persistent"], aux["key"]


def train_rbm(state: RBMTrainState,
              train_loader,
              num_epochs: int,
              rng: PRNGKey,
              pcd_reset = 5,
              scheduler=None):
    metrics = {}
    for epoch in range(num_epochs):
        # initialize persistent chain once per epoch
        rng, sk = jax.random.split(rng)
        # shape = (batch_size, n_visible), grab from the first batch
        first_batch = next(iter(train_loader))
        v_persistent = jax.random.bernoulli(sk, p=0.5,
                                            shape=first_batch.shape).astype(jnp.float32)

        tot_loss, n_batches = 0.0, 0
        for b_idx, data in enumerate(train_loader):
            # only reset if pcd_reset is an integer
            if (pcd_reset is not None) and (b_idx % pcd_reset == 0):
                rng, sk = jax.random.split(rng)
                v_persistent = jax.random.bernoulli(sk, p=0.5,
                                                    shape=data.shape).astype(jnp.float32)

            state, loss, v_persistent, rng = train_step(state,
                                                        data,
                                                        v_persistent,
                                                        rng)
            tot_loss += loss
            n_batches += 1

        avg = tot_loss / n_batches
        metrics[epoch] = {"free_energy_loss": float(avg)}
        print(f"Epoch [{epoch+1}/{num_epochs}] – FE-loss: {avg:.4f}")
    return state, metrics, rng
