import jax.numpy as jnp


def get_cosine_schedule(T_high: float,
                        T_low: float,
                        n_steps: int) -> jnp.ndarray:
    """Cosine annealed temperature schedule."""
    steps = jnp.arange(n_steps, dtype=jnp.float32)
    cos   = 0.5 * (1 + jnp.cos(jnp.pi * steps / (n_steps - 1)))
    return T_low + (T_high - T_low) * cos
