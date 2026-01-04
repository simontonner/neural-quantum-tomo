from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp

# Global dtypes
DTYPE  = jnp.float64
CDTYPE = jnp.complex128
