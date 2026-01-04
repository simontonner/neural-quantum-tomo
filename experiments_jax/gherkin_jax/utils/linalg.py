from math import prod
from typing import List
import jax.numpy as jnp
from ..config import CDTYPE

def inverse(z: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Safe complex inverse: conj(z) / max(|z|^2, eps)."""
    zz = z.astype(CDTYPE)
    return jnp.conj(zz) / jnp.maximum(jnp.abs(zz) ** 2, eps)

def kron_mult(matrices: List[jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    """
    Apply (⊗_s U_s) to psi without forming the Kronecker explicitly.
    """
    if x.dtype != CDTYPE:
        raise TypeError("x must be complex128")
    if any(m.dtype != CDTYPE for m in matrices):
        raise TypeError("unitaries must be complex128")

    x_cd = x
    L = x_cd.shape[0]
    batch = int(x_cd.size // L)
    y = x_cd.reshape(L, batch)

    n = [m.shape[-1] for m in matrices]
    if prod(n) != L:
        raise ValueError(f"Incompatible sizes: expected leading dim {prod(n)}, got {L}")

    left = L
    for U in reversed(matrices):
        ns = U.shape[-1]
        left //= ns
        y = y.reshape(left, ns, -1)
        y = jnp.einsum('ij,ljm->lim', U, y).reshape(left * ns, -1)

    return y.reshape(*x_cd.shape)
