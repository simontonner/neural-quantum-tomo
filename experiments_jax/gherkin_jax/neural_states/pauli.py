from math import sqrt
import jax.numpy as jnp
from ..config import CDTYPE

def create_dict(**overrides):
    """Return {X,Y,Z} single-qubit unitaries as complex128."""
    inv_sqrt2 = 1.0 / sqrt(2.0)
    X = inv_sqrt2 * jnp.array([[1+0j, 1+0j],[1+0j, -1+0j]], dtype=CDTYPE)
    Y = inv_sqrt2 * jnp.array([[1+0j, 0-1j],[1+0j, 0+1j]], dtype=CDTYPE)
    Z = jnp.array([[1+0j, 0+0j],[0+0j, 1+0j]], dtype=CDTYPE)
    U = {"X": X, "Y": Y, "Z": Z}
    for name, mat in overrides.items():
        U[name] = as_complex_unitary(mat)
    return U

def as_complex_unitary(U):
    U_t = jnp.asarray(U)
    if U_t.ndim != 2 or U_t.shape != (2, 2):
        raise ValueError(f"as_complex_unitary expects (2,2), got {tuple(U_t.shape)}")
    return U_t.astype(CDTYPE)
