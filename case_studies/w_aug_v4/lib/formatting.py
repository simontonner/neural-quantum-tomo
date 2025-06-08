import jax.numpy as jnp
from typing import List

def bitstring_to_int(bitstring: jnp.ndarray) -> jnp.ndarray:
    powers = 2 ** jnp.arange(bitstring.shape[-1] - 1, -1, -1)
    return jnp.sum(bitstring * powers, axis=-1).astype(jnp.int32)

def int_to_bitstring(indices: jnp.ndarray, num_bits: int) -> jnp.ndarray:
    powers = 2 ** jnp.arange(num_bits - 1, -1, -1)
    bits = (indices[..., None] & powers) > 0
    return bits.astype(jnp.uint8)

def format_bytes(num_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def bitstring_to_filestring(bitstring: jnp.ndarray, measurement_basis: List[str]) -> str:
    result = []
    for bit, op in zip(bitstring, measurement_basis):
        if bit == 0 or bit == '0':
            result.append(op.upper())
        elif bit == 1 or bit == '1':
            result.append(op.lower())
        else:
            result.append('?')
    return ''.join(result)
