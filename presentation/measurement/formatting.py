import jax.numpy as jnp
from typing import List, Tuple


def format_bytes(num_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def bitstring_to_int(bitstring: jnp.ndarray) -> jnp.ndarray:
    powers = 2 ** jnp.arange(bitstring.shape[-1] - 1, -1, -1)
    return jnp.sum(bitstring * powers, axis=-1).astype(jnp.int32)


def int_to_bitstring(indices: jnp.ndarray, num_bits: int) -> jnp.ndarray:
    powers = 2 ** jnp.arange(num_bits - 1, -1, -1)
    bits = (indices[..., None] & powers) > 0
    return bits.astype(jnp.uint8)


def sample_to_filestring(bitstring: jnp.ndarray, field_strength: float) -> str:
    bitstring_str = ''.join(str(int(b)) for b in bitstring)
    return f"{bitstring_str} {field_strength}"


def filestring_to_sample_tensor(filestring: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    bit_part, field_part = filestring.strip().split()

    bit_tensor = jnp.array([int(b) for b in bit_part], dtype=jnp.uint8)
    field_tensor = jnp.array(float(field_part), dtype=jnp.float32)

    return bit_tensor, field_tensor
