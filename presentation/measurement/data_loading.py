from pathlib import Path
from typing import Tuple

from jax.random import split, permutation, PRNGKey
import jax.numpy as jnp
from tqdm import tqdm

from .formatting import filestring_to_sample_tensor


class MixedDataLoader:
    def __init__(self,
                 bits: jnp.ndarray,
                 field: jnp.ndarray,
                 batch_size: int = 128,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 rng: PRNGKey = None):
        """
        bits:  shape (N, num_qubits), dtype uint8
        field: shape (N,),            dtype float32
        """
        assert bits.shape[0] == field.shape[0], "Mismatched sample counts"
        self.bits         = bits
        self.field        = field
        self.dataset_size = bits.shape[0]
        self.batch_size   = batch_size
        self.shuffle      = shuffle
        self.drop_last    = drop_last
        self.rng          = PRNGKey(0) if rng is None else rng

        # precompute slice boundaries
        self.slice_boundaries = [
            (i, i + batch_size)
            for i in range(0, self.dataset_size, batch_size)
            if not drop_last or (i + batch_size) <= self.dataset_size
        ]

    def __iter__(self):
        self.rng, rng_shuffle = split(self.rng)
        if self.shuffle:
            self.order = permutation(rng_shuffle, self.dataset_size)
        else:
            self.order = jnp.arange(self.dataset_size)
        self.slice_idx = 0
        return self

    def __next__(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.slice_idx >= len(self.slice_boundaries):
            raise StopIteration
        start, end = self.slice_boundaries[self.slice_idx]
        self.slice_idx += 1
        idxs = self.order[start:end]
        batch_bits  = self.bits[idxs]   # (batch_size, num_qubits)
        batch_field = self.field[idxs]  # (batch_size,)
        return batch_bits, batch_field

    def __len__(self) -> int:
        return len(self.slice_boundaries)


def load_measurements(file_path: Path) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Reads a file where each non-empty line is:
        010110 2.0
    and returns two arrays:
      - bits:  shape (N, num_qubits), dtype uint8
      - field: shape (N,),            dtype float32
    """
    with open(file_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    bits_list: list[jnp.ndarray]  = []
    field_list: list[jnp.ndarray] = []

    for ln in tqdm(lines, desc="Parsing measurements"):
        b, h = filestring_to_sample_tensor(ln)
        bits_list.append(b)
        field_list.append(h)

    bits  = jnp.stack(bits_list,  axis=0)
    field = jnp.stack(field_list, axis=0)
    return bits, field