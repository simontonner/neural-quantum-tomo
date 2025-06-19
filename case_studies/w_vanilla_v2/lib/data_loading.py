from pathlib import Path
from typing import Dict
from collections import defaultdict

from jax.random import split, permutation, PRNGKey
import jax.numpy as jnp
from tqdm import tqdm

from .formatting import filestring_to_tensor


class MixedDataLoader:
    def __init__(self,
                 data: jnp.ndarray,
                 batch_size: int = 128,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 rng: PRNGKey = None):

        self.data = data
        self.dataset_size = data.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = PRNGKey(0) if rng is None else rng

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

    def __next__(self) -> jnp.ndarray:
        if self.slice_idx >= len(self.slice_boundaries):
            raise StopIteration
        start, end = self.slice_boundaries[self.slice_idx]
        self.slice_idx += 1
        batch_idx = self.order[start:end]
        return self.data[batch_idx]   # shape (batch_size, num_qubits, 3)

    def __len__(self) -> int:
        return len(self.slice_boundaries)


def load_measurements(file_path: Path) -> jnp.ndarray:
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    tensors = [filestring_to_tensor(line) for line in tqdm(lines, desc="Parsing measurements")]
    return jnp.stack(tensors, axis=0)
