from pathlib import Path
from typing import Dict
from collections import defaultdict

from jax.random import split, permutation, PRNGKey
import jax.numpy as jnp


class MultiBasisDataLoader:

    def __init__(self,
                 data: Dict[str, jnp.ndarray],
                 batch_size: int = 128,
                 basis_encoding: Dict[str, int] = None,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 rng: PRNGKey = None):

        lengths = [v.shape[0] for v in data.values()]
        if len(set(lengths)) != 1:
            raise ValueError(f"Inconsistent sample counts across bases: {lengths}")

        self.bases = list(data.keys())
        self.arrays = list(data.values())
        self.num_qubits = self.arrays[0].shape[1]
        self.encoded_bases = jnp.array([[basis_encoding[c] for c in b] for b in self.bases], dtype=jnp.int8)

        self.dataset_size = lengths[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = PRNGKey(0) if rng is None else rng

        # boundary tuples are only created based on a specified condition, jax truncates overshooting indices
        self.slice_boundaries = [
            (i, i + batch_size)
            for i in range(0, self.dataset_size, batch_size)
            if not drop_last or (i + batch_size) <= self.dataset_size
        ]

    def __iter__(self):
        self.rng, rng_shuffle = split(self.rng)
        self.order = (permutation(rng_shuffle, self.dataset_size) if self.shuffle else jnp.arange(self.dataset_size))
        self.slice_idx = 0
        return self

    def __next__(self):
        if self.slice_idx >= len(self.slice_boundaries):
            raise StopIteration
        s, e = self.slice_boundaries[self.slice_idx]
        self.slice_idx += 1

        base_batches = jnp.stack([arr[self.order[s:e]] for arr in self.arrays], axis=0)
        return base_batches, self.encoded_bases # (num_bases, batch_size, num_qubits), (num_bases, num_qubits)

    def __len__(self):
        return len(self.slice_boundaries)


def load_measurements(folder: str, file_pattern: str = "w_*.txt") -> dict[str, jnp.ndarray]:
    measurements = defaultdict(list)

    for file in Path(folder).glob(file_pattern):
        basis = file.stem.split("_")[2]
        with file.open() as f:
            for line in f:
                measurements[basis].append([1.0 if c.islower() else 0.0 for c in line.strip()])

    return {b: jnp.array(v, dtype=jnp.float32) for b, v in measurements.items()}
