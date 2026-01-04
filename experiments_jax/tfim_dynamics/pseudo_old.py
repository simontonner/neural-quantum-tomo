# =====================================================================
# tomography_data.py — minimal dataset + grouped loader (JAX-friendly)
# =====================================================================
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
import re
import numpy as np

# If you want jax arrays downstream, keep this import. Otherwise, replace jnp with np.
import jax.numpy as jnp
import jax

DTYPE = jnp.float64
CDTYPE = jnp.complex128


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _parse_basis_code_from_filename(name: str, prefix: str) -> str:
    """
    Accept 'w_phase_XXZZ_5000.txt' -> 'XXZZ'.
    Basis code must be uppercase [XYZI]+.
    """
    stem = Path(name).stem  # e.g. 'w_phase_XXZZ_5000'
    if not stem.startswith(prefix):
        raise ValueError(f"Bad file name (prefix): {name}")
    tail = stem[len(prefix):]  # 'XXZZ_5000'
    if "_" not in tail:
        raise ValueError(f"Bad file name (missing shots part): {name}")
    code = tail.rsplit("_", 1)[0]
    if not re.fullmatch(r"[XYZI]+", code):
        raise ValueError(f"Basis code must be [XYZI]+, got '{code}' from {name}")
    return code


# ---------------------------------------------------------------------
# Dataset: per-basis measurements directory
# ---------------------------------------------------------------------
class PerBasisTomographyDataset:
    """
    Load from a directory layout:
      measurements/
        w_phase_state.txt                -> target psi: two columns (Re, Im)
        w_phase_<BASIS>_<shots>.txt      -> outcomes per line, case-encoded bits

    Encoding example:
      basis "ZZZX", line "ZzZX"  -> [0,1,0,0]

    Exposes:
      - train_samples : jnp.float64 array, shape (N, n)
      - train_bases   : list[tuple[str,...]] length N
      - target_state  : jnp.complex128 array, shape (2**n,)
      - _unique_bases : list[tuple[str,...]]
      - _z_indices    : jnp.int32 array of indices for Z-only rows
      - _groups       : dict[basis_tuple -> np.ndarray of row indices]
    """
    def __init__(
            self,
            directory: str = "measurements",
            state_filename: str = "w_phase_state.txt",
            file_prefix: str = "w_phase_",
    ):
        d = Path(directory)
        if not d.is_dir():
            raise FileNotFoundError(f"Directory not found: {d}")

        # --- target psi ---
        psi_path = d / state_filename
        if not psi_path.exists():
            raise FileNotFoundError(f"Missing state file: {psi_path}")
        psi_np = np.loadtxt(str(psi_path), dtype="float64")
        if psi_np.ndim != 2 or psi_np.shape[1] != 2:
            raise ValueError("State file must have two columns: Re Im.")
        self.target_state = jnp.asarray(psi_np[:, 0] + 1j * psi_np[:, 1], dtype=CDTYPE)

        # --- per-basis files ---
        per_basis_files = [
            p for p in d.glob("*.txt")
            if p.name != state_filename and p.name.startswith(file_prefix)
        ]
        if not per_basis_files:
            raise FileNotFoundError(f"No per-basis files matching '{file_prefix}*' in {d}")

        samples: List[np.ndarray] = []
        bases_rows: List[Tuple[str, ...]] = []
        unique_bases: Dict[Tuple[str, ...], None] = {}

        for p in sorted(per_basis_files):
            bcode = _parse_basis_code_from_filename(p.name, prefix=file_prefix)
            basis = tuple(list(bcode))  # e.g. ("Z","Z","Z","X")

            with open(p, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if not lines:
                continue

            n = len(basis)
            for ln in lines:
                if len(ln) != n:
                    raise ValueError(f"Line length != basis width in {p}: '{ln}'")
                # case-encode: uppercase->0, lowercase->1
                bits = []
                for ch in ln:
                    if not ch.isalpha():
                        raise ValueError(f"Illegal char '{ch}' in {p}")
                    bits.append(0 if ch.isupper() else 1)
                samples.append(np.asarray(bits, dtype=np.float64))
                bases_rows.append(basis)

            unique_bases[basis] = None

        if not samples:
            raise ValueError("No samples read from per-basis files.")

        self.train_samples = jnp.asarray(np.stack(samples, axis=0), dtype=DTYPE)
        self.train_bases: List[Tuple[str, ...]] = bases_rows
        self._unique_bases = list(unique_bases.keys())

        # --- guardrails / inferred n ---
        widths = {len(row) for row in self.train_bases}
        if len(widths) != 1:
            raise ValueError("Inconsistent basis widths across files.")
        n = next(iter(widths))
        if self.train_samples.shape[1] != n:
            raise ValueError("Sample width != basis width.")
        expected_dim = 1 << n
        if int(self.target_state.size) != expected_dim:
            raise ValueError(f"State length {int(self.target_state.size)} != 2^{n} ({expected_dim}).")

        # --- Z-only row indices (used as negative pool in CD-k) ---
        z_mask = np.array([all(ch == "Z" for ch in row) for row in self.train_bases], dtype=bool)
        self._z_indices = jnp.asarray(np.nonzero(z_mask)[0], dtype=jnp.int32)
        if int(self._z_indices.size) == 0:
            raise ValueError("No Z-only rows found; needed for negative sampling.")

        # --- group rows by basis ---
        self._groups: Dict[Tuple[str, ...], np.ndarray] = {}
        for i, row in enumerate(self.train_bases):
            self._groups.setdefault(tuple(row), []).append(i)
        for k in list(self._groups.keys()):
            self._groups[k] = np.asarray(self._groups[k], dtype=np.int32)

    # --- light API ---
    def __len__(self) -> int: return int(self.train_samples.shape[0])
    def num_visible(self) -> int: return int(self.train_samples.shape[1])
    def z_indices(self) -> jnp.ndarray: return self._z_indices.copy()
    def eval_bases(self) -> List[Tuple[str, ...]]: return list(self._unique_bases)
    def target(self) -> jnp.ndarray: return self.target_state
    def groups(self) -> Dict[Tuple[str, ...], np.ndarray]: return self._groups


# ---------------------------------------------------------------------
# Grouped loader: yields homogeneous-basis batches
# ---------------------------------------------------------------------
class RBMTomographyLoaderGrouped:
    """
    Yields:
      (pos_batch, neg_batch, basis_tuple, is_z)

    - pos_batch: jnp.float64, shape (~pos_bs, n)
    - neg_batch: jnp.float64, shape (neg_bs, n), sampled from Z-only rows
    - basis_tuple: tuple[str,...], current basis of pos_batch
    - is_z: bool, True if basis is all 'Z'
    """
    def __init__(
            self,
            dataset: PerBasisTomographyDataset,
            pos_batch_size: int = 100,
            neg_batch_size: Optional[int] = None,
            seed: Optional[int] = None,
    ):
        self.ds = dataset
        self.pos_bs = int(pos_batch_size)
        self.neg_bs = int(neg_batch_size or pos_batch_size)
        self._key = jax.random.PRNGKey(0 if seed is None else int(seed))

        n = self.ds.num_visible()
        for key in self.ds.groups().keys():
            if len(key) != n:
                raise ValueError("RBMTomographyLoaderGrouped: inconsistent basis widths in dataset")
        if self.ds.z_indices().size == 0:
            raise ValueError("RBMTomographyLoaderGrouped: Z-only pool is empty")

        self._plan_batches()

    def __len__(self) -> int:
        return self._planned_batches

    def _plan_batches(self):
        planned = []
        for _, idxs in self.ds.groups().items():
            planned.append(int(np.ceil(idxs.shape[0] / self.pos_bs)))
        self._planned_batches = int(np.sum(planned))

    def iter_epoch(self):
        """Yield homogeneous-basis batches in random basis order, once per epoch."""
        self._key, k_order, k_perm, k_neg = jax.random.split(self._key, 4)

        bases = list(self.ds.groups().keys())
        order = np.asarray(jax.random.permutation(k_order, jnp.arange(len(bases))).tolist(), dtype=int)
        bases = [bases[i] for i in order]

        total_batches = len(self)
        z_pool = self.ds.z_indices()
        pool_len = int(z_pool.size)
        # pre-sample all negatives for the epoch
        neg_choices = jax.random.randint(k_neg, shape=(total_batches * self.neg_bs,), minval=0, maxval=pool_len)
        neg_rows = z_pool[neg_choices]
        neg_samples_all = self.ds.train_samples[neg_rows].astype(DTYPE)

        nb_cursor = 0
        for basis in bases:
            idxs = self.ds.groups()[tuple(basis)]
            k_perm, k_bucket = jax.random.split(k_perm)
            perm = np.asarray(jax.random.permutation(k_bucket, jnp.arange(idxs.shape[0])).tolist(), dtype=int)
            idxs = idxs[perm]

            for start in range(0, idxs.shape[0], self.pos_bs):
                end = min(start + self.pos_bs, idxs.shape[0])
                pos_rows = idxs[start:end]
                pos_batch = self.ds.train_samples[pos_rows].astype(DTYPE)

                nb_start = nb_cursor * self.neg_bs
                nb_end = nb_start + self.neg_bs
                neg_batch = neg_samples_all[nb_start:nb_end]
                nb_cursor += 1

                is_z = all(ch == "Z" for ch in basis)
                yield pos_batch, neg_batch, tuple(basis), bool(is_z)


# ---------------------------------------------------------------------
# Minimal usage (how it’s called). Replace the body with your training.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Configure your measurement dir + naming
    ds = PerBasisTomographyDataset(
        directory="measurements",
        state_filename="w_phase_state.txt",
        file_prefix="w_phase_",
    )

    loader = RBMTomographyLoaderGrouped(
        ds,
        pos_batch_size=100,
        neg_batch_size=100,
        seed=1234,
    )

    print(f"samples={len(ds)} | bases={len(ds.eval_bases())} | batches/epoch={len(loader)}")
    print(f"n_visible={ds.num_visible()} | target_dim={ds.target().size}")

    # Example: iterate one epoch and show the first batch shapes
    for i, (pos, neg, basis, is_z) in enumerate(loader.iter_epoch()):
        print(f"[{i:03d}] basis={''.join(basis)} (is_z={is_z}) | pos={pos.shape} | neg={neg.shape}")
        # -> hand off to your training step, e.g. trainer.step(pos, neg, basis, is_z)
        if i == 0:
            break
