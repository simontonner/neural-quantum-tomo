# THIS FILE CONTAINS ADAPTATIONS THAT HAVE NOT YET BEEN REVIEWED BY HUMAN EYES!

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable

import numpy as np
import torch


_VALID_BASIS_LETTERS = set("XYZ")
LoaderFn = Callable[[Path], Tuple[np.ndarray, List[str], Dict[str, Dict[str, Any]]]]

def _basis_tuple(bases: List[str]) -> Tuple[str, ...]:
    up = tuple(b.upper() for b in bases)
    if any(b not in _VALID_BASIS_LETTERS for b in up):
        raise ValueError(f"Invalid basis letters {bases!r}; only X,Y,Z allowed.")
    return up

def _ensure_values(values: np.ndarray, ctx: str) -> np.ndarray:
    if not isinstance(values, np.ndarray) or values.ndim != 2:
        raise ValueError(f"{ctx}: values must be a 2D numpy array.")
    if values.size == 0:
        raise ValueError(f"{ctx}: empty values array.")
    if values.dtype != np.uint8:
        try:
            values = values.astype(np.uint8, copy=False)
        except Exception as e:
            raise ValueError(f"{ctx}: cannot cast values to uint8: {e}")
    if (~np.isin(values, (0, 1))).any():
        raise ValueError(f"{ctx}: values must be only 0/1.")
    return values


class MeasurementDataset:
    """
    Minimal dataset consuming measurement files via a user-supplied loader.

    Loader contract:
        load_fn(path) -> (values, bases, headers)
            values  : np.ndarray, shape (M, nqubits), dtype uint8 in {0,1}
            bases   : list[str] of length nqubits, each in {X,Y,Z} (any case)
            headers : Dict[str, Dict[str, Any]] with at least a 'state' dict

    Parameters are taken strictly from headers['state'] and limited to `system_param_keys`.
    If `system_param_keys` is None or empty, no params are extracted and `system_params=None`.

    New (optional, backwards compatible):
        samples_per_file : iterable of ints, same length as file_paths.
            For each file i, only the first samples_per_file[i] rows are kept.
            If None, full files are used (old behaviour).
    """

    def __init__(
            self,
            file_paths: Iterable[Path],
            load_fn: LoaderFn,
            system_param_keys: Optional[List[str]] = None,
            samples_per_file: Optional[Iterable[int]] = None,   # <-- NEW
    ):
        paths = [Path(p) for p in file_paths]
        if not paths:
            raise ValueError("No measurement files provided.")
        self.system_param_keys = list(system_param_keys) if system_param_keys else []

        # NEW: normalize samples_per_file to a list aligned with paths
        if samples_per_file is not None:
            samples_list = list(samples_per_file)
            if len(samples_list) != len(paths):
                raise ValueError(
                    "samples_per_file must have the same length as file_paths."
                )
        else:
            samples_list = [None] * len(paths)

        per_file: List[Dict[str, Any]] = []
        fixed_bases_seen = set()
        nqubits_global: Optional[int] = None

        for p, max_rows in zip(paths, samples_list):
            values_np, bases_list, headers = load_fn(p)
            ctx = f"{p.name}"
            values_np = _ensure_values(values_np, ctx)

            # NEW: per-file truncation if max_rows is given
            if max_rows is not None:
                if max_rows < 0:
                    raise ValueError("samples_per_file entries must be non-negative.")
                if max_rows < values_np.shape[0]:
                    values_np = values_np[:max_rows]

            basis_t = _basis_tuple(bases_list)

            nqubits = values_np.shape[1]
            if nqubits != len(basis_t):
                raise ValueError(f"{ctx}: values width ({nqubits}) != len(bases) ({len(basis_t)}).")

            if nqubits_global is None:
                nqubits_global = nqubits
            elif nqubits_global != nqubits:
                raise ValueError(f"Inconsistent nqubits across files: {nqubits_global} vs {nqubits} in {p}.")

            state_params: Dict[str, float] = {}
            if self.system_param_keys:
                headers_lc = {h.lower(): d for h, d in headers.items()}
                if "state" not in headers_lc or not isinstance(headers_lc["state"], dict):
                    raise ValueError(f"{ctx}: loader headers must contain a 'state' dict.")
                state = headers_lc["state"]
                for k in self.system_param_keys:
                    if k not in state:
                        raise KeyError(f"{ctx}: missing 'state.{k}' in header.")
                    try:
                        state_params[k] = float(state[k])
                    except Exception:
                        raise ValueError(f"{ctx}: 'state.{k}' must be numeric; got {state[k]!r}.")

            fixed_bases_seen.add(basis_t)
            per_file.append(
                dict(
                    path=p,
                    values_np=values_np,
                    basis=basis_t,
                    state_params=state_params,
                    nrows=int(values_np.shape[0]),
                )
            )

        # NEW: keep track of how many rows we actually used per file
        self.samples_per_file: List[int] = [info["nrows"] for info in per_file]

        assert nqubits_global is not None
        self.num_qubits = nqubits_global

        # Homogeneous vs mixed across files (fixed basis per file)
        if len(fixed_bases_seen) == 1:
            self.is_mixed = False
            self.implicit_basis: Optional[Tuple[str, ...]] = next(iter(fixed_bases_seen))
            bases_list: Optional[List[Tuple[str, ...]]] = None
        else:
            self.is_mixed = True
            self.implicit_basis = None
            bases_list = []

        # Stack tensors
        values_tensors: List[torch.Tensor] = []
        params_accum: Dict[str, List[float]] = {k: [] for k in self.system_param_keys}

        for info in per_file:
            v = torch.from_numpy(info["values_np"])  # (m, nqubits) uint8
            m = int(v.shape[0])
            values_tensors.append(v)

            if self.is_mixed and bases_list is not None:
                bases_list.extend([info["basis"]] * m)

            for k in self.system_param_keys:
                params_accum[k].extend([info["state_params"][k]] * m)

        self.values = torch.vstack(values_tensors).to(torch.uint8)  # (N, nqubits)
        self.bases: Optional[List[Tuple[str, ...]]] = bases_list

        # Per-parameter tensors and stacked matrix (only if requested)
        if self.system_param_keys:
            self.params_dict: Dict[str, torch.Tensor] = {
                k: torch.tensor(vs, dtype=torch.float32) for k, vs in params_accum.items()
            }
            self.system_params = torch.stack(
                [self.params_dict[k] for k in self.system_param_keys], dim=-1
            )  # (N, d)
        else:
            self.params_dict = {}
            self.system_params = None  # yields None in the loader

        # z_mask
        N = int(self.values.shape[0])
        if self.is_mixed:
            self.z_mask = torch.tensor([all(b == "Z" for b in row) for row in self.bases], dtype=torch.bool)
        else:
            all_z = all(b == "Z" for b in self.implicit_basis)
            self.z_mask = torch.full((N,), bool(all_z), dtype=torch.bool)

    def __len__(self) -> int:
        return int(self.values.shape[0])


class MeasurementLoader:
    """
    Minibatch wrapper around MeasurementDataset.

    Yields:
        batch_values        : (B, nqubits) uint8
        batch_bases         : list[Tuple[str,...]]  (always materialized)
        batch_system_params : (B, d) float32 or None
    """
    def __init__(
            self,
            dataset: MeasurementDataset,
            batch_size: int = 128,
            shuffle: bool = True,
            drop_last: bool = False,
            rng: Optional[torch.Generator] = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.ds = dataset
        self.bs = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.gen = rng or torch.Generator().manual_seed(0)

        N = len(self.ds)
        self.slice_bounds = [
            (i, i + self.bs)
            for i in range(0, N, self.bs)
            if not self.drop_last or (i + self.bs) <= N
        ]

    def __len__(self) -> int:
        return len(self.slice_bounds)

    def __iter__(self):
        N = len(self.ds)
        self.order = torch.randperm(N, generator=self.gen) if self.shuffle else torch.arange(N)
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= len(self.slice_bounds):
            raise StopIteration

        start, end = self.slice_bounds[self._idx]
        self._idx += 1

        idxs = self.order[start:end]
        values = self.ds.values[idxs]

        if self.ds.bases is not None:
            bases = [self.ds.bases[int(i)] for i in idxs]
        else:
            if self.ds.implicit_basis is None:
                raise RuntimeError("Homogeneous dataset without implicit_basis.")
            bases = [self.ds.implicit_basis for _ in range(values.shape[0])]

        sys = self.ds.system_params[idxs] if self.ds.system_params is not None else None
        return values, bases, sys
