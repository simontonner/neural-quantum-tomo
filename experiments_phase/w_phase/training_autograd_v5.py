#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import re


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


# =========================================================
# TXT IO HELPERS
# =========================================================

_INT_RGX = re.compile(r'^[+-]?\d+\Z')
_FLOAT_RGX = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z')
_VALID_BASIS_LETTERS = set("XYZ")

LoaderFn = Callable[[Path], Tuple[np.ndarray, List[str], Dict[str, Dict[str, Any]]]]


def _format_header_txt(name: str, field_dict: dict) -> str:
    label = name.upper()
    parts = [label]
    for key, value in field_dict.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.2f}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def _parse_header_txt(header: str) -> tuple[str, Dict[str, Any]]:
    name_str, fields_str = header.split("|", 1)
    name = name_str.strip().lower()

    field_dict = {}
    for field_str in fields_str.split("|"):
        key_str, value_str = field_str.split("=", 1)
        key = key_str.strip()
        value = value_str.strip()

        if _INT_RGX.match(value):
            field_dict[key] = int(value)
        elif _FLOAT_RGX.match(value):
            field_dict[key] = float(value)
        else:
            field_dict[key] = value

    return name, field_dict


def load_state_txt(file_path: Path) -> tuple[np.ndarray, Dict[str, Dict[str, Any]]]:
    with open(file_path, "r") as f:
        state_header_name, state_header_fields = _parse_header_txt(f.readline())
        data = np.loadtxt(f, dtype=float)
        if data.ndim == 1:
            data = data[None, :]

    headers = {state_header_name: state_header_fields}
    amplitudes = (data[:, 0] + 1j * data[:, 1]).astype(np.complex128)
    return amplitudes, headers


def load_measurements_txt(file_path: Path) -> tuple[np.ndarray, list[str], Dict[str, Dict[str, Any]]]:
    with open(file_path, "r") as f:
        state_header_name, state_header_fields = _parse_header_txt(f.readline())
        meas_header_name, meas_header_fields = _parse_header_txt(f.readline())
        measurements = [ln.strip() for ln in f]

    headers = {state_header_name: state_header_fields, meas_header_name: meas_header_fields}

    bases = [ch.upper() for ch in measurements[0]]
    n = len(bases)
    m = len(measurements)

    values = np.empty((m, n), dtype=np.uint8)
    for i, s in enumerate(measurements):
        values[i] = [0 if ch.isupper() else 1 for ch in s]

    return values, bases, headers


# =========================================================
# GENERIC DATASET
# =========================================================


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
    """

    def __init__(
        self,
        file_paths: Iterable[Path],
        load_fn: LoaderFn,
        system_param_keys: Optional[List[str]] = None,
        samples_per_file: Optional[Iterable[int]] = None,
    ):
        paths = [Path(p) for p in file_paths]
        if not paths:
            raise ValueError("No measurement files provided.")
        self.system_param_keys = list(system_param_keys) if system_param_keys else []

        if samples_per_file is not None:
            samples_list = list(samples_per_file)
            if len(samples_list) != len(paths):
                raise ValueError("samples_per_file must have same length as file_paths.")
        else:
            samples_list = [None] * len(paths)

        per_file: List[Dict[str, Any]] = []
        fixed_bases_seen = set()
        nqubits_global: Optional[int] = None

        for p, max_rows in zip(paths, samples_list):
            values_np, bases_list, headers = load_fn(p)
            ctx = f"{p.name}"
            values_np = _ensure_values(values_np, ctx)

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

        self.samples_per_file: List[int] = [info["nrows"] for info in per_file]

        assert nqubits_global is not None
        self.num_qubits = nqubits_global

        if len(fixed_bases_seen) == 1:
            self.is_mixed = False
            self.implicit_basis: Optional[Tuple[str, ...]] = next(iter(fixed_bases_seen))
            bases_list2: Optional[List[Tuple[str, ...]]] = None
        else:
            self.is_mixed = True
            self.implicit_basis = None
            bases_list2 = []

        values_tensors: List[torch.Tensor] = []
        params_accum: Dict[str, List[float]] = {k: [] for k in self.system_param_keys}

        for info in per_file:
            v = torch.from_numpy(info["values_np"])
            m = int(v.shape[0])
            values_tensors.append(v)

            if self.is_mixed and bases_list2 is not None:
                bases_list2.extend([info["basis"]] * m)

            for k in self.system_param_keys:
                params_accum[k].extend([info["state_params"][k]] * m)

        self.values = torch.vstack(values_tensors).to(torch.uint8)
        self.bases: Optional[List[Tuple[str, ...]]] = bases_list2

        if self.system_param_keys:
            self.params_dict: Dict[str, torch.Tensor] = {
                k: torch.tensor(vs, dtype=torch.float32) for k, vs in params_accum.items()
            }
            self.system_params = torch.stack([self.params_dict[k] for k in self.system_param_keys], dim=-1)
        else:
            self.params_dict = {}
            self.system_params = None

        N = int(self.values.shape[0])
        if self.is_mixed:
            self.z_mask = torch.tensor([all(b == "Z" for b in row) for row in self.bases], dtype=torch.bool)
        else:
            all_z = all(b == "Z" for b in self.implicit_basis)
            self.z_mask = torch.full((N,), bool(all_z), dtype=torch.bool)

    def __len__(self) -> int:
        return int(self.values.shape[0])


# =========================================================
# BALANCED STRATIFIED CD LOADER
# =========================================================

class StratifiedCDMeasurementLoader:
    """
    Balanced loader for CD-based multi-basis tomography.

    Each step contains:
      - one positive minibatch from every basis in the dataset
      - one Z-only minibatch for CD-k initialization

    Returned tuple:
      pos_values : (B_total, nqubits) uint8
      neg_values : (B_z, nqubits) uint8
      pos_bases  : list[Tuple[str,...]]
      pos_sys    : (B_total, d) or None
      neg_sys    : (B_z, d) or None
    """

    def __init__(
        self,
        dataset: MeasurementDataset,
        batch_size_per_basis: int = 128,
        shuffle: bool = True,
        drop_last: bool = True,
        gen: Optional[torch.Generator] = None,
    ):
        if batch_size_per_basis <= 0:
            raise ValueError("batch_size_per_basis must be positive.")
        if dataset.bases is None:
            raise ValueError("StratifiedCDMeasurementLoader requires a mixed dataset.")

        self.ds = dataset
        self.bs = int(batch_size_per_basis)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.gen = gen or torch.Generator().manual_seed(0)

        seen: List[Tuple[str, ...]] = []
        for b in self.ds.bases:
            if b not in seen:
                seen.append(b)

        self.z_basis = tuple("Z" for _ in range(self.ds.num_qubits))
        if self.z_basis not in seen:
            raise ValueError("Dataset must contain an all-Z basis for CD initialization.")

        self.basis_order = [self.z_basis] + [b for b in seen if b != self.z_basis]

        idxs_by_basis: Dict[Tuple[str, ...], List[int]] = {b: [] for b in self.basis_order}
        for i, b in enumerate(self.ds.bases):
            idxs_by_basis[b].append(i)

        self.idxs_by_basis = {b: torch.tensor(v, dtype=torch.long) for b, v in idxs_by_basis.items()}

        if self.drop_last:
            self.steps_per_epoch = min(int(idxs.numel() // self.bs) for idxs in self.idxs_by_basis.values())
        else:
            self.steps_per_epoch = min(int(np.ceil(idxs.numel() / self.bs)) for idxs in self.idxs_by_basis.values())

        if self.steps_per_epoch <= 0:
            raise ValueError("No balanced steps available. Reduce batch_size_per_basis.")

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _permute_basis_indices(self, idxs: torch.Tensor) -> torch.Tensor:
        if not self.shuffle:
            return idxs.clone()
        perm = torch.randperm(int(idxs.numel()), generator=self.gen)
        return idxs[perm]

    def iter_epoch(self):
        work_by_basis = {b: self._permute_basis_indices(idxs) for b, idxs in self.idxs_by_basis.items()}

        if self.drop_last:
            usable = self.steps_per_epoch * self.bs
            work_by_basis = {b: idxs[:usable] for b, idxs in work_by_basis.items()}

        for step in range(self.steps_per_epoch):
            pos_values_parts = []
            pos_bases: List[Tuple[str, ...]] = []
            pos_sys_parts = [] if self.ds.system_params is not None else None
            z_values = None
            z_sys = None

            for b in self.basis_order:
                idxs = work_by_basis[b]
                start = step * self.bs
                end = start + self.bs
                batch_idxs = idxs[start:end]

                if batch_idxs.numel() < self.bs:
                    if self.drop_last:
                        raise RuntimeError("Incomplete batch despite drop_last=True.")
                    raise StopIteration

                vals = self.ds.values[batch_idxs]
                pos_values_parts.append(vals)
                pos_bases.extend([b] * int(vals.shape[0]))

                if pos_sys_parts is not None:
                    sys = self.ds.system_params[batch_idxs]
                    pos_sys_parts.append(sys)
                else:
                    sys = None

                if b == self.z_basis:
                    z_values = vals
                    z_sys = sys

            pos_values = torch.vstack(pos_values_parts).to(torch.uint8)
            pos_sys = torch.vstack(pos_sys_parts) if pos_sys_parts is not None else None
            neg_values = z_values.clone()
            neg_sys = z_sys.clone() if z_sys is not None else None

            yield pos_values, neg_values, pos_bases, pos_sys, neg_sys


# =========================================================
# UNITARIES AND BASIS ROTATION HELPERS
# =========================================================

def create_dict():
    norm = 1.0 / sqrt(2.0)
    X = norm * torch.tensor([[1 + 0j, 1 + 0j], [1 + 0j, -1 + 0j]], dtype=torch.cdouble, device=DEVICE)
    Y = norm * torch.tensor([[1 + 0j, -1j], [1 + 0j, 1j]], dtype=torch.cdouble, device=DEVICE)
    Z = torch.eye(2, dtype=torch.cdouble, device=DEVICE)
    return {"X": X.contiguous(), "Y": Y.contiguous(), "Z": Z.contiguous()}


def as_complex_unitary(U, device: torch.device):
    if torch.is_tensor(U):
        return U.to(device=device, dtype=torch.cdouble).contiguous()
    U_t = torch.tensor(U, device=device)
    return U_t.to(dtype=torch.cdouble, device=device).contiguous()


def _rotate_basis_state(nn_state, basis, states, unitaries=None):
    device = nn_state.device
    n_vis = nn_state.num_visible
    basis_seq = list(basis)

    if len(basis_seq) != n_vis:
        raise ValueError(f"_rotate_basis_state: basis length {len(basis_seq)} != num_visible {n_vis}")
    if states.shape[-1] != n_vis:
        raise ValueError(f"_rotate_basis_state: states width {states.shape[1]} != num_visible {n_vis}")

    sites = [i for i, b in enumerate(basis_seq) if b != "Z"]
    if len(sites) == 0:
        v = states.unsqueeze(0)
        Ut = torch.ones(v.shape[:-1], dtype=torch.cdouble, device=device)
        return Ut, v

    src = nn_state.U if unitaries is None else unitaries
    Ulist = [as_complex_unitary(src[basis_seq[i]], device).reshape(2, 2).contiguous() for i in sites]
    Uc = torch.stack(Ulist, dim=0)

    S = len(sites)
    B = states.shape[0]
    C = 2 ** S

    combos = nn_state.generate_hilbert_space(size=S, device=device)

    v = states.unsqueeze(0).repeat(C, 1, 1)
    v[:, :, sites] = combos.unsqueeze(1)
    v = v.contiguous()

    inp_sb = states[:, sites].round().long().T
    outp_csb = v[:, :, sites].round().long().permute(0, 2, 1)
    inp_csb = inp_sb.unsqueeze(0).expand(C, -1, -1)

    s_idx = torch.arange(S, device=device).view(1, S, 1).expand(C, S, B)
    sel = Uc[s_idx, inp_csb, outp_csb]
    Ut = sel.prod(dim=1)

    return Ut.to(torch.cdouble), v


# =========================================================
# BINARY RBM
# =========================================================

class BinaryRBM(nn.Module):
    """Bernoulli/Bernoulli RBM with free energy F(v)."""

    def __init__(self, num_visible, num_hidden=None, zero_weights=False, device: torch.device = DEVICE):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden) if num_hidden else self.num_visible
        self.device = device
        self.initialize_parameters(zero_weights=zero_weights)

    def initialize_parameters(self, zero_weights=False):
        gen_tensor = torch.zeros if zero_weights else torch.randn
        scale = 1.0 / np.sqrt(self.num_visible)

        self.weights = nn.Parameter(
            gen_tensor(self.num_hidden, self.num_visible, device=self.device, dtype=DTYPE) * scale,
            requires_grad=True,
        )
        self.visible_bias = nn.Parameter(
            torch.zeros(self.num_visible, device=self.device, dtype=DTYPE),
            requires_grad=True,
        )
        self.hidden_bias = nn.Parameter(
            torch.zeros(self.num_hidden, device=self.device, dtype=DTYPE),
            requires_grad=True,
        )

    def effective_energy(self, v):
        unsq = False
        if v.dim() < 2:
            v = v.unsqueeze(0)
            unsq = True

        v = v.to(self.weights)
        visible_bias_term = torch.matmul(v, self.visible_bias)
        hid_bias_term = F.softplus(F.linear(v, self.weights, self.hidden_bias)).sum(-1)
        out = -(visible_bias_term + hid_bias_term)
        return out.squeeze(0) if unsq else out

    @torch.no_grad()
    def gibbs_steps(self, k, initial_state, overwrite=False):
        v = (initial_state if overwrite else initial_state.clone()).to(self.weights)
        h = torch.empty(*v.shape[:-1], self.num_hidden, device=self.device, dtype=DTYPE)

        for _ in range(k):
            h_lin = F.linear(v, self.weights, self.hidden_bias)
            h_prob = torch.sigmoid(h_lin)
            h_prob = torch.nan_to_num(h_prob, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            torch.bernoulli(h_prob, out=h)

            v_lin = F.linear(h, self.weights.t(), self.visible_bias)
            v_prob = torch.sigmoid(v_lin)
            v_prob = torch.nan_to_num(v_prob, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            torch.bernoulli(v_prob, out=v)

        return v


# =========================================================
# COMPLEX WAVE FUNCTION (AMPLITUDE + PHASE RBM)
# =========================================================

class ComplexWaveFunction:
    """psi(sigma) = exp(-F_lambda/2) * exp(-i F_mu/2)."""

    def __init__(self, num_visible, num_hidden=None, unitary_dict=None, device: torch.device = DEVICE):
        self.device = device
        self.rbm_am = BinaryRBM(num_visible, num_hidden, device=self.device)
        self.rbm_ph = BinaryRBM(num_visible, num_hidden, device=self.device)

        self.num_visible = self.rbm_am.num_visible
        self.num_hidden = self.rbm_am.num_hidden

        raw = unitary_dict if unitary_dict is not None else create_dict()
        self.U = {k: as_complex_unitary(v, self.device) for k, v in raw.items()}

        self._stop_training = False
        self._max_size = 20

    @property
    def stop_training(self):
        return self._stop_training

    @stop_training.setter
    def stop_training(self, new_val):
        if isinstance(new_val, bool):
            self._stop_training = new_val
        else:
            raise ValueError("stop_training must be bool")

    def reinitialize_parameters(self):
        self.rbm_am.initialize_parameters()
        self.rbm_ph.initialize_parameters()

    def psi_complex_normalized(self, v):
        v = v.to(self.device, dtype=DTYPE)
        E = self.rbm_am.effective_energy(v)
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        logZ = torch.logsumexp(-E, dim=0)
        return torch.exp(((-0.5 * E) - 0.5 * logZ).to(torch.cdouble) + 1j * ph.to(torch.cdouble))

    def generate_hilbert_space(self, size=None, device=None):
        device = self.device if device is None else device
        size = self.num_visible if size is None else int(size)
        if size > self._max_size:
            raise ValueError(f"Hilbert space too large (n={size} > max={self._max_size}).")
        n = 1 << size
        ar = torch.arange(n, device=device, dtype=torch.long)
        shifts = torch.arange(size - 1, -1, -1, device=device, dtype=torch.long)
        return ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)

    def _stable_log_overlap_amp2(self, basis: Tuple[str, ...], states: torch.Tensor,
                                 eps_rot: float = 1e-6, unitaries=None):
        Ut, v = _rotate_basis_state(self, basis, states, unitaries=unitaries)
        F_am = self.rbm_am.effective_energy(v)
        F_ph = self.rbm_ph.effective_energy(v)

        logmag_total = (-0.5 * F_am) + torch.log(Ut.abs().to(DTYPE).clamp_min(1e-300))
        phase_total = (-0.5 * F_ph).to(torch.cdouble) + torch.angle(Ut).to(torch.cdouble)

        M, _ = torch.max(logmag_total, dim=0, keepdim=True)
        scaled_mag = torch.exp((logmag_total - M).to(DTYPE))
        contrib = scaled_mag.to(torch.cdouble) * torch.exp(1j * phase_total)
        S_prime = contrib.sum(dim=0)
        S_abs2 = (S_prime.conj() * S_prime).real.to(DTYPE)
        log_amp2 = (2.0 * M.squeeze(0)).to(DTYPE) + torch.log(S_abs2 + eps_rot)
        return log_amp2

    def _positive_phase_loss(self, samples: torch.Tensor, bases_batch: List[Tuple[str, ...]], eps_rot: float = 1e-6):
        buckets = {}
        for i, row in enumerate(bases_batch):
            buckets.setdefault(tuple(row), []).append(i)

        loss_rot = samples.new_tensor(0.0, dtype=DTYPE)
        loss_z = samples.new_tensor(0.0, dtype=DTYPE)

        for basis_t, idxs in buckets.items():
            idxs_t = torch.tensor(idxs, device=samples.device)
            if any(ch != "Z" for ch in basis_t):
                log_amp2 = self._stable_log_overlap_amp2(basis_t, samples[idxs_t], eps_rot=eps_rot)
                loss_rot = loss_rot - log_amp2.sum().to(DTYPE)
            else:
                Epos = self.rbm_am.effective_energy(samples[idxs_t])
                loss_z = loss_z + Epos.sum()

        return loss_rot + loss_z

    def _negative_phase_loss(self, k: int, neg_init: torch.Tensor):
        with torch.no_grad():
            vk = self.rbm_am.gibbs_steps(k, neg_init, overwrite=True)
        Eneg = self.rbm_am.effective_energy(vk)
        return Eneg.sum(), vk.shape[0]

    def fit(self, loader, epochs=200, k=10, lr=5e-2, log_every=5,
            optimizer=torch.optim.SGD, optimizer_args=None, target=None, space=None,
            print_metrics=True):

        if self.stop_training:
            return {"epoch": []}

        optimizer_args = {} if optimizer_args is None else optimizer_args
        params = list(self.rbm_am.parameters()) + list(self.rbm_ph.parameters())
        opt = optimizer(params, lr=lr, **optimizer_args)

        history = {"epoch": []}
        if target is not None:
            history["Fidelity"] = []
            history["MaxAbsPhaseErr"] = []
            history["MeanAbsPhaseErr"] = []

        if space is None:
            space = self.generate_hilbert_space()

        best_fid = -1.0
        best_state = None

        for ep in range(1, epochs + 1):
            for pos_batch, neg_batch, bases_batch, _, _ in loader.iter_epoch():
                pos_batch = pos_batch.to(self.device, dtype=DTYPE)
                neg_batch = neg_batch.to(self.device, dtype=DTYPE)

                L_pos = self._positive_phase_loss(pos_batch, bases_batch)
                B_pos = float(pos_batch.shape[0])

                L_neg, B_neg = self._negative_phase_loss(k, neg_batch)
                loss = (L_pos / B_pos) - (L_neg / B_neg)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                opt.step()

                if self.stop_training:
                    break

            if target is not None and (ep % log_every == 0):
                with torch.no_grad():
                    fid_val = fidelity(self, target, space=space)
                    ph_stats = phase_error_stats(self, target, space=space)

                history["epoch"].append(ep)
                history["Fidelity"].append(fid_val)
                history["MaxAbsPhaseErr"].append(ph_stats["max_abs_phase_err"])
                history["MeanAbsPhaseErr"].append(ph_stats["mean_abs_phase_err"])

                if fid_val > best_fid:
                    best_fid = fid_val
                    best_state = {
                        "rbm_am": {k: v.detach().clone() for k, v in self.rbm_am.state_dict().items()},
                        "rbm_ph": {k: v.detach().clone() for k, v in self.rbm_ph.state_dict().items()},
                    }

                if print_metrics:
                    print(
                        f"Epoch {ep}: Fidelity = {fid_val:.6f} | "
                        f"max |Δφ| = {ph_stats['max_abs_phase_err']:.6f} | "
                        f"mean |Δφ| = {ph_stats['mean_abs_phase_err']:.6f}"
                    )

            if self.stop_training:
                break

        if best_state is not None:
            self.rbm_am.load_state_dict(best_state["rbm_am"])
            self.rbm_ph.load_state_dict(best_state["rbm_ph"])
            print(f"Restored best checkpoint with fidelity {best_fid:.6f}")

        return history


# =========================================================
# METRICS
# =========================================================

@torch.no_grad()
def fidelity(nn_state, target, space=None):
    if not torch.is_complex(target):
        raise TypeError("fidelity: target must be complex (cdouble)")

    space = nn_state.generate_hilbert_space() if space is None else space

    psi = nn_state.psi_complex_normalized(space).reshape(-1).contiguous()
    tgt = target.to(device=nn_state.device, dtype=torch.cdouble).reshape(-1).contiguous()

    npsi = torch.linalg.vector_norm(psi)
    nt = torch.linalg.vector_norm(tgt)
    if npsi == 0 or nt == 0:
        return 0.0

    psi_n = psi / npsi
    tgt_n = tgt / nt

    inner = (tgt_n.conj() * psi_n).sum()
    return float(inner.abs().pow(2).real)


@torch.no_grad()
def phase_error_stats(nn_state, target, space=None, eps=1e-12):
    if space is None:
        space = nn_state.generate_hilbert_space()

    psi_m = nn_state.psi_complex_normalized(space).reshape(-1).to(torch.cdouble)
    psi_t = target.reshape(-1).to(torch.cdouble)

    psi_m = psi_m / torch.linalg.vector_norm(psi_m)
    psi_t = psi_t / torch.linalg.vector_norm(psi_t)

    ip = torch.sum(psi_t.conj() * psi_m)
    if ip.abs() > 1e-12:
        theta = torch.angle(ip)
    else:
        j = int(torch.argmax(psi_t.abs()))
        theta = torch.angle(psi_m[j]) - torch.angle(psi_t[j])

    psi_m_al = psi_m * torch.exp(-1j * theta)

    bits = space.round().long()
    one_hot_mask = (bits.sum(dim=1) == 1)
    support_mask = one_hot_mask & (psi_t.abs() > eps)

    phi_t = torch.angle(psi_t[support_mask])
    phi_m = torch.angle(psi_m_al[support_mask])

    dphi = torch.remainder(phi_m - phi_t + torch.pi, 2 * torch.pi) - torch.pi
    abs_dphi = dphi.abs()

    return {
        "max_abs_phase_err": float(abs_dphi.max().item()),
        "mean_abs_phase_err": float(abs_dphi.mean().item()),
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    torch.manual_seed(1234)
    np.random.seed(1234)

    batch_size_per_basis = 100
    epochs = 1000
    lr = 5e-2
    k_cd = 10
    log_every = 5

    psi_path = Path("state_vectors/w_phase_state.txt")
    amps_np, _ = load_state_txt(psi_path)
    target_state = torch.tensor(amps_np, dtype=torch.cdouble, device=DEVICE)

    meas_directory = Path("measurements")
    all_paths = [
        meas_directory / "w_phase_ZZZZ_5000.txt",
        meas_directory / "w_phase_XXZZ_5000.txt",
        meas_directory / "w_phase_XYZZ_5000.txt",
        meas_directory / "w_phase_ZXXZ_5000.txt",
        meas_directory / "w_phase_ZXYZ_5000.txt",
        meas_directory / "w_phase_ZZXX_5000.txt",
        meas_directory / "w_phase_ZZXY_5000.txt",
    ]

    ds_all = MeasurementDataset(
        file_paths=all_paths,
        load_fn=load_measurements_txt,
        system_param_keys=None,
    )

    loader = StratifiedCDMeasurementLoader(
        ds_all,
        batch_size_per_basis=batch_size_per_basis,
        shuffle=True,
        drop_last=True,
        gen=torch.Generator().manual_seed(1234),
    )

    U = create_dict()
    nv = ds_all.num_qubits
    nh = nv
    nn_state = ComplexWaveFunction(num_visible=nv, num_hidden=nh, unitary_dict=U, device=DEVICE)
    space = nn_state.generate_hilbert_space()

    history = nn_state.fit(
        loader=loader,
        epochs=epochs,
        k=k_cd,
        lr=lr,
        log_every=log_every,
        optimizer=torch.optim.SGD,
        optimizer_args=None,
        target=target_state,
        space=space,
        print_metrics=True,
    )

    with torch.no_grad():
        psi_m = nn_state.psi_complex_normalized(space).reshape(-1).to(torch.cdouble)
        psi_t = target_state.reshape(-1).to(torch.cdouble)

        psi_m = psi_m / torch.linalg.vector_norm(psi_m)
        psi_t = psi_t / torch.linalg.vector_norm(psi_t)

        ip = torch.sum(psi_t.conj() * psi_m)
        if ip.abs() > 1e-12:
            theta = torch.angle(ip)
        else:
            j = int(torch.argmax(psi_t.abs()))
            theta = torch.angle(psi_m[j]) - torch.angle(psi_t[j])

        psi_m_al = psi_m * torch.exp(-1j * theta)

        phi_m = torch.angle(psi_m_al)
        phi_t = torch.angle(psi_t)

        probs = psi_t.abs().pow(2)
        order = torch.argsort(probs, descending=True)
        cum = torch.cumsum(probs[order], dim=0)
        mass_cut = 0.99
        k_cap = 512
        idx = torch.searchsorted(cum, torch.tensor(mass_cut, device=cum.device)).item()
        k_sel = min(idx + 1, k_cap, probs.numel())
        sel = order[:k_sel]

        phi_m_sel = phi_m[sel]
        phi_t_sel = phi_t[sel]
        phi_diff_sel = torch.remainder(phi_m_sel - phi_t_sel + torch.pi, 2 * torch.pi) - torch.pi

    fig_p, axp = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    axp.plot(range(sel.numel()), phi_t_sel.cpu().numpy(), marker=".", linestyle="", label="target phase")
    axp.plot(range(sel.numel()), phi_m_sel.cpu().numpy(), marker="x", linestyle="", label="model phase (aligned)")
    axp.set_xlabel("basis states (sorted by target mass)")
    axp.set_ylabel("phase [rad]")
    axp.set_title("Phase comparison - top 99% mass")
    axp.grid(True, alpha=0.3)
    axp.legend()
    fig_p.tight_layout()

    fig_e, axe = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    axe.plot(range(sel.numel()), phi_diff_sel.cpu().numpy(), marker=".", linestyle="", label="Δphase (wrapped)")
    axe.axhline(0.0, linewidth=1.0)
    axe.set_xlabel("basis states (sorted by target mass)")
    axe.set_ylabel("Δphase [rad] in [-π, π]")
    axe.set_title("Phase error after global-phase alignment")
    axe.grid(True, alpha=0.3)
    axe.legend()
    fig_e.tight_layout()

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=140)
    ax.plot(history.get("epoch", []), history.get("Fidelity", []), marker="o", label="Fidelity")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$|\langle \psi_t \mid \psi \rangle|^2$")
    ax.set_title("CD-RBM Tomography - training fidelity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    plt.show()
