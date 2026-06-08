#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pooled-CD complex RBM tomography in a project layout closer to the HyperRBM code.

Important:
- training semantics are kept aligned with the working reference:
    * pooled positive minibatches
    * Z-only negative minibatches
    * equal-shot-count check
- optimizer/schedule setup is closer to the modern project:
    * Adam by default
    * optional sigmoid LR schedule
- the code is reorganized so you can swap optimizer / schedules in one place
"""

from __future__ import annotations

from math import ceil, sqrt, prod
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


##### DEVICE AND DTYPES #####

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


##### BASIS / LINEAR-ALGEBRA HELPERS #####

def create_unitary_dict(**overrides) -> Dict[str, torch.Tensor]:
    """Return {X,Y,Z} single-qubit unitaries as torch.cdouble."""
    inv_sqrt2 = 1.0 / sqrt(2.0)

    X = inv_sqrt2 * torch.tensor(
        [[1.0 + 0.0j, 1.0 + 0.0j],
         [1.0 + 0.0j, -1.0 + 0.0j]],
        dtype=torch.cdouble,
        device=DEVICE,
    )

    Y = inv_sqrt2 * torch.tensor(
        [[1.0 + 0.0j, 0.0 - 1.0j],
         [1.0 + 0.0j, 0.0 + 1.0j]],
        dtype=torch.cdouble,
        device=DEVICE,
    )

    Z = torch.tensor(
        [[1.0 + 0.0j, 0.0 + 0.0j],
         [0.0 + 0.0j, 1.0 + 0.0j]],
        dtype=torch.cdouble,
        device=DEVICE,
    )

    U = {"X": X.contiguous(), "Y": Y.contiguous(), "Z": Z.contiguous()}
    for name, mat in overrides.items():
        U[name] = as_complex_unitary(mat, DEVICE)
    return U


def as_complex_unitary(U, device: torch.device) -> torch.Tensor:
    """Ensure a (2,2) complex matrix on device."""
    if torch.is_tensor(U):
        if U.dim() != 2 or U.shape != (2, 2):
            raise ValueError(f"as_complex_unitary expects (2,2), got {tuple(U.shape)}")
        return U.to(device=device, dtype=torch.cdouble).contiguous()

    U_t = torch.tensor(U, device=device)
    if U_t.dim() != 2 or U_t.shape != (2, 2):
        raise ValueError(f"as_complex_unitary expects (2,2), got {tuple(U_t.shape)}")
    return U_t.to(dtype=torch.cdouble, device=device).contiguous()


def kron_apply(matrices: Sequence[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Apply a Kronecker product of local matrices to x without building the full matrix."""
    assert all(torch.is_complex(m) for m in matrices), "unitaries must be complex"
    if not torch.is_complex(x):
        raise TypeError("x must be complex (cdouble)")

    x_cd = x.to(torch.cdouble)
    leading_dim = x_cd.shape[0]
    batch = int(x_cd.numel() // leading_dim)
    y = x_cd.reshape(leading_dim, batch)

    dims = [m.size(-1) for m in matrices]
    if prod(dims) != leading_dim:
        raise ValueError(f"Incompatible sizes: expected leading dim {prod(dims)}, got {leading_dim}")

    left = leading_dim
    for U in reversed(matrices):
        local_dim = U.shape[-1]
        left //= local_dim
        y = y.reshape(left, local_dim, -1)
        y = torch.einsum("ij,ljm->lim", U, y).reshape(left * local_dim, -1)

    return y.reshape(*x_cd.shape)


def rotate_wavefunction(
    model,
    basis: Iterable[str],
    space: torch.Tensor,
    *,
    unitaries: Optional[dict] = None,
    psi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Rotate psi into a product basis specified by a tuple of 'X','Y','Z'."""
    basis = list(basis)
    if len(basis) != model.num_visible:
        raise ValueError(f"rotate_wavefunction: basis length {len(basis)} != num_visible {model.num_visible}")

    if unitaries is None:
        local_ops = [model.U[b].to(device=model.device, dtype=torch.cdouble) for b in basis]
    else:
        Udict = {k: as_complex_unitary(v, model.device) for k, v in unitaries.items()}
        local_ops = [Udict[b] for b in basis]

    if psi is None:
        x = model.psi_complex(space)
    else:
        if not torch.is_complex(psi):
            raise TypeError("rotate_wavefunction: psi must be complex (cdouble)")
        x = psi.to(device=model.device, dtype=torch.cdouble)

    return kron_apply(local_ops, x)


def enumerate_rotated_branches(model, basis, states, unitaries=None):
    """
    Enumerate coherent computational-basis branches contributing to rotated-basis outcomes.

    Returns:
        Ut : (C, B) complex
        v  : (C, B, n) real bitstrings
    """
    device = model.device
    basis_seq = list(basis)

    if len(basis_seq) != model.num_visible:
        raise ValueError(
            f"enumerate_rotated_branches: basis length {len(basis_seq)} != num_visible {model.num_visible}"
        )
    if states.shape[-1] != model.num_visible:
        raise ValueError(
            f"enumerate_rotated_branches: states width {states.shape[-1]} != num_visible {model.num_visible}"
        )

    rotated_sites = [i for i, b in enumerate(basis_seq) if b != "Z"]
    if len(rotated_sites) == 0:
        v = states.unsqueeze(0)
        Ut = torch.ones(v.shape[:-1], dtype=torch.cdouble, device=device)
        return Ut, v

    src = model.U if unitaries is None else unitaries
    local_unitaries = [as_complex_unitary(src[basis_seq[i]], device).reshape(2, 2).contiguous() for i in rotated_sites]
    Uc = torch.stack(local_unitaries, dim=0)

    num_rotated = len(rotated_sites)
    batch_size = states.shape[0]
    num_branches = 2 ** num_rotated

    combos = model.generate_hilbert_space(size=num_rotated, device=device)

    v = states.unsqueeze(0).repeat(num_branches, 1, 1)
    v[:, :, rotated_sites] = combos.unsqueeze(1)
    v = v.contiguous()

    inp_sb = states[:, rotated_sites].round().long().T
    outp_csb = v[:, :, rotated_sites].round().long().permute(0, 2, 1)
    inp_csb = inp_sb.unsqueeze(0).expand(num_branches, -1, -1)

    s_idx = torch.arange(num_rotated, device=device).view(1, num_rotated, 1).expand(
        num_branches, num_rotated, batch_size
    )
    sel = Uc[s_idx, inp_csb, outp_csb]
    Ut = sel.prod(dim=1)

    return Ut.to(torch.cdouble), v


##### DATA #####

class TomographyDataset:
    """
    Old-style flattened tomography dataset.

    Files:
        train_path       : measurement outcomes, shape (N, n)
        train_bases_path : per-sample basis labels, shape (N, n)
        psi_path         : target wavefunction, two columns (re, im)
        bases_path       : unique basis rows used for evaluation
    """

    def __init__(
        self,
        train_path: str | Path,
        psi_path: str | Path,
        train_bases_path: str | Path,
        bases_path: str | Path,
        *,
        device: torch.device = DEVICE,
    ):
        self.device = device

        train_samples_np = np.loadtxt(train_path, dtype="float32")
        psi_np = np.loadtxt(psi_path, dtype="float64")
        train_bases_np = np.loadtxt(train_bases_path, dtype=str)
        eval_bases_np = np.loadtxt(bases_path, dtype=str, ndmin=1)

        self.train_samples = torch.tensor(train_samples_np, dtype=DTYPE, device=device)
        self.target_state = torch.tensor(
            psi_np[:, 0] + 1j * psi_np[:, 1],
            dtype=torch.cdouble,
            device=device,
        )

        self.train_bases = np.asarray(train_bases_np)
        self.eval_basis_rows = np.asarray(eval_bases_np)

        if self.train_samples.shape[0] != len(self.train_bases):
            raise ValueError("TomographyDataset: sample count != basis row count")

        widths = {len(row) for row in self.train_bases}
        if len(widths) != 1:
            raise ValueError("TomographyDataset: inconsistent basis widths")

        self.num_qubits = next(iter(widths))
        if self.num_qubits != self.train_samples.shape[1]:
            raise ValueError("TomographyDataset: basis width != sample width")

        self._train_bases_tuples = [tuple(row) for row in np.asarray(self.train_bases, dtype=object)]
        self._eval_bases_tuples = [tuple(row) for row in np.asarray(self.eval_basis_rows, dtype=object)]

        z_mask_np = np.array([all(ch == "Z" for ch in row) for row in self._train_bases_tuples], dtype=bool)
        self._z_mask = torch.as_tensor(z_mask_np, dtype=torch.bool)
        self._z_indices = self._z_mask.nonzero(as_tuple=False).view(-1)
        if self._z_indices.numel() == 0:
            raise ValueError("TomographyDataset: no Z-only rows for negative sampling")

        counts_by_basis: Dict[Tuple[str, ...], int] = {}
        for row in self._train_bases_tuples:
            counts_by_basis[row] = counts_by_basis.get(row, 0) + 1
        self.counts_by_basis = counts_by_basis
        self.equal_shot_counts = len(set(counts_by_basis.values())) == 1

    def __len__(self) -> int:
        return int(self.train_samples.shape[0])

    def z_indices(self) -> torch.Tensor:
        return self._z_indices.clone()

    def train_bases_as_tuples(self) -> List[Tuple[str, ...]]:
        return list(self._train_bases_tuples)

    def eval_bases(self) -> List[Tuple[str, ...]]:
        return list(self._eval_bases_tuples)

    def target(self) -> torch.Tensor:
        return self.target_state


class RBMTomographyLoader:
    """
    Pooled positive-minibatch loader with separate Z-only negative minibatches.

    Working reference semantics:
        - positive minibatches come from one pooled random permutation
        - negative minibatches are sampled with replacement from the Z-only pool
        - basis rows stay aligned with positive samples
    """

    def __init__(
        self,
        dataset: TomographyDataset,
        batch_size: int = 140,
        neg_batch_size: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        rng: Optional[torch.Generator] = None,
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE,
        strict: bool = True,
    ):
        self.ds = dataset
        self.bs = int(batch_size)
        self.neg_bs = int(neg_batch_size or batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.rng = rng
        self.device = device
        self.dtype = dtype
        self.strict = strict

        if self.bs <= 0:
            raise ValueError("RBMTomographyLoader: batch_size must be positive")
        if self.neg_bs <= 0:
            raise ValueError("RBMTomographyLoader: neg_batch_size must be positive")

        self._bases_list = self.ds.train_bases_as_tuples()
        self._z_pool = self.ds.z_indices()

        if any(len(row) != self.ds.num_qubits for row in self._bases_list):
            raise ValueError("RBMTomographyLoader: inconsistent basis widths in dataset")
        if self._z_pool.numel() == 0:
            raise ValueError("RBMTomographyLoader: Z-only pool is empty (need negatives)")
        if not getattr(self.ds, "equal_shot_counts", False):
            raise ValueError(
                "RBMTomographyLoader: pooled-batch objective equivalence assumes equal shot counts per basis."
            )

        num_samples = len(self.ds)
        self.slice_bounds = [
            (i, i + self.bs)
            for i in range(0, num_samples, self.bs)
            if (not self.drop_last) or ((i + self.bs) <= num_samples)
        ]

    def __len__(self) -> int:
        return len(self.slice_bounds)

    def __iter__(self):
        return self

    def _randperm(self, n: int) -> torch.Tensor:
        if not self.shuffle:
            return torch.arange(n)
        if self.rng is None:
            return torch.randperm(n)
        return torch.randperm(n, generator=self.rng)

    def _randint(self, high: int, size: Tuple[int, ...]) -> torch.Tensor:
        if self.rng is None:
            return torch.randint(high, size=size)
        return torch.randint(high, size=size, generator=self.rng)

    def iter_epoch(self):
        num_samples = len(self.ds)
        num_batches = len(self)

        perm = self._randperm(num_samples)
        pos_samples_all = self.ds.train_samples[perm].to(self.device, dtype=self.dtype)

        perm_idx = perm.detach().cpu().tolist()
        pos_bases_all = [self._bases_list[i] for i in perm_idx]

        pool_len = self._z_pool.numel()
        neg_choices = self._randint(pool_len, size=(num_batches * self.neg_bs,))
        neg_rows = self._z_pool[neg_choices]
        neg_samples_all = self.ds.train_samples[neg_rows].to(self.device, dtype=self.dtype)

        for batch_idx, (pos_start, pos_end) in enumerate(self.slice_bounds):
            pos_batch = pos_samples_all[pos_start:pos_end]
            bases_batch = pos_bases_all[pos_start:pos_end]

            neg_start = batch_idx * self.neg_bs
            neg_end = neg_start + self.neg_bs
            neg_batch = neg_samples_all[neg_start:neg_end]

            if self.strict:
                if len(bases_batch) != pos_batch.shape[0]:
                    raise RuntimeError("Loader invariant broken: bases_batch length mismatch")
                if pos_batch.shape[1] != self.ds.num_qubits:
                    raise RuntimeError("Loader invariant broken: pos_batch width != num_qubits")
                if neg_batch.shape[1] != self.ds.num_qubits:
                    raise RuntimeError("Loader invariant broken: neg_batch width != num_qubits")

            yield pos_batch, neg_batch, bases_batch


##### MODEL #####

class RBM(nn.Module):
    """Bernoulli/Bernoulli RBM with free energy F(v)."""

    def __init__(
        self,
        num_v: int,
        num_h: Optional[int] = None,
        *,
        zero_weights: bool = False,
        device: torch.device = DEVICE,
    ):
        super().__init__()
        self.num_v = int(num_v)
        self.num_h = int(num_h) if num_h else self.num_v
        self.device = device
        self.initialize_weights(zero_weights=zero_weights)

    def __repr__(self):
        return f"RBM(num_v={self.num_v}, num_h={self.num_h}, device='{self.device}')"

    def initialize_weights(self, *, zero_weights: bool = False, std: Optional[float] = None):
        gen_tensor = torch.zeros if zero_weights else torch.randn
        scale = (1.0 / np.sqrt(self.num_v)) if std is None else float(std)

        self.W = nn.Parameter(
            gen_tensor(self.num_h, self.num_v, device=self.device, dtype=DTYPE) * scale,
            requires_grad=True,
        )
        self.b = nn.Parameter(
            torch.zeros(self.num_v, device=self.device, dtype=DTYPE),
            requires_grad=True,
        )
        self.c = nn.Parameter(
            torch.zeros(self.num_h, device=self.device, dtype=DTYPE),
            requires_grad=True,
        )

    def effective_energy(self, v: torch.Tensor) -> torch.Tensor:
        unsq = False
        if v.dim() < 2:
            v = v.unsqueeze(0)
            unsq = True

        v = v.to(self.W)
        visible_bias_term = torch.matmul(v, self.b)
        hidden_bias_term = F.softplus(F.linear(v, self.W, self.c)).sum(-1)
        out = -(visible_bias_term + hidden_bias_term)
        return out.squeeze(0) if unsq else out

    @torch.no_grad()
    def gibbs_steps(self, k: int, initial_state: torch.Tensor, *, overwrite: bool = False) -> torch.Tensor:
        v = (initial_state if overwrite else initial_state.clone()).to(self.W)
        h = torch.empty(*v.shape[:-1], self.num_h, device=self.device, dtype=DTYPE)

        for _ in range(k):
            h_lin = F.linear(v, self.W, self.c)
            h_prob = torch.sigmoid(h_lin)
            h_prob = torch.nan_to_num(h_prob, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            torch.bernoulli(h_prob, out=h)

            v_lin = F.linear(h, self.W.t(), self.b)
            v_prob = torch.sigmoid(v_lin)
            v_prob = torch.nan_to_num(v_prob, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            torch.bernoulli(v_prob, out=v)

        return v


class ComplexRBM(nn.Module):
    """
    Complex RBM:
        psi(s) = exp(-F_lambda(s)/2) * exp(-i F_mu(s)/2)

    The energy-model core sits in two RBM instances.
    Wavefunction-related methods live here because tomography needs them directly.
    """

    def __init__(
        self,
        num_v: int,
        num_h: Optional[int] = None,
        *,
        unitary_dict: Optional[Dict[str, torch.Tensor]] = None,
        k: int = 10,
        device: torch.device = DEVICE,
    ):
        super().__init__()
        self.device = device
        self.k = int(k)

        self.rbm_am = RBM(num_v, num_h, device=self.device)
        self.rbm_ph = RBM(num_v, num_h, device=self.device)

        self.num_v = self.rbm_am.num_v
        self.num_h = self.rbm_am.num_h
        self.num_visible = self.num_v

        raw = unitary_dict if unitary_dict is not None else create_unitary_dict()
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

    @property
    def max_size(self):
        return self._max_size

    def initialize_weights(self, *, std: Optional[float] = None):
        self.rbm_am.initialize_weights(std=std)
        self.rbm_ph.initialize_weights(std=std)

    def amplitude(self, v: torch.Tensor) -> torch.Tensor:
        v = v.to(self.device, dtype=DTYPE)
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()

    def phase(self, v: torch.Tensor) -> torch.Tensor:
        v = v.to(self.device, dtype=DTYPE)
        return -0.5 * self.rbm_ph.effective_energy(v)

    def psi_complex(self, v: torch.Tensor) -> torch.Tensor:
        v = v.to(self.device, dtype=DTYPE)
        amp = (-self.rbm_am.effective_energy(v)).exp().sqrt()
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        return amp.to(torch.cdouble) * torch.exp(1j * ph.to(torch.cdouble))

    def psi_complex_normalized(self, v: torch.Tensor) -> torch.Tensor:
        v = v.to(self.device, dtype=DTYPE)
        E = self.rbm_am.effective_energy(v)
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        logZ = torch.logsumexp(-E, dim=0)
        return torch.exp(((-0.5 * E) - 0.5 * logZ).to(torch.cdouble) + 1j * ph.to(torch.cdouble))

    def generate_hilbert_space(self, size: Optional[int] = None, device: Optional[torch.device] = None) -> torch.Tensor:
        device = self.device if device is None else device
        size = self.num_visible if size is None else int(size)
        if size > self._max_size:
            raise ValueError(f"Hilbert space too large (n={size} > max={self._max_size}).")
        n = 1 << size
        ar = torch.arange(n, device=device, dtype=torch.long)
        shifts = torch.arange(size - 1, -1, -1, device=device, dtype=torch.long)
        return ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)

    def stable_log_overlap_amp2(
        self,
        basis: Tuple[str, ...],
        states: torch.Tensor,
        *,
        eps_rot: float = 1e-6,
        unitaries=None,
    ) -> torch.Tensor:
        Ut, v = enumerate_rotated_branches(self, basis, states, unitaries=unitaries)
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

    def _positive_phase_loss(
        self,
        samples: torch.Tensor,
        bases_batch: List[Tuple[str, ...]],
        *,
        eps_rot: float = 1e-6,
    ):
        buckets: Dict[Tuple[str, ...], List[int]] = {}
        for i, row in enumerate(bases_batch):
            buckets.setdefault(tuple(row), []).append(i)

        loss_rot = samples.new_tensor(0.0, dtype=DTYPE)
        loss_z = samples.new_tensor(0.0, dtype=DTYPE)
        cnt_z = 0
        cnt_rot = 0

        for basis_t, idxs in buckets.items():
            idxs_t = torch.tensor(idxs, device=samples.device)
            if any(ch != "Z" for ch in basis_t):
                log_amp2 = self.stable_log_overlap_amp2(basis_t, samples[idxs_t], eps_rot=eps_rot)
                loss_rot = loss_rot - log_amp2.sum().to(DTYPE)
                cnt_rot += len(idxs)
            else:
                Epos = self.rbm_am.effective_energy(samples[idxs_t])
                loss_z = loss_z + Epos.sum()
                cnt_z += len(idxs)

        return loss_rot + loss_z, loss_z, loss_rot, cnt_z, cnt_rot

    def _negative_phase_loss(self, neg_init: torch.Tensor):
        with torch.no_grad():
            vk = self.rbm_am.gibbs_steps(self.k, neg_init, overwrite=True)
        Eneg = self.rbm_am.effective_energy(vk)
        return Eneg.sum(), vk.shape[0]

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor, List[Tuple[str, ...]]], aux_vars: Dict[str, Any]):
        """
        HyperRBM-style forward.

        batch:
            pos_batch, neg_batch, bases_batch

        returns:
            loss, diagnostics
        """
        pos_batch, neg_batch, bases_batch = batch
        pos_batch = pos_batch.to(self.device, dtype=DTYPE)
        neg_batch = neg_batch.to(self.device, dtype=DTYPE)

        L_pos, L_z_only, L_rot_only, cnt_z, cnt_rot = self._positive_phase_loss(pos_batch, bases_batch)
        B_pos = float(pos_batch.shape[0])

        L_neg, B_neg = self._negative_phase_loss(neg_batch)

        pos_term = L_pos / B_pos
        neg_term = L_neg / B_neg
        loss = pos_term - neg_term

        diagnostics = {
            "pos_term": float(pos_term.detach().item()),
            "neg_term": float(neg_term.detach().item()),
            "zpos_term": float((L_z_only / cnt_z).detach().item()) if cnt_z > 0 else float("nan"),
            "rotpos_term": float((L_rot_only / cnt_rot).detach().item()) if cnt_rot > 0 else float("nan"),
        }
        return loss, diagnostics


##### METRICS #####

@torch.no_grad()
def fidelity(model: ComplexRBM, target: torch.Tensor, basis_states: Optional[torch.Tensor] = None) -> float:
    basis_states = model.generate_hilbert_space() if basis_states is None else basis_states

    psi = model.psi_complex_normalized(basis_states).reshape(-1).contiguous()
    tgt = target.to(device=model.device, dtype=torch.cdouble).reshape(-1).contiguous()

    npsi = torch.linalg.vector_norm(psi)
    nt = torch.linalg.vector_norm(tgt)
    if npsi == 0 or nt == 0:
        return 0.0

    psi_n = psi / npsi
    tgt_n = tgt / nt
    inner = (tgt_n.conj() * psi_n).sum()
    return float(inner.abs().pow(2).real)


@torch.no_grad()
def calculate_average_basis_kl(
    model: ComplexRBM,
    target: torch.Tensor,
    eval_bases: List[Tuple[str, ...]],
    basis_states: torch.Tensor,
) -> float:
    tgt = target.to(device=model.device, dtype=torch.cdouble).reshape(-1)
    nt = torch.linalg.vector_norm(tgt)
    if nt == 0:
        return 0.0
    tgt_norm = tgt / nt

    psi_norm = model.psi_complex_normalized(basis_states).reshape(-1)

    eps = 1e-12
    KL_val = 0.0
    for basis in eval_bases:
        tgt_r = rotate_wavefunction(model, basis, basis_states, psi=tgt_norm)
        psi_r = rotate_wavefunction(model, basis, basis_states, psi=psi_norm)

        p = (tgt_r.abs().to(DTYPE) ** 2)
        q = (psi_r.abs().to(DTYPE) ** 2)
        p = (p / p.sum().clamp_min(eps)).clamp_min(eps)
        q = (q / q.sum().clamp_min(eps)).clamp_min(eps)

        KL_val += torch.sum(p * (torch.log(p) - torch.log(q)))

    return float((KL_val / len(eval_bases)).item())


##### TRAINING HELPERS #####

def get_sigmoid_curve(high, low, steps, falloff, center=None):
    """
    Sigmoid LR schedule in the style of the modern project.

    Returns a callable mapping global step -> learning rate.
    """
    import math

    if center is None:
        center = steps / 2.0

    def fn(step: int) -> float:
        s = min(step, steps)
        return float(low + (high - low) / (1.0 + math.exp(falloff * (s - center))))

    return fn


def get_constant_curve(value: float) -> Callable[[int], float]:
    def fn(step: int) -> float:
        return float(value)
    return fn


def _resolve_dual_lr(lr_value: Any) -> Tuple[float, float]:
    """
    Allow flexible schedules:
        float -> same LR for amplitude and phase
        tuple/list -> (lr_am, lr_ph)
        dict -> {'am': ..., 'ph': ...}
    """
    if isinstance(lr_value, dict):
        lr_am = float(lr_value["am"])
        lr_ph = float(lr_value.get("ph", lr_am))
        return lr_am, lr_ph

    if isinstance(lr_value, (tuple, list)):
        if len(lr_value) != 2:
            raise ValueError("LR tuple/list must have length 2")
        return float(lr_value[0]), float(lr_value[1])

    lr = float(lr_value)
    return lr, lr


def train_loop(
    model: ComplexRBM,
    optimizer: torch.optim.Optimizer,
    loader: RBMTomographyLoader,
    *,
    num_epochs: int,
    lr_schedule_fn: Callable[[int], Any],
    log_every: int = 5,
    grad_clip_norm: float = 10.0,
    basis_states: Optional[torch.Tensor] = None,
    target: Optional[torch.Tensor] = None,
    eval_bases: Optional[List[Tuple[str, ...]]] = None,
):
    """
    Project-style training loop.

    Important:
    - keeps the working pooled-CD semantics
    - supports changing optimizer / schedule externally
    - schedule can control both parameter groups independently
    """
    global_step = 0
    history = {"epoch": []}

    if target is not None:
        history["Fidelity"] = []
        history["KL"] = []
        history["GradNormAM"] = []
        history["GradNormPH"] = []
        history["Pos"] = []
        history["Neg"] = []
        history["ZPos"] = []
        history["RotPos"] = []
        history["LR_AM"] = []
        history["LR_PH"] = []

    model.train()

    all_params = list(model.parameters())

    for epoch in range(num_epochs):
        grad_am_epoch = []
        grad_ph_epoch = []
        pos_epoch = []
        neg_epoch = []
        zpos_epoch = []
        rotpos_epoch = []

        for batch in loader.iter_epoch():
            lr_am, lr_ph = _resolve_dual_lr(lr_schedule_fn(global_step))
            optimizer.param_groups[0]["lr"] = lr_am
            optimizer.param_groups[1]["lr"] = lr_ph

            optimizer.zero_grad(set_to_none=True)
            loss, aux = model(batch, {})
            loss.backward()

            am_sq = 0.0
            for p in model.rbm_am.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    am_sq += float(torch.sum(g * g).item())

            ph_sq = 0.0
            for p in model.rbm_ph.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    ph_sq += float(torch.sum(g * g).item())

            grad_am_epoch.append(am_sq ** 0.5)
            grad_ph_epoch.append(ph_sq ** 0.5)

            pos_epoch.append(aux["pos_term"])
            neg_epoch.append(aux["neg_term"])
            if not np.isnan(aux["zpos_term"]):
                zpos_epoch.append(aux["zpos_term"])
            if not np.isnan(aux["rotpos_term"]):
                rotpos_epoch.append(aux["rotpos_term"])

            torch.nn.utils.clip_grad_norm_(all_params, grad_clip_norm)
            optimizer.step()
            global_step += 1

        if target is not None and ((epoch + 1) % log_every == 0):
            fid = fidelity(model, target, basis_states)
            kl = calculate_average_basis_kl(model, target, eval_bases, basis_states)

            history["epoch"].append(epoch + 1)
            history["Fidelity"].append(fid)
            history["KL"].append(kl)
            history["GradNormAM"].append(float(np.mean(grad_am_epoch)) if grad_am_epoch else float("nan"))
            history["GradNormPH"].append(float(np.mean(grad_ph_epoch)) if grad_ph_epoch else float("nan"))
            history["Pos"].append(float(np.mean(pos_epoch)) if pos_epoch else float("nan"))
            history["Neg"].append(float(np.mean(neg_epoch)) if neg_epoch else float("nan"))
            history["ZPos"].append(float(np.mean(zpos_epoch)) if zpos_epoch else float("nan"))
            history["RotPos"].append(float(np.mean(rotpos_epoch)) if rotpos_epoch else float("nan"))
            history["LR_AM"].append(float(lr_am))
            history["LR_PH"].append(float(lr_ph))

            print(
                f"Epoch {epoch + 1}: "
                f"Fidelity = {fid:.6f} | "
                f"KL = {kl:.6f} | "
                f"g_am = {history['GradNormAM'][-1]:.6f} | "
                f"g_ph = {history['GradNormPH'][-1]:.6f} | "
                f"pos = {history['Pos'][-1]:.6f} | "
                f"neg = {history['Neg'][-1]:.6f} | "
                f"zpos = {history['ZPos'][-1]:.6f} | "
                f"rotpos = {history['RotPos'][-1]:.6f} | "
                f"lr_am = {history['LR_AM'][-1]:.6f} | "
                f"lr_ph = {history['LR_PH'][-1]:.6f}"
            )

    return model, history


##### EXPERIMENT HELPERS #####

def train_experiment_model(
    train_path: str | Path,
    train_bases_path: str | Path,
    psi_path: str | Path,
    bases_path: str | Path,
    config: Dict[str, Any],
    rng: Optional[torch.Generator],
    device: torch.device,
):
    """
    HyperRBM-style factory/training entrypoint.

    This is the function you tweak when you want to swap:
    - optimizer
    - batch size
    - schedule
    - number of hidden units
    - CD steps
    """
    dataset = TomographyDataset(
        train_path=train_path,
        psi_path=psi_path,
        train_bases_path=train_bases_path,
        bases_path=bases_path,
        device=device,
    )

    loader = RBMTomographyLoader(
        dataset,
        batch_size=config["batch_size"],
        neg_batch_size=config.get("neg_batch_size", config["batch_size"]),
        shuffle=config.get("shuffle", True),
        drop_last=config.get("drop_last", False),
        rng=rng,
        device=device,
        dtype=DTYPE,
    )

    model = ComplexRBM(
        num_v=dataset.num_qubits,
        num_h=config["num_hidden"],
        unitary_dict=create_unitary_dict(),
        k=config["k_steps"],
        device=device,
    ).to(device)

    model.initialize_weights(std=config.get("init_std", None))

    optimizer_cls = config.get("optimizer_cls", torch.optim.SGD)
    optimizer_kwargs = dict(config.get("optimizer_kwargs", {}))
    init_lr = float(config["init_lr"])

    optimizer = optimizer_cls(
        [
            {"params": list(model.rbm_am.parameters()), "lr": init_lr},
            {"params": list(model.rbm_ph.parameters()), "lr": init_lr},
        ],
        **optimizer_kwargs,
    )

    steps = config["epochs"] * len(loader)
    if config.get("lr_schedule_fn") is not None:
        scheduler = config["lr_schedule_fn"]
    else:
        scheduler = get_constant_curve(init_lr)

    basis_states = model.generate_hilbert_space()

    model, history = train_loop(
        model,
        optimizer,
        loader,
        num_epochs=config["epochs"],
        lr_schedule_fn=scheduler,
        log_every=config.get("log_every", 5),
        grad_clip_norm=config.get("grad_clip_norm", 10.0),
        basis_states=basis_states,
        target=dataset.target(),
        eval_bases=dataset.eval_bases(),
    )

    return model, history, dataset, basis_states


def plot_phase_comparison(model: ComplexRBM, target_state: torch.Tensor, basis_states: torch.Tensor):
    with torch.no_grad():
        psi_m = model.psi_complex_normalized(basis_states).reshape(-1).to(torch.cdouble)

    psi_t = target_state.reshape(-1).to(torch.cdouble)
    psi_m = psi_m / torch.linalg.vector_norm(psi_m)
    psi_t = psi_t / torch.linalg.vector_norm(psi_t)

    phi_m = torch.angle(psi_m)
    phi_t = torch.angle(psi_t)

    mass_cut = 0.99
    k_cap = 512
    probs = psi_t.abs().pow(2)
    order = torch.argsort(probs, descending=True)
    cum = torch.cumsum(probs[order], dim=0)
    idx = torch.searchsorted(cum, torch.tensor(mass_cut, device=cum.device)).item()
    k_sel = min(idx + 1, k_cap, probs.numel())
    sel = order[:k_sel]

    phi_m_sel = phi_m[sel]
    phi_t_sel = phi_t[sel]
    phi_diff_sel = torch.remainder(phi_m_sel - phi_t_sel + torch.pi, 2 * torch.pi) - torch.pi
    phi_m_sel_shift = phi_m_sel - 0.5 * (phi_diff_sel.min() + phi_diff_sel.max())
    phi_diff_sel = torch.remainder(phi_m_sel_shift - phi_t_sel + torch.pi, 2 * torch.pi) - torch.pi

    fig_p, axp = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    axp.plot(range(sel.numel()), phi_t_sel.cpu().numpy(), marker=".", linestyle="", label="target phase")
    axp.plot(range(sel.numel()), phi_m_sel_shift.cpu().numpy(), marker="x", linestyle="", label="model phase (shifted)")
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
    axe.set_title("Phase error after global shift")
    axe.grid(True, alpha=0.3)
    axe.legend()
    fig_e.tight_layout()


def plot_training_curves(history: Dict[str, List[float]]):
    epochs = history.get("epoch", [])
    if not epochs:
        return

    fig, ax1 = plt.subplots(figsize=(6.0, 4.0), dpi=140)
    ax2 = ax1.twinx()

    ax1.plot(epochs, history["Fidelity"], marker="o", label="Fidelity")
    ax2.plot(epochs, history["KL"], marker="s", linestyle="--", label="KL")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(r"$|\langle \psi_t \mid \psi \rangle|^2$")
    ax2.set_ylabel(r"KL$(p\,\|\,q)$")
    ax1.set_title("RBM Tomography - pooled autodiff CD")
    ax1.grid(True, alpha=0.3)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")
    fig.tight_layout()


##### MAIN #####

if __name__ == "__main__":
    TRAIN_PATH = "w_phase_meas_values.txt"
    TRAIN_BASES_PATH = "w_phase_meas_bases.txt"
    PSI_PATH = "w_phase_state.txt"
    BASES_PATH = "w_phase_unique_bases.txt"

    SEED = 1234
    torch.manual_seed(SEED)

    # Use rng=None to preserve the original global-torch-RNG loader behaviour.
    # Passing a dedicated Generator will change the stochastic trajectory.
    rng = None

    TRAIN_CONFIG = {
        # Keep the tomography-side training semantics
        "epochs": 95,
        "batch_size": 256,
        "neg_batch_size": 256,
        "num_hidden": 4,
        "k_steps": 10,
        "init_std": None,
        "grad_clip_norm": 10.0,
        "log_every": 5,
        "shuffle": True,
        "drop_last": False,

        # More modern optimizer setup
        "optimizer_cls": torch.optim.Adam,
        "optimizer_kwargs": {},

        # Adam + schedule
        "init_lr": 1e-2,
        "final_lr": 1e-4,
        "schedule_falloff": 0.005,
        "lr_schedule_fn": None,
    }

    steps = TRAIN_CONFIG["epochs"] * (
        (len(TomographyDataset(TRAIN_PATH, PSI_PATH, TRAIN_BASES_PATH, BASES_PATH, device=DEVICE))
         + TRAIN_CONFIG["batch_size"] - 1) // TRAIN_CONFIG["batch_size"]
    )
    TRAIN_CONFIG["lr_schedule_fn"] = get_sigmoid_curve(
        TRAIN_CONFIG["init_lr"],
        TRAIN_CONFIG["final_lr"],
        steps,
        TRAIN_CONFIG["schedule_falloff"],
    )

    model, history, dataset, basis_states = train_experiment_model(
        TRAIN_PATH,
        TRAIN_BASES_PATH,
        PSI_PATH,
        BASES_PATH,
        TRAIN_CONFIG,
        rng,
        DEVICE,
    )

    plot_phase_comparison(model, dataset.target(), basis_states)
    plot_training_curves(history)
    plt.show()
