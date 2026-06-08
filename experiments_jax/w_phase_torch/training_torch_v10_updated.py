#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlined reference implementation for pooled-CD complex RBM tomography.

Design goal:
- preserve the working training logic
- organize the code so it visibly shares the same "shape" as more modern projects:
    * data container
    * training loader
    * model
    * metrics
    * training loop
    * experiment config / main entrypoint

This file intentionally keeps the original working semantics:
- pooled positive minibatches
- Z-only negative minibatches
- constant learning rates
- exact small-system diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt, prod
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


##### DEVICE AND DTYPES #####

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


##### CONFIG #####

@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 95
    pos_batch_size: int = 140
    neg_batch_size: int = 140
    learning_rate: float = 1e-1
    cd_k: int = 10
    log_every: int = 5
    grad_clip_norm: float = 10.0
    seed: int = 1234


@dataclass(frozen=True)
class ExperimentPaths:
    train_values: str = "w_phase_meas_values.txt"
    train_bases: str = "w_phase_meas_bases.txt"
    state_vector: str = "w_phase_state.txt"
    eval_bases: str = "w_phase_unique_bases.txt"


##### LOW-LEVEL LINEAR ALGEBRA / BASIS UTILITIES #####

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
    """Ensure a (2, 2) complex matrix on device."""
    if torch.is_tensor(U):
        if U.dim() != 2 or U.shape != (2, 2):
            raise ValueError(f"as_complex_unitary expects (2,2), got {tuple(U.shape)}")
        return U.to(device=device, dtype=torch.cdouble).contiguous()

    U_t = torch.tensor(U, device=device)
    if U_t.dim() != 2 or U_t.shape != (2, 2):
        raise ValueError(f"as_complex_unitary expects (2,2), got {tuple(U_t.shape)}")
    return U_t.to(dtype=torch.cdouble, device=device).contiguous()


def safe_inverse(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Safe complex inverse: conj(z) / max(|z|^2, eps)."""
    zz = z.to(torch.cdouble)
    return zz.conj() / (zz.abs().pow(2).clamp_min(eps))


def kron_apply(matrices: Sequence[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """
    Apply a Kronecker product of local matrices to x without materializing the full matrix.
    """
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
    nn_state,
    basis: Iterable[str],
    space: torch.Tensor,
    *,
    unitaries: Optional[dict] = None,
    psi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Rotate psi into a product basis specified by a tuple of 'X','Y','Z'."""
    basis = list(basis)
    if len(basis) != nn_state.num_visible:
        raise ValueError(f"rotate_wavefunction: basis length {len(basis)} != num_visible {nn_state.num_visible}")

    if unitaries is None:
        local_ops = [nn_state.U[b].to(device=nn_state.device, dtype=torch.cdouble) for b in basis]
    else:
        Udict = {k: as_complex_unitary(v, nn_state.device) for k, v in unitaries.items()}
        local_ops = [Udict[b] for b in basis]

    if psi is None:
        x = nn_state.psi_complex(space)
    else:
        if not torch.is_complex(psi):
            raise TypeError("rotate_wavefunction: psi must be complex (cdouble)")
        x = psi.to(device=nn_state.device, dtype=torch.cdouble)

    return kron_apply(local_ops, x)


def enumerate_rotated_branches(nn_state, basis, states, unitaries=None):
    """
    Enumerate coherent computational-basis branches contributing to rotated-basis outcomes.

    Returns:
        Ut : (C, B) complex
        v  : (C, B, n) real bitstrings
    """
    device = nn_state.device
    basis_seq = list(basis)

    if len(basis_seq) != nn_state.num_visible:
        raise ValueError(
            f"enumerate_rotated_branches: basis length {len(basis_seq)} != num_visible {nn_state.num_visible}"
        )
    if states.shape[-1] != nn_state.num_visible:
        raise ValueError(
            f"enumerate_rotated_branches: states width {states.shape[-1]} != num_visible {nn_state.num_visible}"
        )

    rotated_sites = [i for i, b in enumerate(basis_seq) if b != "Z"]
    if len(rotated_sites) == 0:
        v = states.unsqueeze(0)
        Ut = torch.ones(v.shape[:-1], dtype=torch.cdouble, device=device)
        return Ut, v

    src = nn_state.U if unitaries is None else unitaries
    local_unitaries = [as_complex_unitary(src[basis_seq[i]], device).reshape(2, 2).contiguous() for i in rotated_sites]
    Uc = torch.stack(local_unitaries, dim=0)

    num_rotated = len(rotated_sites)
    batch_size = states.shape[0]
    num_branches = 2 ** num_rotated

    combos = nn_state.generate_hilbert_space(size=num_rotated, device=device)

    v = states.unsqueeze(0).repeat(num_branches, 1, 1)
    v[:, :, rotated_sites] = combos.unsqueeze(1)
    v = v.contiguous()

    inp_sb = states[:, rotated_sites].round().long().T
    outp_csb = v[:, :, rotated_sites].round().long().permute(0, 2, 1)
    inp_csb = inp_sb.unsqueeze(0).expand(num_branches, -1, -1)

    s_idx = torch.arange(num_rotated, device=device).view(1, num_rotated, 1).expand(num_branches, num_rotated, batch_size)
    sel = Uc[s_idx, inp_csb, outp_csb]
    Ut = sel.prod(dim=1)

    return Ut.to(torch.cdouble), v


def basis_rows_to_indices(states: torch.Tensor) -> torch.Tensor:
    """Convert bit rows to flat computational-basis indices."""
    s = states.round().to(torch.long)
    n = s.shape[-1]
    shifts = torch.arange(n - 1, -1, -1, device=s.device, dtype=torch.long)
    return (s << shifts).sum(dim=-1)


def rotated_inner_product(nn_state, basis, states, unitaries=None, psi=None, include_extras=False):
    """Compute overlap for measured outcomes in the given basis."""
    Ut, v = enumerate_rotated_branches(nn_state, basis, states, unitaries=unitaries)

    if psi is None:
        psi_sel = nn_state.psi_complex(v)
    else:
        if not torch.is_complex(psi):
            raise TypeError("rotated_inner_product: psi must be complex.")
        idx = basis_rows_to_indices(v).long()
        psi_c = psi.to(dtype=torch.cdouble, device=nn_state.device)
        psi_sel = psi_c[idx]

    Upsi_v_c = Ut * psi_sel
    Upsi_c = Upsi_v_c.sum(dim=0)

    if include_extras:
        return Upsi_c, Upsi_v_c, v
    return Upsi_c


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

        self._num_visible = next(iter(widths))
        if self._num_visible != self.train_samples.shape[1]:
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

    def num_visible(self) -> int:
        return int(self._num_visible)

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

    This matches the working reference semantics:
        - positive minibatches come from one pooled random permutation
        - negative minibatches are sampled with replacement from the Z-only pool
        - basis rows stay aligned with positive samples
    """

    def __init__(
        self,
        dataset: TomographyDataset,
        *,
        pos_batch_size: int = 140,
        neg_batch_size: Optional[int] = None,
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE,
        strict: bool = True,
    ):
        self.ds = dataset
        self.pos_bs = int(pos_batch_size)
        self.neg_bs = int(neg_batch_size or pos_batch_size)
        self.device = device
        self.dtype = dtype
        self.strict = strict
        self._gen: Optional[torch.Generator] = None

        if self.pos_bs <= 0:
            raise ValueError("RBMTomographyLoader: pos_batch_size must be positive")
        if self.neg_bs <= 0:
            raise ValueError("RBMTomographyLoader: neg_batch_size must be positive")

        self._bases_list = self.ds.train_bases_as_tuples()
        self._z_pool = self.ds.z_indices()
        n = self.ds.num_visible()

        if any(len(row) != n for row in self._bases_list):
            raise ValueError("RBMTomographyLoader: inconsistent basis widths in dataset")
        if self._z_pool.numel() == 0:
            raise ValueError("RBMTomographyLoader: Z-only pool is empty (need negatives)")
        if not getattr(self.ds, "equal_shot_counts", False):
            raise ValueError(
                "RBMTomographyLoader: pooled-batch objective equivalence assumes equal shot counts per basis."
            )

    def set_seed(self, seed: Optional[int]):
        """Optional dedicated CPU generator. Leave unset to use global torch RNG."""
        if seed is None:
            self._gen = None
        else:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            self._gen = g

    def __len__(self):
        return ceil(len(self.ds) / self.pos_bs)

    def __iter__(self):
        return self.iter_epoch()

    def iter_epoch(self):
        num_samples = len(self.ds)
        num_batches = ceil(num_samples / self.pos_bs)

        perm = torch.randperm(num_samples, generator=self._gen) if self._gen is not None else torch.randperm(num_samples)
        pos_samples_all = self.ds.train_samples[perm].to(self.device, dtype=self.dtype)

        perm_idx = perm.detach().cpu().tolist()
        pos_bases_all = [self._bases_list[i] for i in perm_idx]

        pool_len = self._z_pool.numel()
        if self._gen is None:
            neg_choices = torch.randint(pool_len, size=(num_batches * self.neg_bs,))
        else:
            neg_choices = torch.randint(pool_len, size=(num_batches * self.neg_bs,), generator=self._gen)
        neg_rows = self._z_pool[neg_choices]
        neg_samples_all = self.ds.train_samples[neg_rows].to(self.device, dtype=self.dtype)

        for batch_idx in range(num_batches):
            pos_start = batch_idx * self.pos_bs
            pos_end = min(pos_start + self.pos_bs, num_samples)
            pos_batch = pos_samples_all[pos_start:pos_end]
            bases_batch = pos_bases_all[pos_start:pos_end]

            neg_start = batch_idx * self.neg_bs
            neg_end = neg_start + self.neg_bs
            neg_batch = neg_samples_all[neg_start:neg_end]

            if self.strict:
                if len(bases_batch) != pos_batch.shape[0]:
                    raise RuntimeError("Loader invariant broken: bases_batch length mismatch")
                if pos_batch.shape[1] != self.ds.num_visible():
                    raise RuntimeError("Loader invariant broken: pos_batch width != num_visible")
                if neg_batch.shape[1] != self.ds.num_visible():
                    raise RuntimeError("Loader invariant broken: neg_batch width != num_visible")

            yield pos_batch, neg_batch, bases_batch


##### MODEL #####

class RBM(nn.Module):
    """Bernoulli/Bernoulli RBM with free energy F(v)."""

    def __init__(self, num_visible: int, num_hidden: Optional[int] = None, *, zero_weights: bool = False,
                 device: torch.device = DEVICE):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden) if num_hidden else self.num_visible
        self.device = device
        self.initialize_parameters(zero_weights=zero_weights)

    def __repr__(self):
        return f"RBM(num_visible={self.num_visible}, num_hidden={self.num_hidden}, device='{self.device}')"

    def initialize_parameters(self, zero_weights: bool = False):
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

    def effective_energy(self, v: torch.Tensor) -> torch.Tensor:
        unsq = False
        if v.dim() < 2:
            v = v.unsqueeze(0)
            unsq = True

        v = v.to(self.weights)
        visible_bias_term = torch.matmul(v, self.visible_bias)
        hidden_bias_term = F.softplus(F.linear(v, self.weights, self.hidden_bias)).sum(-1)
        out = -(visible_bias_term + hidden_bias_term)
        return out.squeeze(0) if unsq else out

    @torch.no_grad()
    def gibbs_steps(self, k: int, initial_state: torch.Tensor, *, overwrite: bool = False) -> torch.Tensor:
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


class ComplexRBM:
    """
    Complex RBM wavefunction:
        psi(s) = exp(-F_lambda(s)/2) * exp(-i F_mu(s)/2)
    """

    def __init__(
        self,
        num_visible: int,
        num_hidden: Optional[int] = None,
        unitary_dict: Optional[Dict[str, torch.Tensor]] = None,
        *,
        device: torch.device = DEVICE,
    ):
        self.device = device
        self.rbm_am = RBM(num_visible, num_hidden, device=self.device)
        self.rbm_ph = RBM(num_visible, num_hidden, device=self.device)

        self.num_visible = self.rbm_am.num_visible
        self.num_hidden = self.rbm_am.num_hidden

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

    def reinitialize_parameters(self):
        self.rbm_am.initialize_parameters()
        self.rbm_ph.initialize_parameters()

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

    def positive_phase_loss(
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

    def negative_phase_loss(self, k: int, neg_init: torch.Tensor):
        with torch.no_grad():
            vk = self.rbm_am.gibbs_steps(k, neg_init, overwrite=True)
        Eneg = self.rbm_am.effective_energy(vk)
        return Eneg.sum(), vk.shape[0]


##### METRICS #####

@torch.no_grad()
def fidelity(model, target, *, space=None):
    if not torch.is_complex(target):
        raise TypeError("fidelity: target must be complex (cdouble)")

    space = model.generate_hilbert_space() if space is None else space

    psi = model.psi_complex_normalized(space).reshape(-1).contiguous()
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
def average_basis_kl(model, target, *, space=None, bases=None):
    if bases is None:
        raise ValueError("average_basis_kl needs bases")
    if not torch.is_complex(target):
        raise TypeError("average_basis_kl: target must be complex (cdouble)")

    space = model.generate_hilbert_space() if space is None else space

    tgt = target.to(device=model.device, dtype=torch.cdouble).reshape(-1)
    nt = torch.linalg.vector_norm(tgt)
    if nt == 0:
        return 0.0
    tgt_norm = tgt / nt

    psi_norm_cd = model.psi_complex_normalized(space).reshape(-1)

    KL_val = 0.0
    eps = 1e-12

    for basis in bases:
        tgt_psi_r = rotate_wavefunction(model, basis, space, psi=tgt_norm)
        psi_r = rotate_wavefunction(model, basis, space, psi=psi_norm_cd)

        nn_probs_r = (psi_r.abs().to(DTYPE) ** 2)
        tgt_probs_r = (tgt_psi_r.abs().to(DTYPE) ** 2)

        p_sum = tgt_probs_r.sum().clamp_min(eps)
        q_sum = nn_probs_r.sum().clamp_min(eps)
        p = (tgt_probs_r / p_sum).clamp_min(eps)
        q = (nn_probs_r / q_sum).clamp_min(eps)

        KL_val += torch.sum(p * (torch.log(p) - torch.log(q)))

    return (KL_val / len(bases)).item()


@torch.no_grad()
def exact_nll_breakdown(model, samples, bases_batch, *, space, eps_rot=1e-12):
    samples = samples.to(model.device, dtype=DTYPE)
    logZ = torch.logsumexp(-model.rbm_am.effective_energy(space), dim=0)

    buckets: Dict[Tuple[str, ...], List[int]] = {}
    for i, row in enumerate(bases_batch):
        buckets.setdefault(tuple(row), []).append(i)

    total_sum = samples.new_tensor(0.0, dtype=DTYPE)
    z_sum = samples.new_tensor(0.0, dtype=DTYPE)
    rot_sum = samples.new_tensor(0.0, dtype=DTYPE)
    n_total = 0
    n_z = 0
    n_rot = 0

    for basis_t, idxs in buckets.items():
        idxs_t = torch.tensor(idxs, device=samples.device)
        batch = samples[idxs_t]

        if all(ch == "Z" for ch in basis_t):
            Epos = model.rbm_am.effective_energy(batch)
            nll = Epos + logZ
            z_sum += nll.sum()
            total_sum += nll.sum()
            n_z += len(idxs)
            n_total += len(idxs)
        else:
            log_amp2 = model.stable_log_overlap_amp2(basis_t, batch, eps_rot=eps_rot)
            nll = logZ - log_amp2
            rot_sum += nll.sum()
            total_sum += nll.sum()
            n_rot += len(idxs)
            n_total += len(idxs)

    return {
        "exact_nll_total": float((total_sum / max(n_total, 1)).item()),
        "exact_nll_z": float((z_sum / max(n_z, 1)).item()) if n_z > 0 else float("nan"),
        "exact_nll_rot": float((rot_sum / max(n_rot, 1)).item()) if n_rot > 0 else float("nan"),
    }


@torch.no_grad()
def kl_breakdown(model, target, *, space, bases):
    tgt = target.to(device=model.device, dtype=torch.cdouble).reshape(-1)
    tgt = tgt / torch.linalg.vector_norm(tgt)
    psi = model.psi_complex_normalized(space).reshape(-1)

    eps = 1e-12
    z_kl = float("nan")
    rot_kls = []

    for basis in bases:
        tgt_r = rotate_wavefunction(model, basis, space, psi=tgt)
        psi_r = rotate_wavefunction(model, basis, space, psi=psi)

        p = (tgt_r.abs().to(DTYPE) ** 2)
        q = (psi_r.abs().to(DTYPE) ** 2)

        p = (p / p.sum().clamp_min(eps)).clamp_min(eps)
        q = (q / q.sum().clamp_min(eps)).clamp_min(eps)

        kl = float(torch.sum(p * (torch.log(p) - torch.log(q))).item())

        if all(ch == "Z" for ch in basis):
            z_kl = kl
        else:
            rot_kls.append(kl)

    return {
        "z_kl": z_kl,
        "rot_kl_mean": float(np.mean(rot_kls)) if rot_kls else float("nan"),
        "rot_kl_max": float(np.max(rot_kls)) if rot_kls else float("nan"),
    }


@torch.no_grad()
def support_mass_stats(model, target, *, space, eps=1e-12):
    psi = model.psi_complex_normalized(space).reshape(-1)
    probs = psi.abs().pow(2)

    tgt = target.to(device=model.device, dtype=torch.cdouble).reshape(-1)
    support = tgt.abs() > eps
    off_support = ~support

    support_mass = float(probs[support].sum().item())
    off_support_mass = float(probs[off_support].sum().item())
    max_off_support_prob = float(probs[off_support].max().item()) if off_support.any() else 0.0

    return {
        "support_mass": support_mass,
        "off_support_mass": off_support_mass,
        "max_off_support_prob": max_off_support_prob,
    }


##### TRAINING #####

class RBMTrainer:
    """
    Thin trainer wrapper around the working pooled-CD training logic.

    This is intentionally close to the reference implementation, but the
    responsibilities are separated more clearly:
        - model owns energies / amplitudes / overlaps
        - loader owns batching
        - trainer owns optimization / logging
    """

    def __init__(
        self,
        model: ComplexRBM,
        *,
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs: Optional[dict] = None,
    ):
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)

    def fit(
        self,
        loader: PooledTomographyLoader,
        *,
        epochs: int = 70,
        k: int = 10,
        lr: float = 1e-1,
        log_every: int = 5,
        target: Optional[torch.Tensor] = None,
        bases: Optional[List[Tuple[str, ...]]] = None,
        space: Optional[torch.Tensor] = None,
        grad_clip_norm: float = 10.0,
        print_metrics: bool = True,
    ):
        model = self.model
        if model.stop_training:
            return {"epoch": []}

        opt = self.optimizer_cls(
            [
                {"params": list(model.rbm_am.parameters()), "lr": lr},
                {"params": list(model.rbm_ph.parameters()), "lr": lr},
            ],
            **self.optimizer_kwargs,
        )

        params = list(model.rbm_am.parameters()) + list(model.rbm_ph.parameters())

        history = {"epoch": []}
        if target is not None:
            history["Fidelity"], history["KL"] = [], []
            history["ExactNLL"] = []
            history["ExactNLL_Z"] = []
            history["ExactNLL_Rot"] = []
            history["Z_KL"] = []
            history["RotKLMean"] = []
            history["RotKLMax"] = []
            history["SupportMass"] = []
            history["OffSupportMass"] = []
            history["MaxOffSupportProb"] = []
            history["GradNormAM"] = []
            history["GradNormPH"] = []
            history["Pos"] = []
            history["Neg"] = []
            history["ZPos"] = []
            history["RotPos"] = []

        if space is None:
            space = model.generate_hilbert_space()

        for ep in range(1, epochs + 1):
            grad_am_epoch = []
            grad_ph_epoch = []
            pos_epoch = []
            neg_epoch = []
            zpos_epoch = []
            rotpos_epoch = []

            for pos_batch, neg_batch, bases_batch in loader.iter_epoch():
                pos_batch = pos_batch.to(model.device, dtype=DTYPE)
                neg_batch = neg_batch.to(model.device, dtype=DTYPE)

                L_pos, L_z_only, L_rot_only, cnt_z, cnt_rot = model.positive_phase_loss(pos_batch, bases_batch)
                B_pos = float(pos_batch.shape[0])

                L_neg, B_neg = model.negative_phase_loss(k, neg_batch)

                pos_term = L_pos / B_pos
                neg_term = L_neg / B_neg
                loss = pos_term - neg_term

                opt.zero_grad()
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

                grad_am = am_sq ** 0.5
                grad_ph = ph_sq ** 0.5

                torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
                opt.step()

                grad_am_epoch.append(grad_am)
                grad_ph_epoch.append(grad_ph)
                pos_epoch.append(float(pos_term.item()))
                neg_epoch.append(float(neg_term.item()))
                if cnt_z > 0:
                    zpos_epoch.append(float((L_z_only / cnt_z).item()))
                if cnt_rot > 0:
                    rotpos_epoch.append(float((L_rot_only / cnt_rot).item()))

                if model.stop_training:
                    break

            if (target is not None) and (ep % log_every == 0):
                with torch.no_grad():
                    fid_val = fidelity(model, target, space=space)
                    kl_val = average_basis_kl(model, target, space=space, bases=bases)

                    exact_diag = exact_nll_breakdown(
                        model,
                        loader.ds.train_samples,
                        loader.ds.train_bases_as_tuples(),
                        space=space,
                    )
                    kl_diag = kl_breakdown(model, target, space=space, bases=bases)
                    support_diag = support_mass_stats(model, target, space=space)

                history["epoch"].append(ep)
                history["Fidelity"].append(fid_val)
                history["KL"].append(kl_val)
                history["ExactNLL"].append(exact_diag["exact_nll_total"])
                history["ExactNLL_Z"].append(exact_diag["exact_nll_z"])
                history["ExactNLL_Rot"].append(exact_diag["exact_nll_rot"])
                history["Z_KL"].append(kl_diag["z_kl"])
                history["RotKLMean"].append(kl_diag["rot_kl_mean"])
                history["RotKLMax"].append(kl_diag["rot_kl_max"])
                history["SupportMass"].append(support_diag["support_mass"])
                history["OffSupportMass"].append(support_diag["off_support_mass"])
                history["MaxOffSupportProb"].append(support_diag["max_off_support_prob"])
                history["GradNormAM"].append(float(np.mean(grad_am_epoch)) if grad_am_epoch else float("nan"))
                history["GradNormPH"].append(float(np.mean(grad_ph_epoch)) if grad_ph_epoch else float("nan"))
                history["Pos"].append(float(np.mean(pos_epoch)) if pos_epoch else float("nan"))
                history["Neg"].append(float(np.mean(neg_epoch)) if neg_epoch else float("nan"))
                history["ZPos"].append(float(np.mean(zpos_epoch)) if zpos_epoch else float("nan"))
                history["RotPos"].append(float(np.mean(rotpos_epoch)) if rotpos_epoch else float("nan"))

                if print_metrics:
                    print(
                        f"Epoch {ep}: "
                        f"Fidelity = {fid_val:.6f} | "
                        f"KL = {kl_val:.6f} | "
                        f"exactNLL = {exact_diag['exact_nll_total']:.6f} | "
                        f"zKL = {kl_diag['z_kl']:.6f} | "
                        f"rotKL = {kl_diag['rot_kl_mean']:.6f} | "
                        f"support = {support_diag['support_mass']:.6f} | "
                        f"off = {support_diag['off_support_mass']:.6f} | "
                        f"maxOff = {support_diag['max_off_support_prob']:.6f} | "
                        f"g_am = {history['GradNormAM'][-1]:.6f} | "
                        f"g_ph = {history['GradNormPH'][-1]:.6f} | "
                        f"pos = {history['Pos'][-1]:.6f} | "
                        f"neg = {history['Neg'][-1]:.6f} | "
                        f"zpos = {history['ZPos'][-1]:.6f} | "
                        f"rotpos = {history['RotPos'][-1]:.6f}"
                    )

            if model.stop_training:
                break

        return history


##### PLOTTING #####

def plot_phase_comparison(model, target_state, space):
    with torch.no_grad():
        psi_m = model.psi_complex_normalized(space).reshape(-1).to(torch.cdouble)
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


def plot_training_curves(history):
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
    ax1.set_title("RBM Tomography - pooled autodiff CD with constant LR")
    ax1.grid(True, alpha=0.3)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")
    fig.tight_layout()


##### MAIN #####

def main():
    paths = ExperimentPaths()
    cfg = TrainingConfig()

    torch.manual_seed(cfg.seed)

    data = TomographyDataset(
        train_path=paths.train_values,
        psi_path=paths.state_vector,
        train_bases_path=paths.train_bases,
        bases_path=paths.eval_bases,
        device=DEVICE,
    )

    model = ComplexRBM(
        num_visible=data.num_visible(),
        num_hidden=data.num_visible(),
        unitary_dict=create_unitary_dict(),
        device=DEVICE,
    )

    loader = RBMTomographyLoader(
        data,
        pos_batch_size=cfg.pos_batch_size,
        neg_batch_size=cfg.neg_batch_size,
        device=DEVICE,
        dtype=DTYPE,
    )

    trainer = RBMTrainer(model)
    space = model.generate_hilbert_space()

    history = trainer.fit(
        loader,
        epochs=cfg.epochs,
        k=cfg.cd_k,
        lr=cfg.learning_rate,
        log_every=cfg.log_every,
        target=data.target(),
        bases=data.eval_bases(),
        space=space,
        grad_clip_norm=cfg.grad_clip_norm,
        print_metrics=True,
    )

    plot_phase_comparison(model, data.target(), space)
    plot_training_curves(history)
    plt.show()


if __name__ == "__main__":
    main()
