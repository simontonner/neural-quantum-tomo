#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import ceil, sqrt, prod
from typing import Iterable, List, Tuple, Optional
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import matplotlib.pyplot as plt

##### DEVICE AND DTYPES #####
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double  # RBM energies in float64 for stability

##### META-LINE FILTER #####
META_PREFIXES = ("HEADER", "STATE", "MEASUREMENT")

def _is_meta_line(s: str) -> bool:
    """Return True if the trimmed line begins with HEADER, STATE, or MEASUREMENT."""
    return bool(re.match(r"^\s*(HEADER|STATE|MEASUREMENT)\b", s))

##### SINGLE-QUBIT UNITARIES (as cdouble) #####
def create_dict(**overrides):
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

    # For Z-basis measurements we want identity (computational basis)
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


def as_complex_unitary(U, device: torch.device):
    """Ensure a (2,2) complex matrix on device."""
    if torch.is_tensor(U):
        if U.dim() != 2 or U.shape != (2, 2):
            raise ValueError(f"as_complex_unitary expects (2,2), got {tuple(U.shape)}")
        return U.to(device=device, dtype=torch.cdouble).contiguous()

    U_t = torch.tensor(U, device=device)
    if U_t.dim() != 2 or U_t.shape != (2, 2):
        raise ValueError(f"as_complex_unitary expects (2,2), got {tuple(U_t.shape)}")
    return U_t.to(dtype=torch.cdouble, device=device).contiguous()


def inverse(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Safe complex inverse: conj(z) / max(|z|^2, eps)."""
    zz = z.to(torch.cdouble)
    return zz.conj() / (zz.abs().pow(2).clamp_min(eps))


##### KRON-APPLY WITHOUT EXPLICIT TENSOR PRODUCT #####
def _kron_mult(matrices: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Apply tensor product of matrices to x without building the big matrix."""
    assert all(torch.is_complex(m) for m in matrices), "unitaries must be complex"
    if not torch.is_complex(x):
        raise TypeError("x must be complex (cdouble)")

    x_cd = x.to(torch.cdouble)
    L = x_cd.shape[0]
    batch = int(x_cd.numel() // L)
    y = x_cd.reshape(L, batch)

    n = [m.size(-1) for m in matrices]
    if prod(n) != L:
        raise ValueError(f"Incompatible sizes: expected leading dim {prod(n)}, got {L}")

    left = L
    for U in reversed(matrices):
        ns = U.shape[-1]
        left //= ns
        y = y.reshape(left, ns, -1)
        y = torch.einsum("ij,ljm->lim", U, y).reshape(left * ns, -1)

    return y.reshape(*x_cd.shape)


def rotate_psi(nn_state, basis: Iterable[str], space: torch.Tensor,
               unitaries: Optional[dict] = None, psi: Optional[torch.Tensor] = None):
    """Rotate psi into a product basis specified by a tuple of 'X','Y','Z'."""
    n_vis = nn_state.num_visible
    basis = list(basis)
    if len(basis) != n_vis:
        raise ValueError(f"rotate_psi: basis length {len(basis)} != num_visible {n_vis}")

    if unitaries is None:
        us = [nn_state.U[b].to(device=nn_state.device, dtype=torch.cdouble) for b in basis]
    else:
        Udict = {k: as_complex_unitary(v, nn_state.device) for k, v in unitaries.items()}
        us = [Udict[b] for b in basis]

    if psi is None:
        x = nn_state.psi_complex(space)
    else:
        if not torch.is_complex(psi):
            raise TypeError("rotate_psi: psi must be complex (cdouble).")
        x = psi.to(device=nn_state.device, dtype=torch.cdouble)

    return _kron_mult(us, x)


##### BASIS-BRANCH ENUMERATION #####
def _rotate_basis_state(nn_state, basis, states, unitaries=None):
    """Enumerate coherent branches for measured batch of states under given basis."""
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

    combos = nn_state.generate_hilbert_space(size=S, device=device)  # (C, S)

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


def _convert_basis_element_to_index(states):
    """Map rows in {0,1}^n to flat indices [0, 2^n - 1] (MSB-first)."""
    s = states.round().to(torch.long)
    n = s.shape[-1]
    shifts = torch.arange(n - 1, -1, -1, device=s.device, dtype=torch.long)
    return (s << shifts).sum(dim=-1)


def rotate_psi_inner_prod(nn_state, basis, states, unitaries=None, psi=None, include_extras=False):
    """Compute overlap for measured outcomes in the given basis."""
    Ut, v = _rotate_basis_state(nn_state, basis, states, unitaries=unitaries)

    if psi is None:
        psi_sel = nn_state.psi_complex(v)
    else:
        if not torch.is_complex(psi):
            raise TypeError("rotate_psi_inner_prod: psi must be complex.")
        idx = _convert_basis_element_to_index(v).long()
        psi_c = psi.to(dtype=torch.cdouble, device=nn_state.device)
        psi_sel = psi_c[idx]

    Upsi_v_c = Ut * psi_sel
    Upsi_c = Upsi_v_c.sum(dim=0)

    if include_extras:
        return Upsi_c, Upsi_v_c, v
    return Upsi_c


##### BINARY RBM #####
class BinaryRBM(nn.Module):
    """Bernoulli/Bernoulli RBM with free energy F(v)."""

    def __init__(self, num_visible, num_hidden=None, zero_weights=False, device: torch.device = DEVICE):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden) if num_hidden else self.num_visible
        self.device = device
        self.initialize_parameters(zero_weights=zero_weights)

    def __repr__(self):
        return f"BinaryRBM(num_visible={self.num_visible}, num_hidden={self.num_hidden}, device='{self.device}')"

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
        """Free energy F(v). Accepts (..., n). Returns (...,)."""
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
        """k-step block Gibbs from initial_state in {0,1}."""
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


##### COMPLEX WAVE FUNCTION (ampl + phase RBM) #####
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

    # control
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

    # psi accessors
    def amplitude(self, v):
        v = v.to(self.device, dtype=DTYPE)
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()

    def phase(self, v):
        v = v.to(self.device, dtype=DTYPE)
        return -0.5 * self.rbm_ph.effective_energy(v)

    def psi_complex(self, v):
        v = v.to(self.device, dtype=DTYPE)
        amp = (-self.rbm_am.effective_energy(v)).exp().sqrt()
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        return amp.to(torch.cdouble) * torch.exp(1j * ph.to(torch.cdouble))

    def psi_complex_normalized(self, v):
        v = v.to(self.device, dtype=DTYPE)
        E = self.rbm_am.effective_energy(v)
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        logZ = torch.logsumexp(-E, dim=0)
        return torch.exp(((-0.5 * E) - 0.5 * logZ).to(torch.cdouble) + 1j * ph.to(torch.cdouble))

    # aliases
    def psi(self, v):
        return self.psi_complex(v)

    def psi_normalized(self, v):
        return self.psi_complex_normalized(v)

    def phase_angle(self, v):
        return self.phase(v)

    # utilities
    def generate_hilbert_space(self, size=None, device=None):
        device = self.device if device is None else device
        size = self.num_visible if size is None else int(size)
        if size > self._max_size:
            raise ValueError(f"Hilbert space too large (n={size} > max={self._max_size}).")
        n = 1 << size
        ar = torch.arange(n, device=device, dtype=torch.long)
        shifts = torch.arange(size - 1, -1, -1, device=device, dtype=torch.long)
        return ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)

    # stable overlap for rotated bases
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

    # loss pieces
    def _positive_phase_loss(self, samples: torch.Tensor,
                             bases_batch: List[Tuple[str, ...]],
                             eps_rot: float = 1e-6):
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

    # training loop
    def fit(self, loader, epochs=70, k=10, lr=1e-1, log_every=5,
            optimizer=torch.optim.SGD, optimizer_args=None,
            target=None, bases=None, space=None, print_metrics=True,
            metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}"):
        if self.stop_training:
            return {"epoch": []}

        optimizer_args = {} if optimizer_args is None else optimizer_args
        params = list(self.rbm_am.parameters()) + list(self.rbm_ph.parameters())
        opt = optimizer(params, lr=lr, **optimizer_args)

        history = {"epoch": []}
        if target is not None:
            history["Fidelity"], history["KL"] = [], []

        if space is None:
            space = self.generate_hilbert_space()

        for ep in range(1, epochs + 1):
            for pos_batch, neg_batch, bases_batch in loader.iter_epoch():
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

            if (target is not None) and (ep % log_every == 0):
                with torch.no_grad():
                    fid_val = fidelity(self, target, space=space, bases=bases)
                    kl_val = KL(self, target, space=space, bases=bases)
                history["epoch"].append(ep)
                history["Fidelity"].append(fid_val)
                history["KL"].append(kl_val)
                if print_metrics:
                    print(metric_fmt.format(ep=ep, fid=fid_val, kl=kl_val))

            if self.stop_training:
                break

        return history


##### METRICS #####
@torch.no_grad()
def fidelity(nn_state, target, space=None, **kwargs):
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
def KL(nn_state, target, space=None, bases=None, **kwargs):
    if bases is None:
        raise ValueError("KL needs bases")
    if not torch.is_complex(target):
        raise TypeError("KL: target must be complex (cdouble)")

    space = nn_state.generate_hilbert_space() if space is None else space

    tgt = target.to(device=nn_state.device, dtype=torch.cdouble).reshape(-1)
    nt = torch.linalg.vector_norm(tgt)
    if nt == 0:
        return 0.0
    tgt_norm = tgt / nt

    psi_norm_cd = nn_state.psi_complex_normalized(space).reshape(-1)

    KL_val = 0.0
    eps = 1e-12

    for basis in bases:
        tgt_psi_r = rotate_psi(nn_state, basis, space, psi=tgt_norm)
        psi_r = rotate_psi(nn_state, basis, space, psi=psi_norm_cd)

        nn_probs_r = (psi_r.abs().to(DTYPE)) ** 2
        tgt_probs_r = (tgt_psi_r.abs().to(DTYPE)) ** 2

        p_sum = tgt_probs_r.sum().clamp_min(eps)
        q_sum = nn_probs_r.sum().clamp_min(eps)
        p = (tgt_probs_r / p_sum).clamp_min(eps)
        q = (nn_probs_r / q_sum).clamp_min(eps)

        KL_val += torch.sum(p * (torch.log(p) - torch.log(q)))

    return (KL_val / len(bases)).item()


##### PER-BASIS DATASET (META-LINE AWARE) #####
class TomographyDataset:
    """
    Reconstructs the original (samples, bases) arrays from per-basis files.

    Layout produced by the updated generator:
      state_vectors/
        w_phase_state.txt
          STATE | ...
          HEADER | ...
          Re Im
          Re Im
          ...

      measurements/
        w_phase_<BASIS>_<shots>.txt
          MEASUREMENT | ...
          HEADER | ...
          <encoded_shot>
          <encoded_shot>
          ...

    Encoding:
      - In per-basis files:
          uppercase letter -> bit 0
          lowercase letter -> bit 1
      - Basis is encoded in the filename via <BASIS> (e.g. 'XXZZ').

    Basis order:
      - Canonical, matching the generator:
          Z^N,
          sliding 'XX' over Z,
          sliding 'XY' over Z,
          then any remaining bases in sorted order.
    """

    def __init__(self,
                 meas_directory: str = "measurements",
                 state_directory: str = "state_vectors",
                 psi_filename: str = "w_phase_state.txt",
                 file_prefix: str = "w_phase_",
                 device: torch.device = DEVICE):

        self.device = device
        meas_dir = Path(meas_directory)
        state_dir = Path(state_directory)

        if not meas_dir.is_dir():
            raise FileNotFoundError(f"Measurement directory not found: {meas_dir}")
        if not state_dir.is_dir():
            raise FileNotFoundError(f"State directory not found: {state_dir}")

        # ---- load target psi (META-line aware) ----
        psi_path = state_dir / psi_filename
        if not psi_path.is_file():
            raise FileNotFoundError(f"State file not found: {psi_path}")

        amps: List[complex] = []
        with psi_path.open("r") as f:
            for line in f:
                s = line.strip()
                if not s or _is_meta_line(s):
                    continue
                parts = s.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid state line '{s}' in {psi_path}")
                re_val, im_val = map(float, parts)
                amps.append(re_val + 1j * im_val)

        if not amps:
            raise ValueError(f"No amplitudes read from {psi_path}")

        self.target_state = torch.tensor(
            amps,
            dtype=torch.cdouble,
            device=self.device,
        )

        dim = self.target_state.numel()
        nqubits = int(round(np.log2(dim)))
        if (1 << nqubits) != dim:
            raise ValueError(f"Inconsistent state length {dim}, not a power of 2.")
        self._nqubits = nqubits

        # ---- collect per-basis files from meas_directory ----
        per_basis_files: List[Path] = []
        for p in meas_dir.glob("*.txt"):
            if p.name.startswith(file_prefix):
                per_basis_files.append(p)

        if not per_basis_files:
            raise FileNotFoundError(f"No per-basis files '{file_prefix}*.txt' in {meas_dir}")

        # parse basis codes from filenames
        code_to_files = {}
        codes = set()
        for p in per_basis_files:
            code = self._parse_basis_code_from_filename(p.name, prefix=file_prefix)
            codes.add(code)
            code_to_files.setdefault(code, []).append(p)

        codes_sorted = sorted(codes)
        basis_order = self._infer_basis_order(codes_sorted, nqubits)

        # ---- reconstruct samples + bases in canonical order ----
        samples = []
        bases_rows: List[Tuple[str, ...]] = []

        for code in basis_order:
            basis = tuple(code)
            L = len(basis)
            if L != nqubits:
                raise ValueError(f"Basis code '{code}' length {L} != nqubits {nqubits}")

            files = sorted(code_to_files[code], key=lambda x: x.name)
            for fpath in files:
                with fpath.open("r") as f:
                    for ln in f:
                        s = ln.strip()
                        if not s or _is_meta_line(s):
                            continue
                        if len(s) != L:
                            raise ValueError(f"Line length {len(s)} != basis width {L} in {fpath}")
                        bits = []
                        for ch in s:
                            if not ch.isalpha():
                                raise ValueError(f"Illegal char '{ch}' in {fpath}: '{ch}'")
                            bits.append(0 if ch.isupper() else 1)
                        samples.append(np.asarray(bits, dtype=np.float32))
                        bases_rows.append(basis)

        if not samples:
            raise ValueError("No samples read from per-basis files.")

        self.train_samples = torch.tensor(
            np.stack(samples, axis=0),
            dtype=DTYPE,
            device=self.device,
        )
        self.train_bases = bases_rows
        self.bases = [tuple(code) for code in basis_order]

        # ---- consistency checks ----
        if self.train_samples.shape[1] != nqubits:
            raise ValueError("Sample width != inferred nqubits")

        widths = {len(row) for row in self.train_bases}
        if len(widths) != 1 or next(iter(widths)) != nqubits:
            raise ValueError("Inconsistent basis widths in reconstructed dataset")

        if self.target_state.numel() != (1 << nqubits):
            raise ValueError("Target state dimension mismatch after reconstruction.")

        # indices of all-Z basis rows (for negative samples)
        z_mask_np = np.array([all(ch == "Z" for ch in row) for row in self.train_bases], dtype=bool)
        self._z_indices = torch.as_tensor(np.nonzero(z_mask_np)[0], dtype=torch.long)
        if self._z_indices.numel() == 0:
            raise ValueError("No all-Z rows found (needed for negatives).")

    @staticmethod
    def _parse_basis_code_from_filename(name: str, prefix: str) -> str:
        # e.g. w_phase_XXZZ_5000.txt -> XXZZ
        stem = Path(name).stem
        if not stem.startswith(prefix):
            raise ValueError(f"Bad file name (prefix): {name}")
        tail = stem[len(prefix):]  # 'XXZZ_5000'
        if "_" not in tail:
            raise ValueError(f"Bad file name (missing '_shots'): {name}")
        code = tail.rsplit("_", 1)[0]
        if not re.fullmatch(r"[XYZ]+", code):
            raise ValueError(f"Invalid basis code '{code}' from {name}")
        return code

    @staticmethod
    def _sliding_window_bases(window: Tuple[str, ...], num_qubits: int, background: str = "Z") -> List[str]:
        """Slide `window` over a `background` string, return codes as strings."""
        w = list(window)
        L = len(w)
        if L == 0 or L > num_qubits:
            return []
        out = []
        for i in range(0, num_qubits - L + 1):
            b = [background] * num_qubits
            b[i:i + L] = w
            out.append("".join(b))
        return out

    def _infer_basis_order(self, codes_sorted: List[str], nqubits: int) -> List[str]:
        """
        Canonical order (matching generator):
          Z^N,
          sliding 'XX' over Z,
          sliding 'XY' over Z,
          then any remaining codes in sorted order.
        """
        codes_set = set(codes_sorted)
        order: List[str] = []

        z_code = "Z" * nqubits
        if z_code in codes_set:
            order.append(z_code)

        for w in [("X", "X"), ("X", "Y")]:
            for b in self._sliding_window_bases(w, nqubits, background="Z"):
                if b in codes_set and b not in order:
                    order.append(b)

        for c in codes_sorted:
            if c not in order:
                order.append(c)

        return order

    # API matching original TomographyDataset usage
    def __len__(self):
        return int(self.train_samples.shape[0])

    def num_visible(self) -> int:
        return int(self.train_samples.shape[1])

    def z_indices(self) -> torch.Tensor:
        return self._z_indices.clone()

    def train_bases_as_tuples(self) -> List[Tuple[str, ...]]:
        return list(self.train_bases)

    def eval_bases(self) -> List[Tuple[str, ...]]:
        return list(self.bases)

    def target(self) -> torch.Tensor:
        return self.target_state


##### LOADER (unchanged semantics) #####
class RBMTomographyLoader:
    """Yield (pos_batch, neg_batch, bases_batch) minibatches for training."""

    def __init__(self, dataset: TomographyDataset, pos_batch_size: int = 100,
                 neg_batch_size: Optional[int] = None,
                 device: torch.device = DEVICE, dtype: torch.dtype = DTYPE,
                 strict: bool = True):
        self.ds = dataset
        self.pos_bs = int(pos_batch_size)
        self.neg_bs = int(neg_batch_size or pos_batch_size)
        self.device = device
        self.dtype = dtype
        self.strict = strict
        self._gen: Optional[torch.Generator] = None

        n = self.ds.num_visible()
        bases_seq = self.ds.train_bases_as_tuples()
        if any(len(row) != n for row in bases_seq):
            raise ValueError("RBMTomographyLoader: inconsistent basis widths in dataset")
        if self.ds.z_indices().numel() == 0:
            raise ValueError("RBMTomographyLoader: Z-only pool is empty (need negatives)")

    def set_seed(self, seed: Optional[int]):
        if seed is None:
            self._gen = None
        else:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            self._gen = g

    def __len__(self):
        return ceil(len(self.ds) / self.pos_bs)

    def iter_epoch(self):
        N = len(self.ds)
        n_batches = ceil(N / self.pos_bs)

        perm = torch.randperm(N, generator=self._gen) if self._gen is not None else torch.randperm(N)
        pos_samples = self.ds.train_samples[perm].to(self.device, dtype=self.dtype)

        bases_list = self.ds.train_bases_as_tuples()
        perm_idx = perm.detach().cpu().tolist()
        pos_bases_perm = [bases_list[i] for i in perm_idx]

        z_pool = self.ds.z_indices()
        pool_len = z_pool.numel()
        if self._gen is None:
            neg_choices = torch.randint(pool_len, size=(n_batches * self.neg_bs,))
        else:
            neg_choices = torch.randint(pool_len, size=(n_batches * self.neg_bs,), generator=self._gen)
        neg_rows = z_pool[neg_choices]
        neg_samples_all = self.ds.train_samples[neg_rows].to(self.device, dtype=self.dtype)

        for b in range(n_batches):
            start = b * self.pos_bs
            end = min(start + self.pos_bs, N)
            pos_batch = pos_samples[start:end]

            nb_start = b * self.neg_bs
            nb_end = nb_start + self.neg_bs
            neg_batch = neg_samples_all[nb_start:nb_end]

            bases_batch = pos_bases_perm[start:end]

            if self.strict:
                if len(bases_batch) != pos_batch.shape[0]:
                    raise RuntimeError("Loader invariant broken: bases_batch length mismatch")
                if pos_batch.shape[1] != self.ds.num_visible():
                    raise RuntimeError("Loader invariant broken: pos_batch width != num_visible")
                if neg_batch.shape[1] != self.ds.num_visible():
                    raise RuntimeError("Loader invariant broken: neg_batch width != num_visible")

            yield pos_batch, neg_batch, bases_batch


##### MAIN TRAINING SCRIPT #####
if __name__ == "__main__":
    torch.manual_seed(1234)

    data = TomographyDataset(
        meas_directory="measurements",
        state_directory="state_vectors",
        psi_filename="w_phase_state.txt",
        file_prefix="w_phase_",
        device=DEVICE,
    )

    U = create_dict()

    nv = data.num_visible()
    nh = nv
    nn_state = ComplexWaveFunction(num_visible=nv, num_hidden=nh, unitary_dict=U, device=DEVICE)

    epochs = 70
    pbs = 100
    nbs = 100
    lr = 1e-1
    k_cd = 10
    log_every = 5

    loader = RBMTomographyLoader(data, pos_batch_size=pbs, neg_batch_size=nbs,
                                 device=DEVICE, dtype=DTYPE)

    space = nn_state.generate_hilbert_space()

    history = nn_state.fit(
        loader,
        epochs=epochs,
        k=k_cd,
        lr=lr,
        log_every=log_every,
        optimizer=torch.optim.SGD,
        optimizer_args=None,
        target=data.target(),
        bases=data.eval_bases(),
        space=space,
        print_metrics=True,
        metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}",
    )

    # ---------- Phase comparison diagnostic ----------
    with torch.no_grad():
        space = nn_state.generate_hilbert_space()
        psi_m = nn_state.psi_complex_normalized(space).reshape(-1).contiguous()
        psi_t = data.target().to(device=DEVICE, dtype=torch.cdouble).reshape(-1).contiguous()

        nm = torch.linalg.vector_norm(psi_m)
        nt = torch.linalg.vector_norm(psi_t)
        if nm > 0:
            psi_m = psi_m / nm
        if nt > 0:
            psi_t = psi_t / nt

        ip = torch.sum(psi_t.conj() * psi_m)
        if ip.abs() > 1e-12:
            theta = torch.angle(ip)
        else:
            j = int(torch.argmax(psi_t.abs()))
            theta = torch.angle(psi_m[j]) - torch.angle(psi_t[j])
        psi_m_al = psi_m * torch.exp(-1j * theta)

        phi_t = torch.angle(psi_t).cpu().numpy()
        phi_m = torch.angle(psi_m_al).cpu().numpy()
        dphi = np.remainder((phi_m - phi_t) + np.pi, 2.0 * np.pi) - np.pi

        probs = (psi_t.abs() ** 2).cpu().numpy()
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        mass_cut = 0.99
        k_cap = 512
        k_sel = int(min(np.searchsorted(cum, mass_cut) + 1, k_cap, len(order)))
        sel = order[:k_sel]

        fig_p, axp = plt.subplots(figsize=(7.2, 3.8), dpi=150)
        axp.plot(range(k_sel), phi_t[sel], marker=".", linestyle="", label="target phase")
        axp.plot(range(k_sel), phi_m[sel], marker="x", linestyle="", label="model phase (aligned)")
        axp.set_xlabel("basis states (sorted by target mass)")
        axp.set_ylabel("phase [rad]")
        axp.set_title("Phase comparison – top 99% mass")
        axp.grid(True, alpha=0.3)
        axp.legend()
        fig_p.tight_layout()

        fig_e, axe = plt.subplots(figsize=(7.2, 3.8), dpi=150)
        axe.plot(range(k_sel), dphi[sel], marker=".", linestyle="", label="Δphase (wrapped)")
        axe.axhline(0.0, linewidth=1.0)
        axe.set_xlabel("basis states (sorted by target mass)")
        axe.set_ylabel("Δphase [rad] in [-π, π]")
        axe.set_title("Phase error (global phase aligned)")
        axe.grid(True, alpha=0.3)
        axe.legend()
        fig_e.tight_layout()

    # ---------- Metrics plot ----------
    ep_hist = history.get("epoch", [])
    if ep_hist and ("Fidelity" in history) and ("KL" in history):
        fig, ax1 = plt.subplots(figsize=(6.0, 4.0), dpi=140)
        ax2 = ax1.twinx()
        ax1.plot(ep_hist, history["Fidelity"], marker="o", label="Fidelity")
        ax2.plot(ep_hist, history["KL"], marker="s", linestyle="--", label="KL")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel(r"$|\langle \psi_t \mid \psi \rangle|^2$")
        ax2.set_ylabel(r"KL$(p\,\|\,q)$")
        ax1.set_title("RBM Tomography – training metrics (per-basis loader, original loss)")
        ax1.grid(True, alpha=0.3)
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="best")
        fig.tight_layout()
    else:
        print("No metrics to plot (did you pass target & bases and set log_every?).")

    plt.show()
