import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_handling import load_measurements_txt, load_state_txt, MeasurementDataset, MeasurementLoader


from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable

import numpy as np
from math import ceil, sqrt, prod

import torch
import torch.nn as nn
from torch.nn import functional as F

import matplotlib.pyplot as plt


DEVICE = "cpu"
DTYPE = torch.double  # RBM energies in float64 for stability




############################################################
# SUBSET + CANONICAL BASIS ORDER HELPERS
############################################################

class MeasurementSubset:
    """
    View onto a MeasurementDataset restricted to a subset of indices.
    Compatible with MeasurementLoader's expectations.
    """

    def __init__(self, base: MeasurementDataset, indices: torch.Tensor):
        indices = indices.to(torch.long)
        self.base = base
        self.indices = indices
        self.num_qubits = base.num_qubits

        # Values
        self.values = base.values[indices]  # (N_sub, nqubits) uint8

        # Bases / implicit basis
        if base.bases is not None:
            self.bases = [base.bases[int(i)] for i in indices]
            self.implicit_basis = None
        else:
            self.bases = None
            self.implicit_basis = base.implicit_basis

        # System params
        if base.system_params is not None:
            self.system_params = base.system_params[indices]
        else:
            self.system_params = None

        # z_mask for this subset
        self.z_mask = base.z_mask[indices]

    def __len__(self) -> int:
        return int(self.values.shape[0])


def _sliding_window_bases(window: Tuple[str, ...], num_qubits: int,
                          background: str = "Z") -> List[str]:
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


def _infer_basis_order(codes_sorted: List[str], nqubits: int) -> List[str]:
    """
    Canonical order (matching generator idea):
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
        for b in _sliding_window_bases(w, nqubits, background="Z"):
            if b in codes_set and b not in order:
                order.append(b)

    for c in codes_sorted:
        if c not in order:
            order.append(c)

    return order


def infer_eval_bases_from_dataset(ds: MeasurementDataset) -> List[Tuple[str, ...]]:
    """Get canonical evaluation bases from a MeasurementDataset."""
    if ds.bases is not None:
        codes = { "".join(row) for row in ds.bases }
    else:
        codes = { "".join(ds.implicit_basis) }

    codes_sorted = sorted(codes)
    ordered_codes = _infer_basis_order(codes_sorted, ds.num_qubits)
    return [tuple(code) for code in ordered_codes]


############################################################
# UNITARIES + ROTATION LOGIC
############################################################

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


############################################################
# BINARY RBM
############################################################

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


############################################################
# COMPLEX WAVE FUNCTION (AMPL + PHASE RBM)
############################################################

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

    # training loop: PLAN A – TWO LOADERS (Z + MIXED)
    def fit(self, loader_z, loader_mixed,
            epochs=70, k=10, lr=1e-1, log_every=5,
            optimizer=torch.optim.SGD, optimizer_args=None,
            target=None, bases=None, space=None, print_metrics=True,
            metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}"):
        if self.stop_training:
            return {"epoch": []}

        optimizer_args = {} if optimizer_args is None else optimizer_args
        params = list(self.rbm_am.parameters()) + list(self.rbm_ph.parameters())
        opt = optimizer(params, lr=lr, **optimizer_args)

        history = {"epoch": []}
        if target is not None and bases is not None:
            history["Fidelity"], history["KL"] = [], []

        if space is None:
            space = self.generate_hilbert_space()

        for ep in range(1, epochs + 1):
            it_z = iter(loader_z)
            it_m = iter(loader_mixed)

            while True:
                # --- Z-basis batch: used for positives AND CD seeds ---
                try:
                    pos_z, bases_z, _ = next(it_z)
                except StopIteration:
                    break

                # --- mixed-basis batch: positives only ---
                try:
                    pos_mix, bases_mix, _ = next(it_m)
                except StopIteration:
                    it_m = iter(loader_mixed)
                    pos_mix, bases_mix, _ = next(it_m)

                pos_z = pos_z.to(self.device, dtype=DTYPE)
                pos_mix = pos_mix.to(self.device, dtype=DTYPE)

                samples = torch.cat([pos_z, pos_mix], dim=0)
                bases_batch = list(bases_z) + list(bases_mix)

                # positive phase
                L_pos = self._positive_phase_loss(samples, bases_batch)
                B_pos = float(samples.shape[0])

                # negative phase: CD chains seeded from Z-basis data
                L_neg, B_neg = self._negative_phase_loss(k, pos_z)

                loss = (L_pos / B_pos) - (L_neg / B_neg)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                opt.step()

                if self.stop_training:
                    break

            if target is not None and bases is not None and (ep % log_every == 0):
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


############################################################
# METRICS
############################################################

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


############################################################
# MAIN TRAINING SCRIPT
############################################################

if __name__ == "__main__":
    torch.manual_seed(1234)

    # ---- paths / layout ----
    meas_directory = Path("measurements")
    state_directory = Path("state_vectors")
    psi_filename = "w_phase_state.txt"
    file_prefix = "w_phase_"

    if not meas_directory.is_dir():
        raise FileNotFoundError(f"Measurement directory not found: {meas_directory}")
    if not state_directory.is_dir():
        raise FileNotFoundError(f"State directory not found: {state_directory}")

    # ---- load measurements into a single dataset ----
    meas_paths = sorted(
        p for p in meas_directory.glob("*.txt") if p.name.startswith(file_prefix)
    )
    if not meas_paths:
        raise FileNotFoundError(f"No per-basis files '{file_prefix}*.txt' in {meas_directory}")

    ds_all = MeasurementDataset(
        file_paths=meas_paths,
        load_fn=load_measurements_txt,
        system_param_keys=None,  # we ignore sys params for tomography here
    )
    nqubits_meas = ds_all.num_qubits

    # ---- load target state ----
    psi_path = state_directory / psi_filename
    if not psi_path.is_file():
        raise FileNotFoundError(f"State file not found: {psi_path}")

    amps_np, state_headers = load_state_txt(psi_path)
    target_state = torch.tensor(amps_np, dtype=torch.cdouble, device=DEVICE)

    dim = target_state.numel()
    nqubits_state = int(round(np.log2(dim)))
    if (1 << nqubits_state) != dim:
        raise ValueError(f"Inconsistent state length {dim}, not a power of 2.")

    if nqubits_state != nqubits_meas:
        raise ValueError(f"nqubits mismatch: state={nqubits_state}, meas={nqubits_meas}")

    # ---- split into computational (Z-only) and mixed ----
    z_mask = ds_all.z_mask
    idx_z = torch.nonzero(z_mask, as_tuple=False).flatten()
    idx_mixed = torch.nonzero(~z_mask, as_tuple=False).flatten()

    if idx_z.numel() == 0:
        raise ValueError("No all-Z rows found (needed for CD negatives).")
    if idx_mixed.numel() == 0:
        print("Warning: no mixed-basis rows found; training will see only Z-basis data.")

    ds_comp = MeasurementSubset(ds_all, idx_z)
    ds_mixed = MeasurementSubset(ds_all, idx_mixed) if idx_mixed.numel() > 0 else ds_comp

    # ---- loaders: Z and mixed ----
    pbs = 100
    loader_comp = MeasurementLoader(ds_comp, batch_size=pbs, shuffle=True)
    loader_mixed = MeasurementLoader(ds_mixed, batch_size=pbs, shuffle=True)

    # ---- canonical evaluation bases for KL ----
    eval_bases = infer_eval_bases_from_dataset(ds_all)

    # ---- model & training hyperparameters ----
    U = create_dict()

    nv = ds_all.num_qubits
    nh = nv
    nn_state = ComplexWaveFunction(num_visible=nv, num_hidden=nh, unitary_dict=U, device=DEVICE)

    epochs = 500
    nbs = 100
    lr = 1e-1
    k_cd = 10
    log_every = 5

    space = nn_state.generate_hilbert_space()

    history = nn_state.fit(
        loader_z=loader_comp,
        loader_mixed=loader_mixed,
        epochs=epochs,
        k=k_cd,
        lr=lr,
        log_every=log_every,
        optimizer=torch.optim.SGD,
        optimizer_args=None,
        target=target_state,
        bases=eval_bases,
        space=space,
        print_metrics=True,
        metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}",
    )

    # ---------- Phase comparison diagnostic ----------
    with torch.no_grad():
        space = nn_state.generate_hilbert_space()
        psi_m = nn_state.psi_complex_normalized(space).reshape(-1).contiguous()
        psi_t = target_state.to(device=DEVICE, dtype=torch.cdouble).reshape(-1).contiguous()

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
        ax1.set_title("RBM Tomography – training metrics (MeasurementDataset loaders)")
        ax1.grid(True, alpha=0.3)
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="best")
        fig.tight_layout()
    else:
        print("No metrics to plot (did you pass target & bases and set log_every?).")

    plt.show()
