# RBM wavefunction — explicit-gradient (improved)
# -----------------------------------------------------------------------------
# Design goals
# - Complex-first: all ψ/rotations are torch.cdouble to avoid mixed-dtype bugs.
# - One device end-to-end: no implicit CPU<->CUDA hops.
# - Numerics: safe inverse, per-basis renorm for KL, contiguous complex ops.
# - Training: explicit positive-phase for both RBMs (amplitude+phase),
#             negative phase only for amplitude RBM (CD/PCD),
#             Adam by default, optional grad clipping, optional phase grad weight.
# - Extras: optional PCD with persistent chains; global-phase-aligned diagnostics.
# -----------------------------------------------------------------------------

from math import ceil, sqrt, prod
from itertools import chain
from typing import Iterable, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------------------------
# Device & dtypes
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double  # real-valued RBM parameters and energies


# -------------------------------
# Unitaries (as cdouble)
# -------------------------------
def create_dict(**overrides):
    """
    Build {X,Y,Z} single-qubit unitaries as cdouble.
    Y uses [[1,-i],[1,i]]/√2 so branch orientation matches our measurement convention.
    """
    inv_sqrt2 = 1.0 / sqrt(2.0)

    X = inv_sqrt2 * torch.tensor([[1.0+0.0j,  1.0+0.0j],
                                  [1.0+0.0j, -1.0+0.0j]],
                                 dtype=torch.cdouble, device=DEVICE)

    Y = inv_sqrt2 * torch.tensor([[1.0+0.0j,  0.0-1.0j],
                                  [1.0+0.0j,  0.0+1.0j]],
                                 dtype=torch.cdouble, device=DEVICE)

    Z = torch.tensor([[1.0+0.0j, 0.0+0.0j],
                      [0.0+0.0j, 1.0+0.0j]],
                     dtype=torch.cdouble, device=DEVICE)

    U = {"X": X.contiguous(), "Y": Y.contiguous(), "Z": Z.contiguous()}
    for name, mat in overrides.items():  # normalize overrides once
        U[name] = as_complex_unitary(mat, DEVICE)
    return U


def as_complex_unitary(U, device: torch.device):
    """Return a (2,2) complex (cdouble) matrix on `device` (contiguous)."""
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


# -------------------------------
# Kronecker apply without forming the full kron
# -------------------------------
def _kron_mult(matrices: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """
    Apply (⊗_s U_s) to ψ without forming the Kronecker explicitly.
    Inputs must be complex; implemented via reshape+einsum.
    """
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
        y = torch.einsum('ij,ljm->lim', U, y).reshape(left * ns, -1)

    return y.reshape(*x_cd.shape)


def rotate_psi(nn_state, basis: Iterable[str], space: torch.Tensor,
               unitaries: Optional[dict] = None, psi: Optional[torch.Tensor] = None):
    """
    Rotate ψ into `basis`.
    - basis: sequence of 'X'/'Y'/'Z' of length num_visible
    - space: computational basis (2^n, n) in DTYPE
    - psi  : optional complex psi; otherwise uses nn_state.psi_complex(space)
    """
    n_vis = nn_state.num_visible
    basis = list(basis)
    if len(basis) != n_vis:
        raise ValueError(f"rotate_psi: basis length {len(basis)} != num_visible {n_vis}")

    if unitaries is None:
        us = [nn_state.U[b].to(device=nn_state.device, dtype=torch.cdouble) for b in basis]
    else:
        Udict = {k: as_complex_unitary(v, nn_state.device) for k, v in unitaries.items()}
        us = [Udict[b] for b in basis]

    x = psi.to(device=nn_state.device, dtype=torch.cdouble) if psi is not None else nn_state.psi_complex(space)
    return _kron_mult(us, x)


# -------------------------------
# Rotation kernel for inner products
# -------------------------------
def _rotate_basis_state(nn_state, basis, states, unitaries=None):
    """
    Enumerate non-Z sites and compute per-branch weights and candidate states.

    Returns:
      Ut : (C, B) complex — product over sites of U[in, out]
      v  : (C, B, n) DTYPE — candidate outcomes for rotated sites
    """
    device = nn_state.device
    n_vis = nn_state.num_visible
    basis_seq = list(basis)

    if len(basis_seq) != n_vis:
        raise ValueError(f"_rotate_basis_state: basis length {len(basis_seq)} != num_visible {n_vis}")
    if states.shape[-1] != n_vis:
        raise ValueError(f"_rotate_basis_state: states width {states.shape[-1]} != num_visible {n_vis}")

    sites = [i for i, b in enumerate(basis_seq) if b != "Z"]
    if len(sites) == 0:
        v = states.unsqueeze(0)  # (1, B, n)
        Ut = torch.ones(v.shape[:-1], dtype=torch.cdouble, device=device)  # (1, B)
        return Ut, v

    Ulist = []
    src = nn_state.U if unitaries is None else unitaries
    for i in sites:
        Ulist.append(as_complex_unitary(src[basis_seq[i]], device).reshape(2, 2).contiguous())
    Uc = torch.stack(Ulist, dim=0)  # (S, 2, 2)

    S = len(sites)
    B = states.shape[0]
    C = 2 ** S

    combos = nn_state.generate_hilbert_space(size=S, device=device)  # (C, S)
    v = states.unsqueeze(0).repeat(C, 1, 1)                          # (C, B, n)
    v[:, :, sites] = combos.unsqueeze(1)
    v = v.contiguous()

    inp_sb   = states[:, sites].round().long().T               # (S, B)
    outp_csb = v[:, :, sites].round().long().permute(0, 2, 1)  # (C, S, B)
    inp_csb  = inp_sb.unsqueeze(0).expand(C, -1, -1)           # (C, S, B)

    s_idx = torch.arange(S, device=device).view(1, S, 1).expand(C, S, B)  # (C, S, B)
    sel = Uc[s_idx, inp_csb, outp_csb]                                    # (C, S, B) complex

    Ut = sel.prod(dim=1)  # (C, B)
    return Ut.to(torch.cdouble), v


def _convert_basis_element_to_index(states):
    """Convert bit-rows (B, n) to flat indices (B,), MSB first."""
    s = states.round().to(torch.long)
    n = s.shape[-1]
    shifts = torch.arange(n - 1, -1, -1, device=s.device, dtype=torch.long)
    return (s << shifts).sum(dim=-1)


def rotate_psi_inner_prod(nn_state, basis, states, unitaries=None, psi=None, include_extras=False):
    """
    Inner-product components for measuring in `basis`.

    Returns:
      total  : (B,) complex
      (opt) branches : (C,B) complex
      (opt) v        : (C,B,n) DTYPE
    """
    Ut, v = _rotate_basis_state(nn_state, basis, states, unitaries=unitaries)

    if psi is None:
        # Do not backprop through psi here; gradients are explicit.
        with torch.no_grad():
            psi_sel = nn_state.psi_complex(v)  # (C,B)
    else:
        if not torch.is_complex(psi):
            raise TypeError("rotate_psi_inner_prod: psi must be complex.")
        idx = _convert_basis_element_to_index(v).long()  # (C,B)
        psi_c = psi.to(dtype=torch.cdouble, device=nn_state.device)
        psi_sel = psi_c[idx]

    Upsi_v_c = Ut * psi_sel
    Upsi_c   = Upsi_v_c.sum(dim=0)

    if include_extras:
        return Upsi_c, Upsi_v_c, v
    return Upsi_c


# -------------------------------
# Explicit gradient utils
# -------------------------------
def vector_to_grads(vec, parameters):
    """Write a flattened gradient vector into a module's .grad buffers."""
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {torch.typename(vec)}")

    offset = 0
    for p in parameters:
        n = p.numel()
        if offset + n > vec.numel():
            raise ValueError("Gradient vector is too short for parameter shapes.")
        g_slice = vec[offset: offset + n].view_as(p)
        if p.grad is None or p.grad.shape != p.shape or p.grad.dtype != p.dtype or p.grad.device != p.device:
            p.grad = torch.empty_like(p)
        p.grad.copy_(g_slice.to(dtype=p.dtype, device=p.device))
        offset += n

    if offset != vec.numel():
        raise ValueError(f"Gradient vector has extra elements: used {offset}, total {vec.numel()}.")


# -------------------------------
# RBM (Bernoulli/Bernoulli)
# -------------------------------
class BinaryRBM(nn.Module):
    """
    Minimal Bernoulli/Bernoulli RBM used twice: amplitude and phase nets.
    All energy math is float64 (DTYPE) for stability.
    """

    def __init__(self, num_visible, num_hidden=None, zero_weights=False, device: torch.device = DEVICE):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden) if num_hidden else self.num_visible
        self.num_pars = (self.num_visible * self.num_hidden) + self.num_visible + self.num_hidden
        self.device = device
        self.initialize_parameters(zero_weights=zero_weights)

    def __repr__(self):
        return f"BinaryRBM(num_visible={self.num_visible}, num_hidden={self.num_hidden}, device='{self.device}')"

    def initialize_parameters(self, zero_weights=False):
        gen_tensor = torch.zeros if zero_weights else torch.randn
        self.weights = nn.Parameter(
            (gen_tensor(self.num_hidden, self.num_visible, device=self.device, dtype=DTYPE)
             / np.sqrt(self.num_visible)),
            requires_grad=False,
        )
        self.visible_bias = nn.Parameter(torch.zeros(self.num_visible, device=self.device, dtype=DTYPE),
                                         requires_grad=False)
        self.hidden_bias = nn.Parameter(torch.zeros(self.num_hidden, device=self.device, dtype=DTYPE),
                                        requires_grad=False)

    def effective_energy(self, v):
        """
        E(v) = -v·a - sum_j softplus(b_j + W_j·v)
        Returns shape matching the input batch rank.
        """
        unsq = False
        if v.dim() < 2:
            v = v.unsqueeze(0)
            unsq = True
        v = v.to(self.weights)
        visible_bias_term = torch.matmul(v, self.visible_bias)
        hid_bias_term = F.softplus(F.linear(v, self.weights, self.hidden_bias)).sum(-1)
        out = -(visible_bias_term + hid_bias_term)
        return out.squeeze(0) if unsq else out

    def effective_energy_gradient(self, v, reduce=True):
        """
        Gradients of E(v) w.r.t. parameters (positive phase).
        If reduce=False, returns per-sample grads with trailing grad-dim.
        """
        v = (v.unsqueeze(0) if v.dim() < 2 else v).to(self.weights)  # (..., V)
        prob = self.prob_h_given_v(v)                                # (..., H)

        if reduce:
            W_grad = -torch.matmul(prob.transpose(0, -1), v)         # (H, V)
            vb_grad = -torch.sum(v, dim=0)                           # (V,)
            hb_grad = -torch.sum(prob, dim=0)                        # (H,)
            return torch.cat([W_grad.reshape(-1), vb_grad, hb_grad], dim=0)

        W_grad = -torch.einsum("...h,...v->...hv", prob, v)          # (..., H, V)
        vb_grad = -v                                                 # (..., V)
        hb_grad = -prob                                              # (..., H)
        vec = [W_grad.view(*v.shape[:-1], -1), vb_grad, hb_grad]
        return torch.cat(vec, dim=-1)                                # (..., num_pars)

    def prob_v_given_h(self, h, out=None):
        unsq = False
        if h.dim() < 2:
            h = h.unsqueeze(0)
            unsq = True
        res = torch.matmul(h, self.weights.data).add_(self.visible_bias.data).sigmoid_().clamp_(0, 1)
        if out is not None:
            out.copy_(res.squeeze(0) if unsq and out.dim() == 1 else res)
            return out
        return res.squeeze(0) if unsq else res

    def prob_h_given_v(self, v, out=None):
        unsq = False
        if v.dim() < 2:
            v = v.unsqueeze(0)
            unsq = True
        res = torch.matmul(v, self.weights.data.t()).add_(self.hidden_bias.data).sigmoid_().clamp_(0, 1)
        if out is not None:
            out.copy_(res.squeeze(0) if unsq and out.dim() == 1 else res)
            return out
        return res.squeeze(0) if unsq else res

    def sample_v_given_h(self, h, out=None):
        probs = self.prob_v_given_h(h)
        return torch.bernoulli(probs, out=out) if out is not None else torch.bernoulli(probs)

    def sample_h_given_v(self, v, out=None):
        probs = self.prob_h_given_v(v)
        return torch.bernoulli(probs, out=out) if out is not None else torch.bernoulli(probs)

    def gibbs_steps(self, k, initial_state, overwrite=False):
        """
        k-step block Gibbs starting at `initial_state`.
        overwrite=False preserves caller tensor unless explicitly allowed.
        """
        v = (initial_state if overwrite else initial_state.clone()).to(self.weights)
        h = torch.zeros(*v.shape[:-1], self.num_hidden, device=self.device, dtype=DTYPE)
        for _ in range(k):
            self.sample_h_given_v(v, out=h)
            self.sample_v_given_h(h, out=v)
        return v


# -------------------------------
# Complex wavefunction = amplitude RBM + phase RBM
# -------------------------------
class ComplexWaveFunction:
    """
    Two real RBMs define magnitude and phase over bitstrings:
      psi(v) = exp(-E_am(v)/2) * exp(i * (-E_ph(v)/2))
    """

    def __init__(self, num_visible, num_hidden=None, unitary_dict=None, module=None, device: torch.device = DEVICE):
        self.device = device
        if module is None:
            self.rbm_am = BinaryRBM(num_visible, num_hidden, device=self.device)
            self.rbm_ph = BinaryRBM(num_visible, num_hidden, device=self.device)
        else:
            self.rbm_am = module.to(self.device)
            self.rbm_am.device = self.device
            self.rbm_ph = module.to(self.device).clone()
            self.rbm_ph.device = self.device

        self.num_visible = self.rbm_am.num_visible
        self.num_hidden = self.rbm_am.num_hidden

        raw = unitary_dict if unitary_dict is not None else create_dict()
        self.U = {k: as_complex_unitary(v, self.device) for k, v in raw.items()}

        self._stop_training = False
        self._max_size = 20
        self._pcd_state: Optional[torch.Tensor] = None  # persistent chain (V-batch, n)

    # control
    @property
    def stop_training(self):
        return self._stop_training

    @stop_training.setter
    def stop_training(self, new_val):
        if isinstance(new_val, bool):
            self._stop_training = new_val
        else:
            raise ValueError("stop_training must be bool!")

    @property
    def max_size(self):
        return self._max_size

    def reinitialize_parameters(self):
        self.rbm_am.initialize_parameters()
        self.rbm_ph.initialize_parameters()
        self._pcd_state = None

    # amplitudes/phases
    def amplitude(self, v):
        """|psi(v)| as exp(-E_am/2)."""
        v = v.to(self.device, dtype=DTYPE)
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()

    def phase(self, v):
        """Phase angle from the phase RBM: -E_ph/2."""
        v = v.to(self.device, dtype=DTYPE)
        return -0.5 * self.rbm_ph.effective_energy(v)

    def psi_complex(self, v):
        """psi(v) as complex tensor (cdouble)."""
        v = v.to(self.device, dtype=DTYPE)
        amp = (-self.rbm_am.effective_energy(v)).exp().sqrt()
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        return amp.to(torch.cdouble) * torch.exp(1j * ph.to(torch.cdouble))

    def psi_complex_normalized(self, v):
        """Normalized psi via exact partition on the amplitude RBM."""
        v = v.to(self.device, dtype=DTYPE)
        E = self.rbm_am.effective_energy(v)
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        logZ = torch.logsumexp(-E, dim=0)
        return torch.exp(((-0.5 * E) - 0.5 * logZ).to(torch.cdouble) + 1j * ph.to(torch.cdouble))

    # user-facing aliases
    def psi(self, v): return self.psi_complex(v)
    def psi_normalized(self, v): return self.psi_complex_normalized(v)
    def phase_angle(self, v): return self.phase(v)

    def generate_hilbert_space(self, size=None, device=None):
        """Enumerate computational basis as a (2^size, size) bit-matrix."""
        device = self.device if device is None else device
        size = self.rbm_am.num_visible if size is None else int(size)
        if size > self.max_size:
            raise ValueError("Hilbert space too large!")
        n = 1 << size
        ar = torch.arange(n, device=device, dtype=torch.long)
        shifts = torch.arange(size - 1, -1, -1, device=device, dtype=torch.long)
        return ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)

    # gradients (complex mapping; final grads are real DTYPE)
    def am_grads(self, v):
        g = self.rbm_am.effective_energy_gradient(v, reduce=False)
        return g.to(torch.cdouble)

    def ph_grads(self, v):
        g = self.rbm_ph.effective_energy_gradient(v, reduce=False)
        return (1j * g.to(torch.cdouble))

    def rotated_gradient(self, basis, sample):
        """Positive-phase grads under a rotated measurement basis."""
        Upsi, Upsi_v, v = rotate_psi_inner_prod(self, basis, sample, include_extras=True)
        inv_Upsi = inverse(Upsi)  # (B,)
        raw_grads = [self.am_grads(v), self.ph_grads(v)]  # complex
        rotated_grad = [torch.einsum("cb,cbg->bg", Upsi_v, g) for g in raw_grads]
        grad = [torch.einsum("b,bg->g", inv_Upsi, rg).real.to(DTYPE) for rg in rotated_grad]
        return grad

    def gradient(self, samples, bases=None, phase_weight: float = 1.0):
        """
        Positive-phase gradients. If `bases` is None, only amplitude grads (Z).
        Otherwise group identical basis rows and accumulate rotated grads.
        `phase_weight` can down/up-weight the phase-RBM positive-phase term.
        """
        G_am = torch.zeros(self.rbm_am.num_pars, dtype=DTYPE, device=self.device)
        G_ph = torch.zeros(self.rbm_ph.num_pars, dtype=DTYPE, device=self.device)

        if bases is None:
            G_am = self.rbm_am.effective_energy_gradient(samples)
            return [G_am, G_ph]

        try:
            bases_seq = [tuple(row) for row in bases]
        except Exception as e:
            raise ValueError("gradient: `bases` must be an iterable of string rows.") from e

        B = len(bases_seq)
        if B == 0:
            return [G_am, G_ph]
        n = len(bases_seq[0])
        if any(len(row) != n for row in bases_seq):
            raise ValueError("gradient: inconsistent basis widths.")
        if n != self.num_visible:
            raise ValueError(f"gradient: basis width {n} != num_visible {self.num_visible}.")
        if samples.shape[0] != B:
            raise ValueError(f"gradient: samples batch {samples.shape[0]} != bases rows {B}.")

        if samples.dim() < 2:
            samples = samples.unsqueeze(0)  # if B==1

        # Bucketize identical basis rows
        buckets = {}
        for i, row in enumerate(bases_seq):
            buckets.setdefault(row, []).append(i)

        for basis_t, idxs in buckets.items():
            idxs_t = torch.tensor(idxs, device=samples.device)
            has_non_z = any(ch != "Z" for ch in basis_t)
            if has_non_z:
                g_am, g_ph = self.rotated_gradient(basis_t, samples[idxs_t, :])
                G_am += g_am
                G_ph += (phase_weight * g_ph)
            else:
                G_am += self.rbm_am.effective_energy_gradient(samples[idxs_t, :])

        return [G_am, G_ph]

    def positive_phase_gradients(self, samples_batch, bases_batch=None, phase_weight: float = 1.0):
        grad = self.gradient(samples_batch, bases=bases_batch, phase_weight=phase_weight)
        return [g / float(samples_batch.shape[0]) for g in grad]

    # -------- Persistent chain utils (PCD) --------
    def _pcd_step(self, batch_size: int, k: int) -> torch.Tensor:
        """Advance (or initialize) a persistent chain and return the new visibles (vk)."""
        n = self.num_visible
        if (self._pcd_state is None) or (self._pcd_state.shape != (batch_size, n)):
            # initialize from Bernoulli(0.5)
            self._pcd_state = torch.bernoulli(0.5 * torch.ones(batch_size, n, device=self.device, dtype=DTYPE))
        # run k Gibbs steps in-place
        self._pcd_state = self.rbm_am.gibbs_steps(k, self._pcd_state, overwrite=True)
        return self._pcd_state

    # -------- Training loop (explicit) --------
    def fit(self, loader, epochs=100, k=10, lr=5e-3, log_every=5,
            optimizer=torch.optim.Adam, optimizer_args=None,
            target=None, bases=None, space=None,
            print_metrics=True, metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}",
            use_pcd: bool = False, phase_weight: float = 1.0,
            grad_clip_value: Optional[float] = 1.0):
        """
        Contrastive Divergence (CD/PCD) training using RBMTomographyLoader.
        The loader yields (pos_batch, neg_batch, bases_batch).

        Improvements:
        - Adam default optimizer (set weight_decay in optimizer_args for mild L2).
        - Optional PCD via `use_pcd=True` (persistent chains ignore loader.neg_batch).
        - Optional elementwise grad clipping before writing .grad buffers.
        - Optional `phase_weight` to temper noisy phase positive-phase gradients.
        """
        if self.stop_training:
            return {"epoch": []}
        optimizer_args = {} if optimizer_args is None else optimizer_args
        all_params = list(chain.from_iterable(getattr(self, n).parameters() for n in ["rbm_am", "rbm_ph"]))
        opt = optimizer(all_params, lr=lr, **optimizer_args)

        history = {"epoch": []}
        if target is not None:
            history["Fidelity"], history["KL"] = [], []
        if space is None:
            space = self.generate_hilbert_space()

        for ep in range(1, epochs + 1):
            for pos_batch, neg_batch, bases_batch in loader.iter_epoch():
                # positive phase (both nets)
                grads = self.positive_phase_gradients(pos_batch, bases_batch=bases_batch, phase_weight=phase_weight)

                # negative phase (amplitude RBM only)
                if use_pcd:
                    vk = self._pcd_step(batch_size=neg_batch.shape[0], k=k)
                else:
                    vk = self.rbm_am.gibbs_steps(k, neg_batch)
                grad_model = self.rbm_am.effective_energy_gradient(vk) / float(neg_batch.shape[0])
                grads[0] -= grad_model

                # optional elementwise clipping (helps stabilize explicit writes)
                if grad_clip_value is not None and grad_clip_value > 0:
                    grads = [g.clamp_(-grad_clip_value, grad_clip_value) for g in grads]

                # write grads and step
                opt.zero_grad()
                vector_to_grads(grads[0], self.rbm_am.parameters())
                vector_to_grads(grads[1], self.rbm_ph.parameters())
                opt.step()

                if self.stop_training:
                    break

            if (target is not None) and (ep % log_every == 0):
                with torch.no_grad():
                    fid_val = fidelity(self, target, space=space, bases=bases)
                    kl_val = KL(self, target, space=space, bases=bases)
                history["epoch"].append(ep)
                history.setdefault("Fidelity", []).append(fid_val)
                history.setdefault("KL", []).append(kl_val)
                if print_metrics:
                    print(metric_fmt.format(ep=ep, fid=fid_val, kl=kl_val))

            if self.stop_training:
                break
        return history


# -------------------------------
# Metrics
# -------------------------------
@torch.no_grad()
def fidelity(nn_state, target, space=None, **kwargs):
    """Squared overlap |<target|psi>|^2 of normalized states."""
    if not torch.is_complex(target):
        raise TypeError("fidelity: `target` must be complex (cdouble).")
    space = nn_state.generate_hilbert_space() if space is None else space
    psi = nn_state.psi_complex_normalized(space).reshape(-1).contiguous()
    tgt = target.to(device=nn_state.device, dtype=torch.cdouble).reshape(-1).contiguous()
    npsi = torch.linalg.vector_norm(psi)
    nt   = torch.linalg.vector_norm(tgt)
    if npsi == 0 or nt == 0:
        return 0.0
    psi_n = psi / npsi
    tgt_n = tgt / nt
    inner = (tgt_n.conj() * psi_n).sum()
    return float(inner.abs().pow(2).real)


@torch.no_grad()
def KL(nn_state, target, space=None, bases=None, **kwargs):
    """
    Average KL(p||q) across measurement bases.
    Target is normalized once; per-basis probabilities are renormalized.
    """
    if bases is None:
        raise ValueError("KL needs `bases`")
    if not torch.is_complex(target):
        raise TypeError("KL: `target` must be complex (cdouble).")

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
        psi_r     = rotate_psi(nn_state, basis, space, psi=psi_norm_cd)
        nn_probs_r  = (psi_r.abs().to(DTYPE)) ** 2
        tgt_probs_r = (tgt_psi_r.abs().to(DTYPE)) ** 2
        p_sum = tgt_probs_r.sum().clamp_min(eps)
        q_sum = nn_probs_r.sum().clamp_min(eps)
        p = (tgt_probs_r / p_sum).clamp_min(eps)
        q = (nn_probs_r / q_sum).clamp_min(eps)
        KL_val += torch.sum(p * (torch.log(p) - torch.log(q)))
    return (KL_val / len(bases)).item()


# -------------------------------
# Data: small dataset + guardrailed loader
# -------------------------------
class TomographyDataset:
    """
    Minimal dataset for tomography:
      - train_samples: (N, n) float64 on `device`
      - train_bases  : (N, n) array of {'X','Y','Z'} (NumPy, CPU)
      - target_state : (2^n,) complex (cdouble) on `device`
      - bases        : (M, n) array of bases for evaluation (NumPy, CPU)
    """
    def __init__(self, train_path, psi_path, train_bases_path, bases_path, device: torch.device = DEVICE):
        self.device = device

        # training samples (N, n)
        self.train_samples = torch.tensor(
            np.loadtxt(train_path, dtype="float32"),
            dtype=DTYPE, device=device
        )

        # target wavefunction (Re, Im) -> complex
        psi_np = np.loadtxt(psi_path, dtype="float64")
        self.target_state = torch.tensor(
            psi_np[:, 0] + 1j * psi_np[:, 1],
            dtype=torch.cdouble, device=device
        )

        # basis metadata (remain on CPU)
        self.train_bases = np.loadtxt(train_bases_path, dtype=str)
        self.bases = np.loadtxt(bases_path, dtype=str, ndmin=1)

        # Precompute Z-mask & indices (CPU tensors for indexing convenience)
        tb = np.asarray(self.train_bases)
        z_mask_np = (tb == "Z").all(axis=1)
        self._z_mask = torch.as_tensor(z_mask_np, dtype=torch.bool)
        self._z_indices = self._z_mask.nonzero(as_tuple=False).view(-1)

        # Basic guardrails
        if self.train_samples.shape[0] != len(self.train_bases):
            raise ValueError("TomographyDataset: sample count != basis row count.")

        # All rows should share width n
        widths = {len(row) for row in self.train_bases}
        if len(widths) != 1:
            raise ValueError("TomographyDataset: inconsistent basis widths.")
        n = next(iter(widths))
        if n != self.train_samples.shape[1]:
            raise ValueError("TomographyDataset: basis width != sample width.")

    def __len__(self):
        return int(self.train_samples.shape[0])

    def num_visible(self) -> int:
        return int(self.train_samples.shape[1])

    def z_indices(self) -> torch.Tensor:
        """Indices (CPU long) for Z-only rows."""
        return self._z_indices.clone()

    def train_bases_as_tuples(self) -> List[Tuple[str, ...]]:
        return [tuple(row) for row in np.asarray(self.train_bases)]

    def eval_bases(self) -> List[Tuple[str, ...]]:
        return [tuple(row) for row in np.asarray(self.bases)]

    def target(self) -> torch.Tensor:
        return self.target_state


class RBMTomographyLoader:
    """
    Purpose-built loader that yields (pos_batch, neg_batch, bases_batch) per epoch:
      - Positive samples are a one-pass shuffle of all rows.
      - Negative samples are drawn with replacement from the Z-only pool (for CD warm starts).
      - All device/dtype moves + invariants are handled here.
    Seeding:
      - If no generator is set, uses global torch RNG (so torch.manual_seed controls it).
      - Optionally call .set_seed(seed) to use a dedicated RNG stream.
    """
    def __init__(self, dataset: TomographyDataset,
                 pos_batch_size: int = 100, neg_batch_size: Optional[int] = None,
                 device: torch.device = DEVICE, dtype: torch.dtype = DTYPE, strict: bool = True):
        self.ds = dataset
        self.pos_bs = int(pos_batch_size)
        self.neg_bs = int(neg_batch_size or pos_batch_size)
        self.device = device
        self.dtype = dtype
        self.strict = strict
        self._gen: Optional[torch.Generator] = None  # independent RNG (optional)

        # upfront validation / shape invariants
        n = self.ds.num_visible()
        bases_seq = self.ds.train_bases_as_tuples()
        if any(len(row) != n for row in bases_seq):
            raise ValueError("RBMTomographyLoader: inconsistent basis widths in dataset.")
        if self.ds.z_indices().numel() == 0:
            raise ValueError("RBMTomographyLoader: Z-only pool is empty (need negatives).")

    def set_seed(self, seed: Optional[int]):
        """Set an independent RNG stream. If None, fall back to global torch RNG."""
        if seed is None:
            self._gen = None
        else:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            self._gen = g

    def __len__(self):
        return ceil(len(self.ds) / self.pos_bs)

    def iter_epoch(self):
        """Yield batches for a single epoch."""
        N = len(self.ds)
        n_batches = ceil(N / self.pos_bs)

        # One-pass shuffle for positives
        perm = torch.randperm(N, generator=self._gen) if self._gen is not None else torch.randperm(N)
        pos_samples = self.ds.train_samples[perm].to(self.device, dtype=self.dtype)

        bases_list = self.ds.train_bases_as_tuples()
        perm_idx = perm.detach().cpu().tolist()
        pos_bases_perm = [bases_list[i] for i in perm_idx]

        # Draw all negatives from Z-pool up-front (with replacement)
        z_pool = self.ds.z_indices()
        pool_len = z_pool.numel()
        if self._gen is None:
            neg_choices = torch.randint(pool_len, size=(n_batches * self.neg_bs,))
        else:
            neg_choices = torch.randint(pool_len, size=(n_batches * self.neg_bs,), generator=self._gen)
        neg_rows = z_pool[neg_choices]
        neg_samples_all = self.ds.train_samples[neg_rows].to(self.device, dtype=self.dtype)

        # Chunk and yield
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
                    raise RuntimeError("Loader invariant broken: bases_batch length mismatch.")
                if pos_batch.shape[1] != self.ds.num_visible():
                    raise RuntimeError("Loader invariant broken: pos_batch width != num_visible.")
                if neg_batch.shape[1] != self.ds.num_visible():
                    raise RuntimeError("Loader invariant broken: neg_batch width != num_visible.")

            yield pos_batch, neg_batch, bases_batch


# -------------------------------
# Simple training script example (aligned diagnostics)
# -------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt  # local import so the module stays light

    # keep seeding behavior identical to previous scripts
    torch.manual_seed(1234)

    train_path = "w_state_meas.txt"
    train_bases_path = "w_state_basis.txt"
    psi_path = "w_state_aug.txt"
    bases_path = "w_state_bases.txt"

    data = TomographyDataset(train_path, psi_path, train_bases_path, bases_path, device=DEVICE)

    U = create_dict()

    nv = data.num_visible()
    nh = nv
    nn_state = ComplexWaveFunction(nv, nh, U, device=DEVICE)

    # hyperparams (tuned to push past the ~0.9 plateau)
    epochs = 100
    pbs = 100
    nbs = 100
    lr = 5e-3
    k = 25                # longer chains help model negative phase
    log_every = 5

    loader = RBMTomographyLoader(data, pos_batch_size=pbs, neg_batch_size=nbs, device=DEVICE, dtype=DTYPE)
    # Optionally use a dedicated RNG stream for the loader:
    # loader.set_seed(1234)

    space = nn_state.generate_hilbert_space()
    history = nn_state.fit(
        loader,
        epochs=epochs,
        k=k,
        lr=lr,
        log_every=log_every,
        optimizer=torch.optim.Adam,
        optimizer_args=dict(betas=(0.9, 0.99), weight_decay=1e-6),
        target=data.target(),
        bases=data.eval_bases(),
        space=space,
        print_metrics=True,
        metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}",
        use_pcd=True,            # persistent CD (recommended)
        phase_weight=1.0,        # try 0.7 if phase noise is high
        grad_clip_value=1.0,     # elementwise clamp on explicit grads
    )

    # -------------------------------
    # Phase comparison (global phase aligned)
    # -------------------------------
    with torch.no_grad():
        space = nn_state.generate_hilbert_space()
        psi_m = nn_state.psi_complex_normalized(space).reshape(-1).contiguous()
        psi_t = data.target().to(device=DEVICE, dtype=torch.cdouble).reshape(-1).contiguous()

        # Normalize both
        nm = torch.linalg.vector_norm(psi_m)
        nt = torch.linalg.vector_norm(psi_t)
        if nm > 0:
            psi_m = psi_m / nm
        if nt > 0:
            psi_t = psi_t / nt

        # Global phase alignment via overlap
        ip = torch.sum(psi_t.conj() * psi_m)
        if ip.abs() > 1e-12:
            theta = torch.angle(ip)
        else:
            # Fallback: align by max-target component
            j = int(torch.argmax(psi_t.abs()))
            theta = torch.angle(psi_m[j]) - torch.angle(psi_t[j])
        psi_m_al = psi_m * torch.exp(-1j * theta)

        # Phases and wrapped phase error Δφ ∈ [-π, π]
        phi_t = torch.angle(psi_t).cpu().numpy()
        phi_m = torch.angle(psi_m_al).cpu().numpy()
        dphi = np.remainder((phi_m - phi_t) + np.pi, 2.0 * np.pi) - np.pi

        # Focus on informative support: top-99% target mass (cap 512 points)
        probs = (psi_t.abs() ** 2).cpu().numpy()
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        mass_cut = 0.99
        k_cap = 512
        k_sel = int(min(np.searchsorted(cum, mass_cut) + 1, k_cap, len(order)))
        sel = order[:k_sel]

        # Plot: target vs model phases (aligned)
        fig_p, axp = plt.subplots(figsize=(7.2, 3.8), dpi=150)
        axp.plot(range(k_sel), phi_t[sel], marker=".", linestyle="", label="target phase")
        axp.plot(range(k_sel), phi_m[sel], marker="x", linestyle="", label="model phase (aligned)")
        axp.set_xlabel("basis states (sorted by target mass)")
        axp.set_ylabel("phase [rad]")
        axp.set_title("Phase comparison — top 99% probability mass")
        axp.grid(True, alpha=0.3)
        axp.legend()
        fig_p.tight_layout()

        # Plot: wrapped phase error
        fig_e, axe = plt.subplots(figsize=(7.2, 3.8), dpi=150)
        axe.plot(range(k_sel), dphi[sel], marker=".", linestyle="", label="Δphase (wrapped)")
        axe.axhline(0.0, linewidth=1.0)
        axe.set_xlabel("basis states (sorted by target mass)")
        axe.set_ylabel("Δphase [rad] in [-π, π]")
        axe.set_title("Phase error (global phase aligned)")
        axe.grid(True, alpha=0.3)
        axe.legend()
        fig_e.tight_layout()

    # -------------------------------
    # Metrics plot: Fidelity (left) & KL (right)
    # -------------------------------
    ep = history.get("epoch", [])
    if ep and ("Fidelity" in history) and ("KL" in history):
        fig, ax1 = plt.subplots(figsize=(6.0, 4.0), dpi=140)
        ax2 = ax1.twinx()

        ax1.plot(ep, history["Fidelity"], marker="o", label="Fidelity")
        ax2.plot(ep, history["KL"], marker="s", linestyle="--", label="KL")

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel(r"$|\langle \psi_t \mid \psi \rangle|^2$")
        ax2.set_ylabel(r"KL$(p\,\|\,q)$")
        ax1.set_title("RBM Tomography — training metrics")
        ax1.grid(True, alpha=0.3)

        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="best")

        fig.tight_layout()
    else:
        print("No metrics to plot (did you pass target & bases and set log_every?).")

    # Show all figures
    plt.show()
