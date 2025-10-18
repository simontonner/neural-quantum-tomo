# RBM wavefunction - one-device refactor (ASCII, backward-compatible)
# -----------------------------------------------------------------------------
# Purpose & compatibility
# - Keep legacy 2-row public API [Re; Im] while doing all complex math in torch.cdouble.
# - Works on a single device end-to-end; no silent cpu<->cuda hops.
# - Adds human-readable aliases without breaking old names.
#
# Numerics & devices
# - RBM params/energies in float64 (DTYPE); complex paths in cdouble.
# - Bridge helpers: _to_cd (2-row/native -> cdouble), _from_cd (cdouble -> 2-row).
# - Never change device implicitly; always respect the input tensor’s .device or nn_state.device.
# - Inverse uses magnitude clamp to avoid blow-ups (eps).
# - Indices are rounded before .long() to kill 0.999999 -> 1 glitches.
#
# Rotations & unitaries
# - Do not build a full Kronecker matrix; apply factor-by-factor.
# - Unitary matrices are cached once in cdouble on the model (lazy).
# - rotate_psi can rotate a provided state or the model state.
# - Inner-product kernel enumerates branches and gathers <out|U|in> explicitly.
#
# Gradients & training
# - Two real RBMs model amplitude and phase; complex grads formed via mapping.
# - Rotated positive-phase gradients use branch weights; final grads are real.
# - Contrastive Divergence training with optional rotated-basis supervision.
# - Negative samples come from Z-basis rows when bases are provided.
# - Requires input_bases to avoid accidental unsupervised runs.
# - Optional scheduler, timing, early-stop flag; compact history logging.
#
# Metrics
# - Fidelity: normalized states; explicit real/imag sums (no torch.vdot).
# - KL: average over bases; rotate normalized complex state; clamp probs.
#
# Data & caching
# - generate_hilbert_space is cached by (size, device).
# - create_dict builds X/Y/Z in 2-row form; converted to cdouble once on first use.
#
# Safety guardrails
# - Max enumeration size to prevent accidental blow-ups.
# - Shape checks in Kronecker application.
# - No hidden dtype/device conversions; explicit and local only.
#
# API surface (legacy kept, aliases added)
# - Legacy: psi, psi_cd, psi_cd_normalized, psi_normalized, phase, fidelity, KL.
# - Aliases: wavefunction_2row, wavefunction_complex, wavefunction_complex_normalized,
#            wavefunction_normalized_2row, phase_angle, state_fidelity, average_kl_divergence.
#
# Plotting example
# - Example script uses ASCII labels (no LaTeX/Unicode), independent of core logic.
# -----------------------------------------------------------------------------

import time
from math import ceil
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.convert_parameters import _check_param_device
from torch.nn.utils import parameters_to_vector
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------------
# Device & dtypes
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double  # real-valued RBM parameters and energies in float64

# -------------------------------
# Minimal complex helpers (ASCII-only comments)
# Internally use torch.cdouble, keep 2-row [Re; Im] compatibility at the edges.
# -------------------------------

def numpy(x: torch.Tensor):
    """Detach -> CPU -> numpy complex. Accepts 2-row or native complex."""
    z = x if torch.is_complex(x) else _to_cd(x)
    return z.detach().to("cpu").numpy()


def _to_cd(x: torch.Tensor) -> torch.Tensor:
    """
    Bridge: accept 2-row [Re; Im] or native complex and return cdouble on x.device.
    Guardrail: never hop to global DEVICE here; respect x.device.
    """
    dev = x.device
    if torch.is_complex(x):
        return x.to(dtype=torch.cdouble, device=dev)
    # x is expected to be shape (2, ...): [Re; Im]
    return x[0].to(dtype=torch.cdouble, device=dev) + 1j * x[1].to(dtype=torch.cdouble, device=dev)


def _from_cd(z: torch.Tensor) -> torch.Tensor:
    """
    Bridge back to 2-row [Re; Im] in DTYPE on z.device.
    Guardrail: do not force global DEVICE here.
    """
    return torch.stack((z.real.to(DTYPE), z.imag.to(DTYPE)), dim=0)


def inverse(z: torch.Tensor, eps: float = 1e-6):
    """
    Numerically safe complex inverse.
    If `z` is native complex: return native complex; if 2-row: return 2-row.
    Guardrail: clamp magnitude in the denominator to avoid blow-up.
    """
    if torch.is_complex(z):
        zz = z
        return zz.conj() / (zz.abs().pow(2).clamp_min(eps))
    zz = _to_cd(z)
    invz = zz.conj() / (zz.abs().pow(2).clamp_min(eps))
    return _from_cd(invz)


# -------------------------------
# Unitaries and rotations
# -------------------------------

def create_dict(**kwargs):
    """
    Single-qubit unitaries stored in 2-row [Re; Im] format to match the legacy API.
    X and Y are normalized; Z is the identity.
    """
    dictionary = {
        "X": torch.tensor(
            [[[1.0, 1.0], [1.0, -1.0]], [[0.0, 0.0], [0.0, 0.0]]],
            dtype=DTYPE, device=DEVICE,
        ) / np.sqrt(2),
        "Y": torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0]], [[0.0, -1.0], [0.0, 1.0]]],
            dtype=DTYPE, device=DEVICE,
        ) / np.sqrt(2),
        "Z": torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
            dtype=DTYPE, device=DEVICE,
        ),
    }
    # Allow user-supplied unitaries (kept in 2-row format)
    dictionary.update(
        {
            name: (
                matrix.clone().detach()
                if isinstance(matrix, torch.Tensor)
                else torch.tensor(matrix)
            ).to(dtype=DTYPE, device=DEVICE)
            for name, matrix in kwargs.items()
        }
    )
    return dictionary


def _kron_mult(matrices, x):
    """
    Apply the product of factor unitaries (Kronecker structure) to a state vector
    without forming the full Kronecker matrix.

    Accepts 2-row or native complex for both inputs; returns the same complex-ness as `x`.
    Guardrails:
      - Convert each U once to cdouble.
      - Respect input dtype (return native complex if `x` was complex; else return 2-row).
      - Shape check ensures the product dimension matches the state length.
    """
    mats_cd = [_to_cd(m) for m in matrices]  # each (2,2) cdouble
    x_cd = _to_cd(x)                         # (L, ...)

    n = [m.size(-1) for m in mats_cd]        # typically [2,2,...]
    L, r = int(np.prod(n)), 1
    if L != x_cd.shape[0]:
        raise ValueError(f"Incompatible sizes: expected leading dim {L}, got {x_cd.shape[0]}")

    y = x_cd.clone()
    # Apply factors from right to left to respect strides
    for s in reversed(range(len(n))):
        L //= n[s]
        U = mats_cd[s]                       # (2,2) cdouble
        for k in range(L):
            for i in range(r):
                slc = slice(k * n[s] * r + i, (k + 1) * n[s] * r + i, r)
                tmp = y[slc, ...]            # (n[s], ...)
                y[slc, ...] = U @ tmp        # (2,2) @ (2, ...) -> (2, ...)
        r *= n[s]

    return y if torch.is_complex(x) else _from_cd(y)


def rotate_psi(nn_state, basis, space, unitaries=None, psi=None):
    """
    Rotate a wavefunction into `basis`. If `psi` is supplied (2-row or complex),
    rotate that; otherwise rotate nn_state's wavefunction. Returns native complex if input was complex.

    Guardrail: unitary dict is optionally cached in cdouble on nn_state for speed.
    """
    use_cdU = (unitaries is None) and (getattr(nn_state, "_unitary_dict_cd", None) is not None)
    Udict = nn_state._unitary_dict_cd if use_cdU else (unitaries if unitaries is not None else nn_state.unitary_dict)
    us = [Udict[b] for b in basis]

    dev = nn_state.device
    if psi is None:
        x = nn_state.psi_cd(space)  # complex wavefunction on device
    else:
        x = psi.to(dtype=torch.cdouble, device=dev) if torch.is_complex(psi) \
            else psi.to(device=dev, dtype=DTYPE)

    return _kron_mult(us, x)


# -------------------------------
# Rotation kernel for inner products (full-torch, branch enumeration)
# -------------------------------

def _rotate_basis_state(nn_state, basis, states, unitaries=None):
    """
    Compute branch weights and candidate states after measuring in `basis`.

    Returns:
      Ut : (C, B) torch.cdouble    product over sites of <out|U|in>
      v  : (C, B, n) DTYPE         candidate outcomes (rotated sites enumerated)

    Guardrails:
      - Build and cache unitaries in cdouble once (lazy).
      - Round indices before .long() to avoid floating drift (0.999999 -> 1).
      - Use gather to pick matrix elements with explicit (in, out) orientation.
    """
    device = nn_state.device
    unitaries = (unitaries if unitaries is not None else nn_state.unitary_dict)

    # Lazy cdouble cache on the model (avoids per-call conversions)
    if getattr(nn_state, "_unitary_dict_cd", None) is None:
        nn_state._unitary_dict_cd = {
            k: (v[0].to(dtype=torch.cdouble, device=device) + 1j * v[1].to(dtype=torch.cdouble, device=device))
            .detach().contiguous()
            for k, v in unitaries.items()
        }

    basis_arr = np.array(list(basis))
    sites = np.where(basis_arr != "Z")[0]    # only rotate non-Z sites

    if sites.size == 0:
        v = states.unsqueeze(0)  # (1, B, n)
        Ut = torch.ones(v.shape[:-1], dtype=torch.cdouble, device=device)  # (1, B)
        return Ut, v

    # fetch pre-converted cdouble unitaries
    Uc = torch.stack([nn_state._unitary_dict_cd[b] for b in basis_arr[sites]], dim=0)  # (S,2,2)
    Uio = Uc  # orientation U_{b,z} = <b|z>

    S = len(sites)
    B = states.shape[0]
    C = 2 ** S

    combos = nn_state.generate_hilbert_space(size=S, device=device)  # (C, S), DTYPE
    v = states.unsqueeze(0).repeat(C, 1, 1)                          # (C, B, n)
    v[:, :, sites] = combos.unsqueeze(1)                             # enumerate outcomes
    v = v.contiguous()

    # Round before int cast (eliminates 0.999999 -> 1 glitches)
    inp  = states[:, sites].round().long().T                         # (S, B)
    outp = v[:, :, sites].round().long().permute(0, 2, 1)            # (C, S, B)

    # Gather <out|U|in> per site and multiply over sites
    Uio_exp = Uio.unsqueeze(0).unsqueeze(2).expand(C, S, B, 2, 2)    # (C,S,B,2_in,2_out)
    inp_idx = inp.unsqueeze(0).expand(C, S, B).unsqueeze(-1).unsqueeze(-1)
    sel_in  = torch.gather(Uio_exp, dim=3, index=inp_idx.expand(C, S, B, 1, 2))
    out_idx = outp.unsqueeze(-1).unsqueeze(-1)
    sel_out = torch.gather(sel_in, dim=4, index=out_idx)

    Ut = sel_out.squeeze(-1).squeeze(-1).permute(0, 2, 1).prod(dim=-1)  # (C, B) cdouble
    return Ut, v


def _convert_basis_element_to_index(states):
    """
    Convert bit-rows (B, n) to flat indices (B,), most-significant bit first.
    Guardrail: .round() before .long() is crucial for numerical safety.
    """
    s = states.round().to(torch.long)
    n = s.shape[-1]
    shifts = torch.arange(n - 1, -1, -1, device=s.device, dtype=torch.long)
    return (s << shifts).sum(dim=-1)


def rotate_psi_inner_prod(nn_state, basis, states, unitaries=None, psi=None, include_extras=False):
    """
    Compute the inner product components for measuring the state in `basis`.

    Returns:
      - total_projected : (B,)  cdouble      total projected amplitude per sample
      - branch_amplitudes : (C,B) cdouble    branch amplitudes (if include_extras)
      - v : (C,B,n) DTYPE                    candidate states (if include_extras)

    Guardrails: input state can be provided or computed via nn_state; works for 2-row or complex.
    """
    Ut, v = _rotate_basis_state(nn_state, basis, states, unitaries=unitaries)   # (C,B) cdouble

    if psi is None:
        psi_sel = nn_state.psi_cd(v)                                            # (C,B) cdouble
    else:
        idx = _convert_basis_element_to_index(v).long()                         # (C,B)
        psi_sel = _to_cd(psi)[idx]                                              # (C,B) cdouble

    Upsi_v_c = Ut * psi_sel   # (C,B) branch amplitudes
    Upsi_c   = Upsi_v_c.sum(dim=0)  # (B,)

    if include_extras:
        return Upsi_c, Upsi_v_c, v
    return Upsi_c


# -------------------------------
# Data utils
# -------------------------------

def load_data(tr_samples_path, tr_psi_path=None, tr_bases_path=None, bases_path=None, complex_target: bool = False):
    """
    Loading helper.
    - train_samples: float64 on DEVICE
    - target state: either 2-row [Re; Im] (DTYPE) or native complex (cdouble) if complex_target=True
    - bases: strings
    """
    data = []
    data.append(torch.tensor(np.loadtxt(tr_samples_path, dtype="float32"), dtype=DTYPE, device=DEVICE))
    if tr_psi_path is not None:
        # expects columns [Re, Im] for the target state in the computational basis
        target_psi_data = np.loadtxt(tr_psi_path, dtype="float64")
        if complex_target:
            target_psi = torch.tensor(
                target_psi_data[:, 0] + 1j * target_psi_data[:, 1],
                dtype=torch.cdouble, device=DEVICE
            )
        else:
            target_psi = torch.zeros(2, len(target_psi_data), dtype=DTYPE, device=DEVICE)
            target_psi[0] = torch.tensor(target_psi_data[:, 0], dtype=DTYPE, device=DEVICE)
            target_psi[1] = torch.tensor(target_psi_data[:, 1], dtype=DTYPE, device=DEVICE)
        data.append(target_psi)
    if tr_bases_path is not None:
        data.append(np.loadtxt(tr_bases_path, dtype=str))
    if bases_path is not None:
        data.append(np.loadtxt(bases_path, dtype=str, ndmin=1))
    return data


def extract_refbasis_samples(train_samples, train_bases):
    """
    Pick rows where all measurement axes are 'Z'.
    Guardrail: do mask work in numpy, then bring back a torch.bool mask on the same device.
    """
    tb = np.asarray(train_bases)
    mask_np = (tb == "Z").all(axis=1)
    mask = torch.as_tensor(mask_np, device=train_samples.device, dtype=torch.bool)
    return train_samples[mask]


# -------------------------------
# Explicit gradient utils
# -------------------------------

def vector_to_grads(vec, parameters):
    """
    Write a flattened gradient vector back into a module's parameters.
    Guardrail: validate type and respect parameter device layout.
    """
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, but got: {torch.typename(vec)}")
    param_device = None
    pointer = 0
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        num_param = param.numel()
        param.grad = vec[pointer: pointer + num_param].view(param.size()).data
        pointer += num_param


# -------------------------------
# RBM (Bernoulli/Bernoulli) - no decorators, pure torch
# -------------------------------
class BinaryRBM(nn.Module):
    """
    Minimal Bernoulli/Bernoulli RBM used twice: amplitude and phase nets.
    Guardrails:
      - All math in DTYPE (float64) for stability.
      - Parameters have requires_grad=False since we inject grads manually.
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
        Guardrail: inline unsqueeze and return shape matches input batch rank.
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
        Gradients of E(v) with respect to parameters (positive phase).
        Guardrails:
          - reduce=False keeps per-sample grads (shape-broadcast safe).
          - We keep everything in DTYPE and convert to complex only where needed.
        """
        v = (v.unsqueeze(0) if v.dim() < 2 else v).to(self.weights)
        prob = self.prob_h_given_v(v)  # same batch rank
        if reduce:
            W_grad = -torch.matmul(prob.transpose(0, -1), v)
            vb_grad = -torch.sum(v, 0)
            hb_grad = -torch.sum(prob, 0)
            return parameters_to_vector([W_grad, vb_grad, hb_grad])
        else:
            W_grad = -torch.einsum("...j,...k->...jk", prob, v)
            vb_grad = -v
            hb_grad = -prob
            vec = [W_grad.view(*v.shape[:-1], -1), vb_grad, hb_grad]
            return torch.cat(vec, dim=-1)

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
        Guardrail: `overwrite=False` preserves the caller's tensor unless explicitly allowed.
        """
        v = (initial_state if overwrite else initial_state.clone()).to(self.weights)
        h = torch.zeros(*v.shape[:-1], self.num_hidden, device=self.device, dtype=DTYPE)
        for _ in range(k):
            self.sample_h_given_v(v, out=h)
            self.sample_v_given_h(h, out=v)
        return v

    def partition(self, space):
        """Z = sum_v exp(-E(v)) computed exactly over provided `space`."""
        return (-self.effective_energy(space)).logsumexp(0).exp()


# -------------------------------
# ComplexWaveFunction (amp + phase RBMs)
# -------------------------------
class ComplexWaveFunction:
    """
    Two real RBMs define magnitude and phase of the wavefunction over bitstrings:
      wf(v) = exp(-E_am(v)/2) * exp(+i * (-E_ph(v)/2))

    Guardrails:
      - wf_complex / wf_complex_normalized return native complex.
      - wf_2row / wf_normalized_2row return 2-row to keep the legacy surface API stable.
      - generate_hilbert_space caches by (size, device).
    """

    _rbm_am = None
    _rbm_ph = None
    _device = DEVICE
    _stop_training = False

    def __init__(self, num_visible, num_hidden=None, unitary_dict=None, module=None):
        self.device = DEVICE
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

        self.unitary_dict = unitary_dict if unitary_dict is not None else create_dict()
        self.unitary_dict = {k: v.to(self.device) for k, v in self.unitary_dict.items()}

        self._unitary_dict_cd = None  # lazy-built cache of cdouble unitaries
        self._hilbert_cache = {}

    # basic props
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
        return 20

    @property
    def networks(self):
        return ["rbm_am", "rbm_ph"]

    @property
    def rbm_am(self):
        return self._rbm_am

    @rbm_am.setter
    def rbm_am(self, new_val):
        self._rbm_am = new_val

    @property
    def rbm_ph(self):
        return self._rbm_ph

    @rbm_ph.setter
    def rbm_ph(self, new_val):
        self._rbm_ph = new_val

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, new_val):
        self._device = new_val

    def __getattr__(self, attr):
        # delegate unknown attributes to amplitude RBM for convenience
        return getattr(self.rbm_am, attr)

    # --- amplitudes, phases, and wf ---
    def reinitialize_parameters(self):
        for net in self.networks:
            getattr(self, net).initialize_parameters()

    # Original API (kept intact)
    def amplitude(self, v):
        v = v.to(self.device, dtype=DTYPE)
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()

    def phase(self, v):
        v = v.to(self.device, dtype=DTYPE)
        return -0.5 * self.rbm_ph.effective_energy(v)

    def psi(self, v):
        """2-row wavefunction (legacy surface)."""
        amp = (-self.rbm_am.effective_energy(v.to(self.device, dtype=DTYPE))).exp().sqrt()
        ph = -0.5 * self.rbm_ph.effective_energy(v.to(self.device, dtype=DTYPE))
        psi_cd = amp.to(torch.cdouble) * torch.exp(1j * ph.to(torch.cdouble))
        return _from_cd(psi_cd)

    def psi_cd(self, v):
        """Native complex wavefunction."""
        v = v.to(self.device, dtype=DTYPE)
        amp = (-self.rbm_am.effective_energy(v)).exp().sqrt()
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        return amp.to(torch.cdouble) * torch.exp(1j * ph.to(torch.cdouble))

    def psi_cd_normalized(self, v):
        """
        Native complex, normalized wavefunction.
        Guardrail: compute in log-domain for amplitude and normalize via logsumexp.
        """
        v = v.to(self.device, dtype=DTYPE)
        E = self.rbm_am.effective_energy(v)
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        logZ = torch.logsumexp(-E, dim=0)
        return torch.exp(((-0.5 * E) - 0.5 * logZ).to(torch.cdouble) + 1j * ph.to(torch.cdouble))

    def psi_normalized(self, v):
        """2-row normalized wavefunction (wrapper over complex-normalized)."""
        return _from_cd(self.psi_cd_normalized(v))

    # Clearer aliases (preferred in new code)
    def wavefunction_2row(self, v):
        return self.psi(v)

    def wavefunction_complex(self, v):
        return self.psi_cd(v)

    def wavefunction_complex_normalized(self, v):
        return self.psi_cd_normalized(v)

    def wavefunction_normalized_2row(self, v):
        return self.psi_normalized(v)

    def phase_angle(self, v):
        return self.phase(v)

    # --- probabilities and utilities ---
    def probability(self, v, Z=1.0):
        v = v.to(device=self.device, dtype=DTYPE)
        return (-self.rbm_am.effective_energy(v)).exp() / Z

    def normalization(self, space):
        return self.rbm_am.partition(space)

    def generate_hilbert_space(self, size=None, device=None):
        """
        Enumerate computational basis as a bit-matrix of shape (2^size, size).
        Guardrails:
          - Cache by (size, device.type, device.index).
          - Error if size exceeds a safe maximum (avoid accidental blow-ups).
        """
        device = self.device if device is None else device
        size = self.rbm_am.num_visible if size is None else size
        if size > self.max_size:
            raise ValueError("Hilbert space too large!")
        key = (size, device.type, device.index)
        if key in self._hilbert_cache:
            return self._hilbert_cache[key]
        n = 2 ** size
        ar = torch.arange(n, device=device, dtype=torch.long)
        shifts = torch.arange(size - 1, -1, -1, device=device, dtype=torch.long)
        bits = ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)
        self._hilbert_cache[key] = bits
        return bits

    def sample(self, k, num_samples=1, initial_state=None, overwrite=False):
        if initial_state is None:
            dist = torch.distributions.Bernoulli(probs=0.5)
            shape = torch.Size((num_samples, self.num_visible))
            initial_state = dist.sample(shape).to(self.device, dtype=DTYPE)
        return self.rbm_am.gibbs_steps(k, initial_state, overwrite=overwrite)

    # --- gradients (complex via mapping; final grads are real DTYPE) ---
    def am_grads(self, v):
        g = self.rbm_am.effective_energy_gradient(v, reduce=False)  # (..., G) real
        return g.to(torch.cdouble)

    def ph_grads(self, v):
        g = self.rbm_ph.effective_energy_gradient(v, reduce=False)  # (..., G) real
        return (1j * g.to(torch.cdouble))  # i * g matches d(phase)

    def rotated_gradient(self, basis, sample):
        """
        Gradient of log wavefunction under a rotated measurement basis.
        Shapes:
          total_projected : (B,)       complex
          branch weights  : (C,B)      for enumeration
          am/ph grads     : (C,B,G)
        Guardrails:
          - Use complex branch weights directly (no extra conversions).
          - Multiply by inverse total amplitude per batch element.
          - Final grads are real (imag_part=False legacy behavior).
        """
        Upsi, Upsi_v, v = rotate_psi_inner_prod(self, basis, sample, include_extras=True)
        inv_Upsi = inverse(Upsi)  # complex (B,)

        raw_grads = [self.am_grads(v), self.ph_grads(v)]  # both complex
        rotated_grad = [torch.einsum("cb,cbg->bg", Upsi_v, g) for g in raw_grads]
        grad = [torch.einsum("b,bg->g", inv_Upsi, rg).real.to(DTYPE) for rg in rotated_grad]
        return grad

    def gradient(self, samples, bases=None):
        grad = [torch.zeros(getattr(self, net).num_pars, dtype=DTYPE, device=self.device)
                for net in self.networks]
        if bases is None:
            grad[0] = self.rbm_am.effective_energy_gradient(samples)
            return grad
        if samples.dim() < 2:
            samples = samples.unsqueeze(0)
            bases = np.array(list(bases)).reshape(1, -1)

        # Group identical basis-rows to avoid repeated rotation work
        unique_bases, indices = np.unique(bases, axis=0, return_inverse=True)
        indices = torch.tensor(indices, device=samples.device)
        for i in range(unique_bases.shape[0]):
            basis = unique_bases[i, :]
            if np.any(basis != "Z"):
                g_am, g_ph = self.rotated_gradient(basis, samples[indices == i, :])
                grad[0] += g_am
                grad[1] += g_ph
            else:
                grad[0] += self.rbm_am.effective_energy_gradient(samples[indices == i, :])
        return grad

    # --- CD training plumbing ---
    def positive_phase_gradients(self, samples_batch, bases_batch=None):
        grad = self.gradient(samples_batch, bases=bases_batch)
        return [g / float(samples_batch.shape[0]) for g in grad]

    def compute_batch_gradients(self, k, samples_batch, neg_batch, bases_batch=None):
        grad = self.positive_phase_gradients(samples_batch, bases_batch=bases_batch)
        vk = self.rbm_am.gibbs_steps(k, neg_batch)
        grad_model = self.rbm_am.effective_energy_gradient(vk)
        grad[0] -= grad_model / float(neg_batch.shape[0])
        return grad

    # --- training loop ---
    def _shuffle_data(self, pos_batch_size, neg_batch_size, num_batches, train_samples, input_bases, z_samples):
        """
        Create positive and negative batches. If bases are provided, negative samples
        are drawn from Z-basis rows (reference-basis subset).
        Guardrail: reuse permutations to reduce RNG bias when batch sizes match.
        """
        pos_perm = torch.randperm(train_samples.shape[0], device=self.device)
        pos_samples = train_samples[pos_perm]
        if input_bases is None:
            if neg_batch_size == pos_batch_size:
                neg_perm = pos_perm
            else:
                neg_perm = torch.randint(train_samples.shape[0], size=(num_batches * neg_batch_size,),
                                         dtype=torch.long, device=self.device)
            neg_samples = train_samples[neg_perm]
        else:
            neg_perm = torch.randint(z_samples.shape[0], size=(num_batches * neg_batch_size,),
                                     dtype=torch.long, device=self.device)
            neg_samples = z_samples[neg_perm]
        pos_batches = [pos_samples[i:i + pos_batch_size] for i in range(0, len(pos_samples), pos_batch_size)]
        neg_batches = [neg_samples[i:i + neg_batch_size] for i in range(0, len(neg_samples), neg_batch_size)]
        if input_bases is not None:
            pos_bases = np.asarray(input_bases)[pos_perm.cpu().numpy()]
            pos_bases_batches = [pos_bases[i:i + pos_batch_size] for i in range(0, len(train_samples), pos_batch_size)]
            return zip(pos_batches, neg_batches, pos_bases_batches)
        else:
            return zip(pos_batches, neg_batches)

    def fit(self, data, epochs=100, pos_batch_size=100, neg_batch_size=None, k=1, lr=1e-3,
            input_bases=None, log_every=5, progbar=True, starting_epoch=1,
            optimizer=torch.optim.SGD, optimizer_args=None, scheduler=None,
            scheduler_args=None, target=None, bases=None, space=None, timeit=False,
            print_metrics=True, metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}"):
        """
        Contrastive Divergence training with optional rotated-basis positive phase.
        Guardrails:
          - Do not proceed without input_bases (enforces explicit supervision intent).
          - Keep timing optional and lightweight.
          - History contains only logged epochs to reduce memory.
        """
        if input_bases is None:
            raise ValueError("input_bases must be provided to train ComplexWaveFunction!")
        if self.stop_training:
            return {"epoch": []}

        neg_batch_size = neg_batch_size or pos_batch_size
        optimizer_args = {} if optimizer_args is None else optimizer_args
        scheduler_args = {} if scheduler_args is None else scheduler_args

        train_samples = data.clone().detach().to(self.device, dtype=DTYPE) if isinstance(data, torch.Tensor) \
            else torch.tensor(data, device=self.device, dtype=DTYPE)
        z_samples = extract_refbasis_samples(train_samples, input_bases).to(self.device)

        all_params = list(chain.from_iterable(getattr(self, n).parameters() for n in self.networks))
        opt = optimizer(all_params, lr=lr, **optimizer_args)
        sch = scheduler(opt, **scheduler_args) if scheduler is not None else None

        history = {"epoch": []}
        want_metrics = target is not None
        if want_metrics:
            history["Fidelity"], history["KL"] = [], []
        if timeit:
            history["TimeSec"] = []
        if space is None:
            space = self.generate_hilbert_space()

        num_batches = ceil(train_samples.shape[0] / pos_batch_size)
        epoch_iter = tqdm(range(starting_epoch, epochs + 1), desc="Epochs", disable=not progbar)

        for ep in epoch_iter:
            t0 = time.time() if timeit else None
            data_iter = self._shuffle_data(pos_batch_size, neg_batch_size, num_batches,
                                           train_samples, input_bases, z_samples)
            for _, batch in enumerate(data_iter):
                grads = self.compute_batch_gradients(k, *batch)
                opt.zero_grad()
                for i, net in enumerate(self.networks):
                    rbm = getattr(self, net)
                    vector_to_grads(grads[i], rbm.parameters())
                opt.step()
                if self.stop_training:
                    break
            if sch is not None:
                sch.step()

            if want_metrics and (ep % log_every == 0):
                with torch.no_grad():
                    fid_val = fidelity(self, target, space=space, bases=bases)
                    kl_val = KL(self, target, space=space, bases=bases)
                history["epoch"].append(ep)
                history["Fidelity"].append(fid_val)
                history["KL"].append(kl_val)
                if progbar:
                    epoch_iter.set_postfix(Fidelity=f"{fid_val:.4f}", KL=f"{kl_val:.4f}")
                if print_metrics:
                    print(metric_fmt.format(ep=ep, fid=fid_val, kl=kl_val))
            if timeit:
                history.setdefault("TimeSec", []).append(time.time() - t0)
            if self.stop_training:
                break
        return history


# -------------------------------
# Metrics (complex-normalized wavefunction throughout)
# -------------------------------

def fidelity(nn_state, target, space=None, **kwargs):
    """
    Squared overlap of normalized states: abs(<target, model>)^2.
    Accepts `target` as 2-row or native complex.
    Guardrails:
      - Normalize both states explicitly to remove dependence on overall scale.
      - Avoid torch.vdot; compute real and imaginary sums explicitly for clarity.
    """
    space = nn_state.generate_hilbert_space() if space is None else space

    psi = nn_state.psi_cd_normalized(space).reshape(-1).contiguous()   # cdouble
    tgt = _to_cd(target.to(nn_state.device)).reshape(-1).contiguous()  # cdouble

    npsi = torch.linalg.vector_norm(psi)
    nt = torch.linalg.vector_norm(tgt)
    if npsi == 0 or nt == 0:
        return 0.0
    psi_n = psi / npsi
    tgt_n = tgt / nt

    # <tgt, psi> = sum conj(tgt) * psi -> explicit real/imag form
    re = (tgt_n.real * psi_n.real + tgt_n.imag * psi_n.imag).sum()
    im = (tgt_n.real * psi_n.imag - tgt_n.imag * psi_n.real).sum()
    return float((re * re + im * im).real)


def _single_basis_KL(target_probs, nn_probs, eps=1e-12):
    """
    KL(p||q) for a single basis.
    Guardrail: clamp probs to avoid log(0).
    """
    p = target_probs.clamp_min(eps)
    q = nn_probs.clamp_min(eps)
    return torch.sum(p * (torch.log(p) - torch.log(q)))


def KL(nn_state, target, space=None, bases=None, **kwargs):
    """
    Average KL across provided measurement bases.
    Guardrails:
      - Rotate normalized complex wavefunction; unitaries preserve norm.
      - Accept target as 2-row or complex, convert once.
    """
    space = nn_state.generate_hilbert_space() if space is None else space
    tgt = target.to(nn_state.device)

    KL_val = 0.0
    psi_norm_cd = nn_state.psi_cd_normalized(space)  # complex normalized wf
    tgt_cd = _to_cd(tgt)                              # complex target

    for basis in bases:
        tgt_psi_r = rotate_psi(nn_state, basis, space, psi=tgt_cd)       # complex
        psi_r = rotate_psi(nn_state, basis, space, psi=psi_norm_cd)      # complex

        nn_probs_r = (psi_r.abs().to(DTYPE)) ** 2
        tgt_probs_r = (tgt_psi_r.abs().to(DTYPE)) ** 2
        KL_val += _single_basis_KL(tgt_probs_r, nn_probs_r)

    return (KL_val / len(bases)).item()


# User-friendly aliases for metrics

def state_fidelity(nn_state, target, **kwargs):
    return fidelity(nn_state, target, **kwargs)


def average_kl_divergence(nn_state, target, **kwargs):
    return KL(nn_state, target, **kwargs)


# -------------------------------
# Training script (example)
# -------------------------------
if __name__ == "__main__":
    train_path = "w_state_meas.txt"
    train_bases_path = "w_state_basis.txt"
    psi_path = "w_state_aug.txt"
    bases_path = "w_state_bases.txt"

    # Ingest the target as native complex (robust downstream; avoids format ambiguity)
    train_samples, true_state, train_bases, bases = load_data(
        train_path, psi_path, train_bases_path, bases_path, complex_target=True
    )

    # Derive one-hot indices (1,2,4,...) for visualization of phases.
    true_state_c = _to_cd(true_state)
    N = true_state_c.numel()
    one_hot_indices = [2 ** i for i in range(N.bit_length() - 1)]
    one_hot_true_state = true_state_c[one_hot_indices]

    # Plot helper (does not affect training)
    true_phases_raw = torch.angle(one_hot_true_state)
    true_phases_wrapped = (true_phases_raw - true_phases_raw[0]) % (2 * np.pi)

    torch.manual_seed(1234)
    unitary_dict = create_dict()

    nv = train_samples.shape[-1]
    nh = nv
    nn_state = ComplexWaveFunction(nv, nh, unitary_dict)

    # Typical CD-1x settings for small systems (tune as needed)
    epochs = 70
    pbs = 100
    nbs = 100
    lr = 1e-1
    k = 10
    log_every = 5
    space = nn_state.generate_hilbert_space()

    history = nn_state.fit(
        train_samples, epochs=epochs, pos_batch_size=pbs,
        neg_batch_size=nbs, lr=lr, k=k, input_bases=train_bases,
        progbar=True, log_every=log_every, target=true_state,
        bases=bases, space=space, timeit=True, print_metrics=True,
        metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}"
    )

    # Plot: predicted vs true phases on one-hot basis states
    fidelities = np.array(history.get("Fidelity", []))
    KLs = np.array(history.get("KL", []))
    epoch = np.array(history.get("epoch", []))

    full_hs = nn_state.generate_hilbert_space()
    one_hot_hs = full_hs[one_hot_indices, :]
    pred_phases_raw = nn_state.phase(one_hot_hs)
    pred_phases_wrapped = (pred_phases_raw - pred_phases_raw[0]) % (2 * np.pi)

    plt.rcParams.update({"font.family": "serif"})
    bitstrings = ["".join(str(int(b)) for b in row) for row in one_hot_hs.cpu().numpy()]
    indices = np.arange(len(pred_phases_wrapped))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.set_facecolor("white")
    ax.bar(indices - width / 2, true_phases_wrapped, width, alpha=0.7,
           label="phase_true", zorder=1)
    ax.bar(indices + width / 2, pred_phases_wrapped, width, alpha=0.7,
           label="phase_predicted", zorder=2)
    ax.set_xlabel("Basis state", fontsize=14)
    ax.set_ylabel("Phase (radians)", fontsize=14)
    ax.set_title("Phase comparison: phase-augmented W state", fontsize=16)
    ax.set_xticks(indices)
    ax.set_xticklabels(bitstrings, rotation=45)
    ax.set_ylim(0, 2 * np.pi + 0.2)
    ax.legend(frameon=True, framealpha=1, loc="best", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()
