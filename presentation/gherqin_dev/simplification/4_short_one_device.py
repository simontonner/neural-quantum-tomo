# One-device refactor — stable & commented (hybrid 2-row complex API)

import time
from math import ceil
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.convert_parameters import _check_param_device
from torch.nn.utils import parameters_to_vector
from torch.distributions.utils import probs_to_logits
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------------
# Device & dtypes
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double  # real computations & RBM params in float64

# -------------------------------
# Minimal complex arithmetic (public API unchanged)
# Represent complex tensors as (2, ...) with rows [real; imag]
# We compute *internally* using torch.cdouble, but expose the 2-row format.
# -------------------------------
I = torch.tensor([0.0, 1.0], dtype=DTYPE, device=DEVICE)  # imag unit in 2-row format via scalar_mult

def make_complex(x, y=None):
    """Create 2-row complex from numpy or torch. Keep everything on DEVICE/DTYPE."""
    if isinstance(x, np.ndarray):
        return make_complex(torch.tensor(x.real, dtype=DTYPE, device=DEVICE),
                            torch.tensor(x.imag, dtype=DTYPE, device=DEVICE)).contiguous()
    if isinstance(x, torch.Tensor):
        x = x.to(device=DEVICE, dtype=DTYPE)
    if y is None:
        y = torch.zeros_like(x)
    else:
        y = y.to(device=DEVICE, dtype=DTYPE)
    return torch.cat((x.unsqueeze(0), y.unsqueeze(0)), dim=0)

def real(x): return x[0, ...]
def imag(x): return x[1, ...]
def conj(x): return make_complex(real(x), -imag(x))

def numpy(x):
    """Detach → CPU → numpy complex for quick inspection if ever needed."""
    return real(x).detach().cpu().numpy() + 1j * imag(x).detach().cpu().numpy()

# ---- cdouble bridge helpers (internal only) ----
def _to_cd(x: torch.Tensor) -> torch.Tensor:
    """2-row → cdouble tensor on DEVICE."""
    return real(x).to(dtype=torch.cdouble, device=DEVICE) + 1j * imag(x).to(dtype=torch.cdouble, device=DEVICE)

def _from_cd(z: torch.Tensor) -> torch.Tensor:
    """cdouble → 2-row DTYPE."""
    return make_complex(z.real.to(DTYPE), z.imag.to(DTYPE))

# ---- complex ops (compute in cdouble; return 2-row) ----
def scalar_mult(x, y, out=None):
    z = _to_cd(x) * _to_cd(y.to(x))
    res = _from_cd(z)
    if out is not None:
        if out is x or out is y:
            raise RuntimeError("Can't overwrite an argument!")
        out.copy_(res)
        return out
    return res

def matmul(x, y):
    """
    Complex matmul with 2-row convention:
      x: (2, m, n)
      y: (2, n, ...)   (leading '2' is complex channel)
    """
    X = _to_cd(x)  # (m, n)
    Y = _to_cd(y)  # (n, ...)
    Z = torch.einsum('ab,b...->a...', X, Y)  # (m, ...)
    return _from_cd(Z)

def inner_prod(x, y):
    """Conjugate inner product ⟨x|y⟩, returned in 2-row format."""
    zx = _to_cd(x).reshape(-1)
    zy = _to_cd(y).reshape(-1)
    ov = torch.vdot(zx, zy)  # conj(x)·y
    return _from_cd(ov)

def einsum(equation, a, b, real_part=True, imag_part=True):
    za = _to_cd(a)
    zb = _to_cd(b)
    z = torch.einsum(equation, za, zb)
    if real_part and imag_part:
        return _from_cd(z)
    elif real_part:
        return z.real.to(DTYPE)
    elif imag_part:
        return z.imag.to(DTYPE)
    else:
        return None

def absolute_value(x):
    """Return |x| in DTYPE (x is 2-row)."""
    return _to_cd(x).abs().to(DTYPE)

def inverse(z, eps: float = 1e-6):
    """
    Numerically *safe* complex inverse in 2-row form.
    Important change vs. before: eps raised from 1e-12 → 1e-6.
    Reason: when Uψ ≈ 0 (common early in training with exact rotations),
    1e-12 yields huge steps; 1e-6 keeps gradients bounded.
    """
    zz = _to_cd(z)
    invz = zz.conj() / (zz.abs().pow(2).clamp_min(eps))
    return _from_cd(invz)

# -------------------------------
# Unitaries & rotations
# -------------------------------
def create_dict(**kwargs):
    """
    Store single-qubit unitaries in *2-row* format (Re/Im) to match the public API.
    We build X,Y,Z in re/im blocks (so they can be used with the old code paths).
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
    Apply ⊗_s U_s onto ψ (all in 2-row format). We just wrap the cdouble matmul
    through `matmul`, keeping public shapes unchanged.
    """
    n = [m.size()[1] for m in matrices]  # matrix is (2,2,2): channel, row, col
    l, r = np.prod(n), 1
    if l != x.shape[1]:
        raise ValueError("Incompatible sizes!")
    y = x.clone()
    for s in reversed(range(len(n))):
        l //= n[s]
        m = matrices[s]
        for k in range(l):
            for i in range(r):
                slc = slice(k * n[s] * r + i, (k + 1) * n[s] * r + i, r)
                temp = y[:, slc, ...]          # (2, n[s], ...)
                y[:, slc, ...] = matmul(m, temp)
        r *= n[s]
    return y

def rotate_psi(nn_state, basis, space, unitaries=None, psi=None):
    """
    Rotate ψ into a product basis using the 2-row API.
    """
    psi = nn_state.psi(space) if psi is None else psi.to(dtype=DTYPE, device=DEVICE)
    unitaries = unitaries if unitaries else nn_state.unitary_dict
    unitaries = {k: v.to(device=DEVICE) for k, v in unitaries.items()}
    us = [unitaries[b] for b in basis]
    return _kron_mult(us, psi)

# -------------------------------
# FULL-TORCH rotation kernel (no NumPy round-trips)
# -------------------------------
def _rotate_basis_state(nn_state, basis, states, unitaries=None):
    """
    Compute branch weights and candidate states after measuring in `basis`.

    Returns:
      Ut : (C, B) torch.cdouble = product of single-site matrix elements <out|U|in>
      v  : (C, B, n) DTYPE       = candidate outcomes (rotated sites enumerated)

    Two **numerical** precautions here:
    1) We construct U in cdouble (from 2-row re/im) exactly once.
    2) We **round** indices before .long() to avoid 0.999999 → 1 glitches.
    """
    device = nn_state.device
    unitaries = (unitaries if unitaries is not None else nn_state.unitary_dict)

    basis_arr = np.array(list(basis))
    sites = np.where(basis_arr != "Z")[0]

    if sites.size == 0:
        v = states.unsqueeze(0)  # (1, B, n)
        Ut = torch.ones(v.shape[:-1], dtype=torch.cdouble, device=device)  # (1, B)
        return Ut, v

    # Re/Im → cdouble unitaries
    Us = torch.stack([unitaries[b] for b in basis_arr[sites]], dim=0).to(device=device)  # (S,2,2,2)
    Uc = Us[:, 0, ...].to(torch.cdouble) + 1j * Us[:, 1, ...]                            # (S,2,2)
    Uio = Uc  # correct orientation: U_{b,z} = <b|z>

    S = len(sites)
    B = states.shape[0]
    C = 2 ** S

    combos = nn_state.generate_hilbert_space(size=S, device=device)  # (C, S), DTYPE
    v = states.unsqueeze(0).repeat(C, 1, 1)                          # (C, B, n)
    v[:, :, sites] = combos.unsqueeze(1)                             # enumerate outcomes
    v = v.contiguous()

    # ---- IMPORTANT: round before casting to int to kill tiny drift ----
    inp  = states[:, sites].round().long().T                         # (S, B)
    outp = v[:, :, sites].round().long().permute(0, 2, 1)            # (C, S, B)

    # Gather <out|U|in> for each site, then multiply over sites
    Uio_exp = Uio.unsqueeze(0).unsqueeze(2).expand(C, S, B, 2, 2)    # (C,S,B,2_in,2_out)
    inp_idx = inp.unsqueeze(0).expand(C, S, B).unsqueeze(-1).unsqueeze(-1)
    sel_in  = torch.gather(Uio_exp, dim=3, index=inp_idx.expand(C, S, B, 1, 2))
    out_idx = outp.unsqueeze(-1).unsqueeze(-1)
    sel_out = torch.gather(sel_in, dim=4, index=out_idx)

    Ut = sel_out.squeeze(-1).squeeze(-1).permute(0, 2, 1).prod(dim=-1)  # (C, B) cdouble
    return Ut, v

def _convert_basis_element_to_index(states):
    """Convert bit-rows (B,n) to flat indices (B,) respecting MSB-first convention."""
    powers = (2 ** (torch.arange(states.shape[-1], 0, -1, device=DEVICE) - 1)).to(states)
    return torch.matmul(states, powers)

def rotate_psi_inner_prod(nn_state, basis, states, unitaries=None, psi=None, include_extras=False):
    """
    ⟨basis|ψ⟩ over all outcomes (vectorized). Returns 2-row complex.
    """
    Ut, v = _rotate_basis_state(nn_state, basis, states, unitaries=unitaries)   # Ut: (C,B) cdouble

    if psi is None:
        # nn_state.psi(v) gives (2, C, B) → convert to cdouble
        psi_sel = _to_cd(nn_state.psi(v))                                       # (C, B)
    else:
        # psi is (2, N) on computational basis; pick entries for v
        idx = _convert_basis_element_to_index(v).long()                          # (C, B)
        psi_sel = _to_cd(psi)[idx]                                              # (C, B)

    Upsi_v_c = Ut * psi_sel                 # (C, B) cdouble
    Upsi_c   = Upsi_v_c.sum(dim=0)          # (B,)   cdouble

    Upsi     = _from_cd(Upsi_c)             # (2, B) 2-row
    Upsi_v   = _from_cd(Upsi_v_c)           # (2, C, B)

    return (Upsi, Upsi_v, v) if include_extras else Upsi

# -------------------------------
# Data utils
# -------------------------------
def load_data(tr_samples_path, tr_psi_path=None, tr_bases_path=None, bases_path=None):
    data = []
    data.append(torch.tensor(np.loadtxt(tr_samples_path, dtype="float32"), dtype=DTYPE, device=DEVICE))
    if tr_psi_path is not None:
        # expects columns [Re, Im] for ψ in computational basis
        target_psi_data = np.loadtxt(tr_psi_path, dtype="float64")
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
    Robust mask for rows where all measurement axes are 'Z'.
    Accepts numpy / list / torch; avoids .astype on tensors.
    """
    tb = np.asarray(train_bases)
    mask_np = (tb == "Z").all(axis=1)
    mask = torch.as_tensor(mask_np, device=train_samples.device, dtype=torch.bool)
    return train_samples[mask]

# -------------------------------
# Explicit gradient utils
# -------------------------------
def vector_to_grads(vec, parameters):
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
# RBM (Bernoulli/Bernoulli) — NO DECORATORS
# -------------------------------
class BinaryRBM(nn.Module):
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
        """Inline unsqueeze; all math in float64 for stability."""
        unsq = False
        if v.dim() < 2:
            v = v.unsqueeze(0); unsq = True
        v = v.to(self.weights)
        visible_bias_term = torch.matmul(v, self.visible_bias)
        hid_bias_term = F.softplus(F.linear(v, self.weights, self.hidden_bias)).sum(-1)
        out = -(visible_bias_term + hid_bias_term)
        return out.squeeze(0) if unsq else out

    def effective_energy_gradient(self, v, reduce=True):
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
            h = h.unsqueeze(0); unsq = True
        res = torch.matmul(h, self.weights.data).add_(self.visible_bias.data).sigmoid_().clamp_(0, 1)
        if out is not None:
            out.copy_(res.squeeze(0) if unsq and out.dim() == 1 else res); return out
        return res.squeeze(0) if unsq else res

    def prob_h_given_v(self, v, out=None):
        unsq = False
        if v.dim() < 2:
            v = v.unsqueeze(0); unsq = True
        res = torch.matmul(v, self.weights.data.t()).add_(self.hidden_bias.data).sigmoid_().clamp_(0, 1)
        if out is not None:
            out.copy_(res.squeeze(0) if unsq and out.dim() == 1 else res); return out
        return res.squeeze(0) if unsq else res

    def sample_v_given_h(self, h, out=None):
        probs = self.prob_v_given_h(h)
        return torch.bernoulli(probs, out=out) if out is not None else torch.bernoulli(probs)

    def sample_h_given_v(self, v, out=None):
        probs = self.prob_h_given_v(v)
        return torch.bernoulli(probs, out=out) if out is not None else torch.bernoulli(probs)

    def gibbs_steps(self, k, initial_state, overwrite=False):
        v = (initial_state if overwrite else initial_state.clone()).to(self.weights)
        h = torch.zeros(*v.shape[:-1], self.num_hidden, device=self.device, dtype=DTYPE)
        for _ in range(k):
            self.sample_h_given_v(v, out=h)
            self.sample_v_given_h(h, out=v)
        return v

    def partition(self, space):
        return (-self.effective_energy(space)).logsumexp(0).exp()

# -------------------------------
# ComplexWaveFunction (amp+phase RBMs)
# -------------------------------
class ComplexWaveFunction:
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
            self.rbm_am = module.to(self.device); self.rbm_am.device = self.device
            self.rbm_ph = module.to(self.device).clone(); self.rbm_ph.device = self.device

        self.num_visible = self.rbm_am.num_visible
        self.num_hidden = self.rbm_am.num_hidden

        self.unitary_dict = unitary_dict if unitary_dict is not None else create_dict()
        self.unitary_dict = {k: v.to(self.device) for k, v in self.unitary_dict.items()}

    # basic props
    @property
    def stop_training(self): return self._stop_training
    @stop_training.setter
    def stop_training(self, new_val):
        if isinstance(new_val, bool): self._stop_training = new_val
        else: raise ValueError("stop_training must be bool!")
    @property
    def max_size(self): return 20
    @property
    def networks(self): return ["rbm_am", "rbm_ph"]
    @property
    def rbm_am(self): return self._rbm_am
    @rbm_am.setter
    def rbm_am(self, new_val): self._rbm_am = new_val
    @property
    def rbm_ph(self): return self._rbm_ph
    @rbm_ph.setter
    def rbm_ph(self, new_val): self._rbm_ph = new_val
    @property
    def device(self): return self._device
    @device.setter
    def device(self, new_val): self._device = new_val

    def __getattr__(self, attr):
        return getattr(self.rbm_am, attr)

    # core ops (2-row output)
    def reinitialize_parameters(self):
        for net in self.networks:
            getattr(self, net).initialize_parameters()

    def amplitude(self, v):
        v = v.to(self.device, dtype=DTYPE)
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()

    def phase(self, v):
        v = v.to(self.device, dtype=DTYPE)
        return -0.5 * self.rbm_ph.effective_energy(v)

    def psi(self, v):
        """Return ψ(v) in 2-row format."""
        amp, ph = self.amplitude(v), self.phase(v)
        return make_complex(amp * ph.cos(), amp * ph.sin())

    def psi_normalized(self, v):
        """
        Numerically-stable normalized ψ via log-sum-exp.
        Using normalized ψ downstream (e.g., in KL rotations) improves conditioning.
        """
        v = v.to(self.device, dtype=DTYPE)
        E = self.rbm_am.effective_energy(v)
        log_amp = -0.5 * E
        logZ = torch.logsumexp(-E, dim=0)
        a = torch.exp(log_amp - 0.5 * logZ)
        ph = self.phase(v)
        return make_complex(a * ph.cos(), a * ph.sin())

    def probability(self, v, Z=1.0):
        v = v.to(device=self.device, dtype=DTYPE)
        return (-self.rbm_am.effective_energy(v)).exp() / Z

    def normalization(self, space):
        return self.rbm_am.partition(space)

    def generate_hilbert_space(self, size=None, device=None):
        """
        Generate computational basis states (MSB first) purely in Torch.
        """
        device = self.device if device is None else device
        size = self.rbm_am.num_visible if size is None else size
        if size > self.max_size: raise ValueError("Hilbert space too large!")
        n = 2 ** size
        ar = torch.arange(n, device=device, dtype=torch.long)  # [0..2^size-1]
        shifts = torch.arange(size - 1, -1, -1, device=device, dtype=torch.long)
        bits = ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)
        return bits

    def sample(self, k, num_samples=1, initial_state=None, overwrite=False):
        if initial_state is None:
            dist = torch.distributions.Bernoulli(probs=0.5)
            shape = torch.Size((num_samples, self.num_visible))
            initial_state = dist.sample(shape).to(self.device, dtype=DTYPE)
        return self.rbm_am.gibbs_steps(k, initial_state, overwrite=overwrite)

    # gradients in 2-row API (same as before)
    def am_grads(self, v):
        return make_complex(self.rbm_am.effective_energy_gradient(v, reduce=False))

    def ph_grads(self, v):
        return scalar_mult(make_complex(self.rbm_ph.effective_energy_gradient(v, reduce=False)), I)

    def rotated_gradient(self, basis, sample):
        """
        Core learning signal. The only unstable piece historically was inv(Uψ);
        we now use the safer inverse() with eps=1e-6 above.
        """
        Upsi, Upsi_v, v = rotate_psi_inner_prod(self, basis, sample, include_extras=True)
        inv_Upsi = inverse(Upsi)  # <- safer denominator
        raw_grads = [self.am_grads(v), self.ph_grads(v)]
        rotated_grad = [einsum("ib,ibg->bg", Upsi_v, g) for g in raw_grads]
        grad = [einsum("b,bg->g", inv_Upsi, rg, imag_part=False) for rg in rotated_grad]
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
        unique_bases, indices = np.unique(bases, axis=0, return_inverse=True)
        indices = torch.tensor(indices, device=samples.device)
        for i in range(unique_bases.shape[0]):
            basis = unique_bases[i, :]
            if np.any(basis != "Z"):
                g_am, g_ph = self.rotated_gradient(basis, samples[indices == i, :])
                grad[0] += g_am; grad[1] += g_ph
            else:
                grad[0] += self.rbm_am.effective_energy_gradient(samples[indices == i, :])
        return grad

    def positive_phase_gradients(self, samples_batch, bases_batch=None):
        grad = self.gradient(samples_batch, bases=bases_batch)
        return [g / float(samples_batch.shape[0]) for g in grad]

    def compute_batch_gradients(self, k, samples_batch, neg_batch, bases_batch=None):
        grad = self.positive_phase_gradients(samples_batch, bases_batch=bases_batch)
        vk = self.rbm_am.gibbs_steps(k, neg_batch)
        grad_model = self.rbm_am.effective_energy_gradient(vk)
        grad[0] -= grad_model / float(neg_batch.shape[0])
        return grad

    # training
    def _shuffle_data(self, pos_batch_size, neg_batch_size, num_batches, train_samples, input_bases, z_samples):
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
            pos_bases = np.asarray(input_bases)[pos_perm.cpu().numpy()]  # robust for numpy/torch inputs
            pos_bases_batches = [pos_bases[i:i + pos_batch_size] for i in range(0, len(train_samples), pos_batch_size)]
            return zip(pos_batches, neg_batches, pos_bases_batches)
        else:
            return zip(pos_batches, neg_batches)

    def fit(self, data, epochs=100, pos_batch_size=100, neg_batch_size=None, k=1, lr=1e-3,
            input_bases=None, log_every=5, progbar=True, starting_epoch=1,
            optimizer=torch.optim.SGD, optimizer_args=None, scheduler=None,
            scheduler_args=None, target=None, bases=None, space=None, timeit=False,
            print_metrics=True, metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}"):

        if input_bases is None:
            raise ValueError("input_bases must be provided to train ComplexWaveFunction!")
        if self.stop_training: return {"epoch": []}

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
        if timeit: history["TimeSec"] = []
        if space is None: space = self.generate_hilbert_space()

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
                if self.stop_training: break
            if sch is not None: sch.step()

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
            if timeit: history.setdefault("TimeSec", []).append(time.time() - t0)
            if self.stop_training: break
        return history

# -------------------------------
# Metrics (fixed: log-domain ψ + normalized rotations in KL)
# -------------------------------
def fidelity(nn_state, target, space=None, **kwargs):
    """
    Use normalized ψ for the overlap; return |⟨target|ψ⟩|^2.
    """
    space = nn_state.generate_hilbert_space() if space is None else space
    psi = nn_state.psi_normalized(space)               # stable normalized wavefunction (2-row)
    target = target.to(nn_state.device, dtype=DTYPE)

    re = (real(target) * real(psi) + imag(target) * imag(psi)).sum()
    im = (real(target) * imag(psi) - imag(target) * real(psi)).sum()
    return (re * re + im * im).item()

def _single_basis_KL(target_probs, nn_probs):
    return torch.sum(target_probs * probs_to_logits(target_probs)) - torch.sum(
        target_probs * probs_to_logits(nn_probs)
    )

def KL(nn_state, target, space=None, bases=None, **kwargs):
    """
    Critical stabilization: rotate a **normalized** ψ.
    Unitaries preserve norm, so probabilities in any basis sum to 1 automatically.
    That eliminates large/small scale effects and improves convergence.
    """
    space = nn_state.generate_hilbert_space() if space is None else space
    target = target.to(nn_state.device, dtype=DTYPE)

    KL_val = 0.0
    psi_norm = nn_state.psi_normalized(space)  # 2×N, already normalized

    for basis in bases:
        tgt_psi_r = rotate_psi(nn_state, basis, space, psi=target)
        psi_r     = rotate_psi(nn_state, basis, space, psi=psi_norm)

        nn_probs_r  = absolute_value(psi_r) ** 2     # sums to 1
        tgt_probs_r = absolute_value(tgt_psi_r) ** 2
        KL_val += _single_basis_KL(tgt_probs_r, nn_probs_r)

    return (KL_val / len(bases)).item()

# -------------------------------
# Training script (example)
# -------------------------------
if __name__ == "__main__":
    train_path = "w_state_meas.txt"
    train_bases_path = "w_state_basis.txt"
    psi_path = "w_state_aug.txt"
    bases_path = "w_state_bases.txt"

    train_samples, true_psi, train_bases, bases = load_data(
        train_path, psi_path, train_bases_path, bases_path
    )

    one_hot_indices = [2**i for i in range(true_psi.shape[1].bit_length() - 1)]
    one_hot_true_psi = true_psi[:, one_hot_indices]

    # Plot helper (doesn't affect training)
    true_phases_raw = torch.angle(one_hot_true_psi[0, :] + 1j * one_hot_true_psi[1, :])
    true_phases_wrapped = (true_phases_raw - true_phases_raw[0]) % (2 * np.pi)

    torch.manual_seed(1234)
    unitary_dict = create_dict()

    nv = train_samples.shape[-1]; nh = nv
    nn_state = ComplexWaveFunction(nv, nh, unitary_dict)

    epochs = 70; pbs = 100; nbs = 100; lr = 1e-1; k = 10; log_every = 5
    space = nn_state.generate_hilbert_space()

    history = nn_state.fit(
        train_samples, epochs=epochs, pos_batch_size=pbs,
        neg_batch_size=nbs, lr=lr, k=k, input_bases=train_bases,
        progbar=True, log_every=log_every, target=true_psi,
        bases=bases, space=space, timeit=True, print_metrics=True,
        metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}"
    )

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
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    ax.set_facecolor('white')
    ax.bar(indices - width/2, true_phases_wrapped, width, alpha=0.7, color='gray',
           label=r'$\phi_{\mathrm{true}}$', zorder=1)
    ax.bar(indices + width/2, pred_phases_wrapped, width, alpha=0.7, color='blue',
           label=r'$\phi_{\mathrm{predicted}}$', zorder=2)
    ax.set_xlabel("Basis State", fontsize=14)
    ax.set_ylabel("Phase (radians)", fontsize=14)
    ax.set_title("Phase Comparison: Phase-Augmented $W$ State", fontsize=16)
    ax.set_xticks(indices)
    ax.set_xticklabels([f"${b}$" for b in bitstrings], rotation=45)
    ax.set_ylim(0, 2 * np.pi + 0.2)
    ax.legend(frameon=True, framealpha=1, loc='best', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
