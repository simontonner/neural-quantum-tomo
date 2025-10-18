# RBM wavefunction — barebones, one-device (ASCII, backward-compatible)
# -----------------------------------------------------------------------------
# Goals
# - Keep legacy 2-row public API [Re; Im] while doing complex math in torch.cdouble.
# - Single-device end-to-end; no implicit cpu<->cuda hops.
# - Drop runtime optimizations/caches; keep numerics guardrails intact.
# - Keep training logic (CD) and rotated-basis supervision; drop plotting.
# -----------------------------------------------------------------------------

from math import ceil, sqrt, prod
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------------------------
# Device & dtypes
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double  # real-valued RBM parameters and energies in float64

# -------------------------------
# Data provider (type adapters) & minimal complex helper
# -------------------------------

class TomographyData:
    """
    Pythonic data loader that ingests measurement samples, target amplitudes,
    and basis metadata. Exposes tensors/arrays ready for training and
    convenient masked accessors (Z-only vs mixed).
    """
    def __init__(self, train_path, psi_path, train_bases_path, bases_path, device=DEVICE):
        self.device = device

        # training samples: (N, n) -> float64 on device
        self.train_samples = torch.tensor(
            np.loadtxt(train_path, dtype="float32"),
            dtype=DTYPE, device=device
        )

        # target wavefunction: native complex (cdouble)
        psi_np = np.loadtxt(psi_path, dtype="float64")  # columns [Re, Im]
        self.target_state = torch.tensor(
            psi_np[:, 0] + 1j * psi_np[:, 1],
            dtype=torch.cdouble, device=device
        )

        # bases metadata (numpy string arrays)
        self.train_bases = np.loadtxt(train_bases_path, dtype=str)
        self.bases = np.loadtxt(bases_path, dtype=str, ndmin=1)

        # tiny cache for masks
        self._mask_cache = {}

    # --- masks ---
    def z_mask(self):
        """Rows where all measurement axes are 'Z'."""
        m = self._mask_cache.get("z")
        if m is None:
            tb = np.asarray(self.train_bases)
            m = torch.as_tensor((tb == "Z").all(axis=1),
                                device=self.train_samples.device,
                                dtype=torch.bool)
            self._mask_cache["z"] = m
        return m

    def nonz_mask(self):
        """Rows with at least one non-Z axis."""
        m = self._mask_cache.get("nonz")
        if m is None:
            m = ~self.z_mask()
            self._mask_cache["nonz"] = m
        return m

    # --- views ---
    def samples(self, mask=None):
        """Return samples optionally masked."""
        if mask is None:
            return self.train_samples
        return self.train_samples[mask]

    def bases_rows(self, mask=None):
        """Return basis rows (numpy array) optionally masked."""
        if mask is None:
            return self.train_bases
        idx = mask.nonzero(as_tuple=False).view(-1).to("cpu").numpy()
        return np.asarray(self.train_bases)[idx]

    # --- convenience subsets ---
    def z_samples(self):
        return self.samples(self.z_mask())

    def mixed_samples(self):
        return self.samples(self.nonz_mask())

    # --- tiny QoL helpers (read-only) ---
    def n_visible(self) -> int:
        return int(self.train_samples.shape[1])

    def num_samples(self) -> int:
        return int(self.train_samples.shape[0])

    def target(self) -> torch.Tensor:
        """Complex target ψ (cdouble) on device."""
        return self.target_state

    def bases_all(self):
        """Full list of measurement bases (numpy array)."""
        return self.bases

    def train_bases_all(self):
        """Training bases rows (numpy array)."""
        return self.train_bases

    # Back-compat alias (used earlier)
    def ref_basis_mask(self):
        return self.z_mask()





class DataProvider:
    """
    Centralize legacy 2-row <-> native complex conversions so the core stays
    purely complex. Use only at ingestion/egress boundaries.
    """
    @staticmethod
    def from_2row(x: torch.Tensor) -> torch.Tensor:
        """(2, N, ...) real -> native complex (cdouble) on x.device."""
        return (
                x[0].to(dtype=torch.cdouble, device=x.device)
                + 1j * x[1].to(dtype=torch.cdouble, device=x.device)
        )

    @staticmethod
    def to_2row(z: torch.Tensor) -> torch.Tensor:
        """native complex -> (2, N, ...) real (DTYPE) on z.device."""
        return torch.stack((z.real.to(DTYPE), z.imag.to(DTYPE)), dim=0)


def inverse(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Numerically safe complex inverse (always returns complex)."""
    zz = z.to(torch.cdouble)
    return zz.conj() / (zz.abs().pow(2).clamp_min(eps))

# -------------------------------
# Unitaries and rotations
# -------------------------------

def create_dict(**kwargs):
    """
    Single-qubit unitaries stored in 2-row [Re; Im] format.
    X and Y are normalized; Z is identity. User-supplied override allowed.
    """
    inv_sqrt2 = 1.0 / sqrt(2.0)
    dictionary = {
        "X": torch.tensor(
            [[[1.0, 1.0], [1.0, -1.0]], [[0.0, 0.0], [0.0, 0.0]]],
            dtype=DTYPE, device=DEVICE,
        ) * inv_sqrt2,
        "Y": torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0]], [[0.0, -1.0], [0.0, 1.0]]],
            dtype=DTYPE, device=DEVICE,
        ) * inv_sqrt2,
        "Z": torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
            dtype=DTYPE, device=DEVICE,
        ),
    }
    dictionary.update({
        name: (
            matrix.clone().detach()
            if isinstance(matrix, torch.Tensor)
            else torch.tensor(matrix)
        ).to(dtype=DTYPE, device=DEVICE)
        for name, matrix in kwargs.items()
    })
    return dictionary



def _kron_mult(matrices, x):
    """
    Apply (⊗_s U_s) to ψ without forming the full Kronecker.
    Inputs MUST be native complex. Vectorized via reshape+einsum.
    """
    assert all(torch.is_complex(m) for m in matrices), "unitaries must be complex"
    if not torch.is_complex(x):
        raise TypeError("x must be a complex tensor (cdouble)")

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



def rotate_psi(nn_state, basis, space, unitaries=None, psi=None):
    """
    Rotate ψ into `basis`.
    Accepts ψ as native complex OR legacy 2-row [Re; Im].
    """
    # --- shape guard ---
    n_vis = nn_state.num_visible
    if len(basis) != n_vis:
        raise ValueError(f"rotate_psi: basis length {len(basis)} != num_visible {n_vis}")

    # Build unitaries (complex)
    if unitaries is None:
        nn_state._ensure_unitary_cache()
        us = [nn_state._unitary_dict_cd[b] for b in basis]
    else:
        uc = {
            k: (m if torch.is_complex(m) else (
                    m[0].to(dtype=torch.cdouble, device=nn_state.device)
                    + 1j * m[1].to(dtype=torch.cdouble, device=nn_state.device)
            ))
            for k, m in unitaries.items()
        }
        us = [uc[b] for b in basis]

    # Prepare input state (complex)
    if psi is None:
        x = nn_state.psi_cd(space)
    else:
        if torch.is_complex(psi):
            x = psi.to(device=nn_state.device, dtype=torch.cdouble)
        elif psi.dim() >= 1 and psi.size(0) == 2:  # legacy 2-row [Re; Im]
            x = DataProvider.from_2row(psi).to(device=nn_state.device, dtype=torch.cdouble)
        else:
            raise TypeError("rotate_psi: psi must be complex or a 2-row [Re; Im] tensor")

    return _kron_mult(us, x)



# -------------------------------
# Rotation kernel for inner products (branch enumeration)
# -------------------------------

def _rotate_basis_state(nn_state, basis, states, unitaries=None):
    """
    Compute branch weights and candidate states after measuring in `basis`.

    Returns:
      Ut : (C, B) torch.cdouble    product over sites of <out|U|in>
      v  : (C, B, n) DTYPE         candidate outcomes (rotated sites enumerated)
    """
    device = nn_state.device

    # --- shape guards ---
    n_vis = nn_state.num_visible
    if len(basis) != n_vis:
        raise ValueError(f"_rotate_basis_state: basis length {len(basis)} != num_visible {n_vis}")
    if states.shape[-1] != n_vis:
        raise ValueError(f"_rotate_basis_state: states width {states.shape[-1]} != num_visible {n_vis}")

    # Turn basis into a simple Python list and find non-Z sites
    basis_seq = list(basis)
    sites = [i for i, b in enumerate(basis_seq) if b != "Z"]

    # No rotated sites: single branch, unit weight
    if len(sites) == 0:
        v = states.unsqueeze(0)  # (1, B, n)
        Ut = torch.ones(v.shape[:-1], dtype=torch.cdouble, device=device)  # (1, B)
        return Ut, v

    # Build complex unitaries for the active sites
    if unitaries is None:
        nn_state._ensure_unitary_cache()  # lazy
        Uc = torch.stack([nn_state._unitary_dict_cd[basis_seq[i]] for i in sites], dim=0)  # (S,2,2)
    else:
        # External dict may be 2-row or complex; normalize to complex on device
        Uc = torch.stack([
            (U if torch.is_complex(U) else DataProvider.from_2row(U)).to(device=device, dtype=torch.cdouble).contiguous()
            for U in (unitaries[basis_seq[i]] for i in sites)
        ], dim=0)  # (S,2,2)

    S = len(sites)
    B = states.shape[0]
    C = 2 ** S

    # Enumerate outcomes on rotated sites
    combos = nn_state.generate_hilbert_space(size=S, device=device)  # (C, S) in DTYPE
    v = states.unsqueeze(0).repeat(C, 1, 1)                          # (C, B, n)
    v[:, :, sites] = combos.unsqueeze(1)                             # fill rotated positions
    v = v.contiguous()

    # Round before int cast to kill 0.999999 -> 1 drift
    inp  = states[:, sites].round().long().T                  # (S, B)
    outp = v[:, :, sites].round().long().permute(0, 2, 1)     # (C, S, B)

    # Gather <out|U|in> per site, multiply over sites
    # Orientation U[out, in] = <out|U|in>
    Uio_exp = Uc.unsqueeze(0).unsqueeze(2).expand(C, S, B, 2, 2)  # (C,S,B,2,2)
    inp_idx = inp.unsqueeze(0).expand(C, S, B).unsqueeze(-1).unsqueeze(-1)  # (C,S,B,1,1)
    sel_in  = torch.gather(Uio_exp, dim=3, index=inp_idx.expand(C, S, B, 1, 2))  # (C,S,B,1,2)
    out_idx = outp.unsqueeze(-1).unsqueeze(-1)                                   # (C,S,B,1,1)
    sel_out = torch.gather(sel_in, dim=4, index=out_idx)                         # (C,S,B,1,1)

    Ut = sel_out.squeeze(-1).squeeze(-1).permute(0, 2, 1).prod(dim=-1)  # (C, B)
    return Ut.to(torch.cdouble), v




def _convert_basis_element_to_index(states):
    """
    Convert bit-rows (B, n) to flat indices (B,), MSB first.
    Guardrail: .round() before .long() avoids 0.999999 -> 1 glitches.
    """
    s = states.round().to(torch.long)
    n = s.shape[-1]
    shifts = torch.arange(n - 1, -1, -1, device=s.device, dtype=torch.long)
    return (s << shifts).sum(dim=-1)


def rotate_psi_inner_prod(nn_state, basis, states, unitaries=None, psi=None, include_extras=False):
    """
    Compute inner-product components for measuring the state in `basis`.

    Returns:
      - total_projected : (B,)  cdouble
      - branch_amplitudes : (C,B) cdouble    (if include_extras)
      - v : (C,B,n) DTYPE                    (if include_extras)
    """
    Ut, v = _rotate_basis_state(nn_state, basis, states, unitaries=unitaries)

    if psi is None:
        # We never backprop through ψ here (grads are computed explicitly elsewhere),
        # so avoid building a graph to speed things up.
        with torch.no_grad():
            psi_sel = nn_state.psi_cd(v)  # (C,B) cdouble
    else:
        idx = _convert_basis_element_to_index(v).long()  # (C,B)
        psi_c = psi if torch.is_complex(psi) else DataProvider.from_2row(psi)
        psi_sel = psi_c.to(dtype=torch.cdouble, device=nn_state.device)[idx]

    Upsi_v_c = Ut * psi_sel
    Upsi_c   = Upsi_v_c.sum(dim=0)

    if include_extras:
        return Upsi_c, Upsi_v_c, v
    return Upsi_c



# -------------------------------
# Explicit gradient utils
# -------------------------------

def vector_to_grads(vec, parameters):
    """
    Write a flattened gradient vector into a module's parameters.
    Assumes all params already live on their correct device/dtype.
    """
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, but got: {torch.typename(vec)}")

    offset = 0
    for p in parameters:
        n = p.numel()
        if offset + n > vec.numel():
            raise ValueError("Gradient vector is too short for parameter shapes.")
        g_slice = vec[offset: offset + n].view_as(p)
        # Ensure grad buffer exists with the right shape/device/dtype
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
    Guardrails:
      - All math in DTYPE (float64) for stability.
      - Parameters have requires_grad=False; we inject grads manually.
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
        Return shape matches input batch rank.
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
          - If reduce=True: returns a flat vector of shape (num_pars,)
          - If reduce=False: returns per-sample grads with trailing grad-dim
        Shapes:
          W_grad: (H, V), vb_grad: (V,), hb_grad: (H,)
        """
        v = (v.unsqueeze(0) if v.dim() < 2 else v).to(self.weights)  # (..., V)
        prob = self.prob_h_given_v(v)                                # (..., H)

        if reduce:
            # batch-reduced grads
            W_grad = -torch.matmul(prob.transpose(0, -1), v)         # (H, V)
            vb_grad = -torch.sum(v, dim=0)                           # (V,)
            hb_grad = -torch.sum(prob, dim=0)                        # (H,)
            # flat vector (H*V + V + H)
            return torch.cat([W_grad.reshape(-1), vb_grad, hb_grad], dim=0)

        # per-sample grads (keep batch rank)
        W_grad = -torch.einsum("...h,...v->...hv", prob, v)          # (..., H, V)
        vb_grad = -v                                                 # (..., V)
        hb_grad = -prob                                              # (..., H)
        vec = [W_grad.view(*v.shape[:-1], -1), vb_grad, hb_grad]     # (..., H*V), (..., V), (..., H)
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
        Guardrail: `overwrite=False` preserves caller tensor unless explicitly allowed.
        """
        v = (initial_state if overwrite else initial_state.clone()).to(self.weights)
        h = torch.zeros(*v.shape[:-1], self.num_hidden, device=self.device, dtype=DTYPE)
        for _ in range(k):
            self.sample_h_given_v(v, out=h)
            self.sample_v_given_h(h, out=v)
        return v

# -------------------------------
# ComplexWaveFunction (amp + phase RBMs)
# -------------------------------
class ComplexWaveFunction:
    """
    Two real RBMs define magnitude and phase of the wavefunction over bitstrings:
      wf(v) = exp(-E_am(v)/2) * exp(+i * (-E_ph(v)/2))

    Public surface keeps legacy names plus clearer aliases.
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

        self.unitary_dict = unitary_dict if unitary_dict is not None else create_dict()
        self.unitary_dict = {k: v.to(self.device) for k, v in self.unitary_dict.items()}
        self._unitary_dict_cd = None  # build lazily on first use

        self._stop_training = False
        self._max_size = 20

    def _ensure_unitary_cache(self):
        if self._unitary_dict_cd is None:
            self._unitary_dict_cd = {
                k: (v[0].to(dtype=torch.cdouble, device=self.device)
                    + 1j * v[1].to(dtype=torch.cdouble, device=self.device)).contiguous()
                for k, v in self.unitary_dict.items()
            }

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

    # amplitudes, phases, and wf
    def reinitialize_parameters(self):
        self.rbm_am.initialize_parameters()
        self.rbm_ph.initialize_parameters()

    def amplitude(self, v):
        v = v.to(self.device, dtype=DTYPE)
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()

    def phase(self, v):
        v = v.to(self.device, dtype=DTYPE)
        return -0.5 * self.rbm_ph.effective_energy(v)

    # Legacy 2-row API
    def psi(self, v):
        """
        Legacy 2-row wavefunction [Re; Im].
        Delegates to the complex core (psi_cd) and converts at the boundary.
        """
        return DataProvider.to_2row(self.psi_cd(v))


    def psi_cd(self, v):
        v = v.to(self.device, dtype=DTYPE)
        amp = (-self.rbm_am.effective_energy(v)).exp().sqrt()
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        return amp.to(torch.cdouble) * torch.exp(1j * ph.to(torch.cdouble))

    def psi_cd_normalized(self, v):
        v = v.to(self.device, dtype=DTYPE)
        E = self.rbm_am.effective_energy(v)
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        logZ = torch.logsumexp(-E, dim=0)
        return torch.exp(((-0.5 * E) - 0.5 * logZ).to(torch.cdouble) + 1j * ph.to(torch.cdouble))

    def psi_normalized(self, v):
        return DataProvider.to_2row(self.psi_cd_normalized(v))

    # Aliases
    def wavefunction_2row(self, v): return self.psi(v)
    def wavefunction_complex(self, v): return self.psi_cd(v)
    def wavefunction_complex_normalized(self, v): return self.psi_cd_normalized(v)
    def wavefunction_normalized_2row(self, v): return self.psi_normalized(v)
    def phase_angle(self, v): return self.phase(v)

    def generate_hilbert_space(self, size=None, device=None):
        """
        Enumerate computational basis as a bit-matrix of shape (2^size, size).
        Guardrails:
          - Error if size exceeds safe max.
          - Tiny lazy cache keyed by (size, device) to avoid recomputation.
        """
        device = self.device if device is None else device
        size = self.rbm_am.num_visible if size is None else size
        if size > self.max_size:
            raise ValueError("Hilbert space too large!")

        # Lazy cache: create on first use
        if not hasattr(self, "_hilbert_cache"):
            self._hilbert_cache = {}

        key = (int(size), device.type, device.index)
        cached = self._hilbert_cache.get(key, None)
        if cached is not None:
            return cached

        n = 2 ** size
        ar = torch.arange(n, device=device, dtype=torch.long)
        shifts = torch.arange(size - 1, -1, -1, device=device, dtype=torch.long)
        bits = ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)

        self._hilbert_cache[key] = bits
        return bits

    # gradients (complex via mapping; final grads are real DTYPE)
    def am_grads(self, v):
        g = self.rbm_am.effective_energy_gradient(v, reduce=False)
        return g.to(torch.cdouble)

    def ph_grads(self, v):
        g = self.rbm_ph.effective_energy_gradient(v, reduce=False)
        return (1j * g.to(torch.cdouble))

    def rotated_gradient(self, basis, sample):
        Upsi, Upsi_v, v = rotate_psi_inner_prod(self, basis, sample, include_extras=True)
        inv_Upsi = inverse(Upsi)  # complex (B,)
        raw_grads = [self.am_grads(v), self.ph_grads(v)]  # both complex
        rotated_grad = [torch.einsum("cb,cbg->bg", Upsi_v, g) for g in raw_grads]
        grad = [torch.einsum("b,bg->g", inv_Upsi, rg).real.to(DTYPE) for rg in rotated_grad]
        return grad

    def gradient(self, samples, bases=None):
        """
        Compute positive-phase gradients.
        Returns [grad_am, grad_ph] in real DTYPE on self.device.

        - If bases is None: only amplitude gradient in Z basis (legacy path).
        - Else: group identical basis rows and accumulate rotated gradients.
        """
        G_am = torch.zeros(self.rbm_am.num_pars, dtype=DTYPE, device=self.device)
        G_ph = torch.zeros(self.rbm_ph.num_pars, dtype=DTYPE, device=self.device)

        if bases is None:
            G_am = self.rbm_am.effective_energy_gradient(samples)
            return [G_am, G_ph]

        # --- shape guards, NumPy-free ---
        try:
            bases_seq = [tuple(row) for row in bases]  # list[tuple[str,...]]
        except Exception as e:
            raise ValueError("gradient: `bases` must be an iterable of rows (e.g. list/np.array of strings).") from e

        B = len(bases_seq)
        if B == 0:
            return [G_am, G_ph]
        n = len(bases_seq[0])
        if any(len(row) != n for row in bases_seq):
            raise ValueError("gradient: all basis rows must have the same width.")
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
                G_ph += g_ph
            else:
                G_am += self.rbm_am.effective_energy_gradient(samples[idxs_t, :])

        return [G_am, G_ph]


    # CD training plumbing
    def positive_phase_gradients(self, samples_batch, bases_batch=None):
        grad = self.gradient(samples_batch, bases=bases_batch)
        return [g / float(samples_batch.shape[0]) for g in grad]

    def compute_batch_gradients(self, k, samples_batch, neg_batch, bases_batch=None):
        grad = self.positive_phase_gradients(samples_batch, bases_batch=bases_batch)
        vk = self.rbm_am.gibbs_steps(k, neg_batch)
        grad_model = self.rbm_am.effective_energy_gradient(vk)
        grad[0] -= grad_model / float(neg_batch.shape[0])
        return grad

    # training loop (compact)
    def _shuffle_data(self, pos_batch_size, neg_batch_size, num_batches,
                      train_samples, input_bases, z_samples):
        """
        Create positive and negative batches.
        Negative samples are drawn from Z-basis rows (supervised path only).
        NumPy-free basis handling.
        """
        # Positive samples: shuffle once per epoch
        pos_perm = torch.randperm(train_samples.shape[0], device=self.device)
        pos_samples = train_samples[pos_perm]

        # Negative samples: draw from Z-basis pool
        neg_perm = torch.randint(
            z_samples.shape[0],
            size=(num_batches * neg_batch_size,),
            dtype=torch.long,
            device=self.device,
        )
        neg_samples = z_samples[neg_perm]

        # Chunk into batches
        pos_batches = [
            pos_samples[i:i + pos_batch_size]
            for i in range(0, len(pos_samples), pos_batch_size)
        ]
        neg_batches = [
            neg_samples[i:i + neg_batch_size]
            for i in range(0, len(neg_samples), neg_batch_size)
        ]

        # Bases: convert once to a Python list-of-tuples, then index via the same permutation
        try:
            bases_list = [tuple(row) for row in input_bases]
        except Exception as e:
            raise ValueError("shuffle_data: `input_bases` must be an iterable of basis rows.") from e

        perm_idx = pos_perm.detach().cpu().tolist()
        pos_bases = [bases_list[i] for i in perm_idx]
        pos_bases_batches = [
            pos_bases[i:i + pos_batch_size]
            for i in range(0, len(bases_list), pos_batch_size)
        ]

        return zip(pos_batches, neg_batches, pos_bases_batches)



    def fit(self, data, epochs=100, pos_batch_size=100, neg_batch_size=None, k=1, lr=1e-3,
            input_bases=None, log_every=5, starting_epoch=1,
            optimizer=torch.optim.SGD, optimizer_args=None, target=None, bases=None, space=None,
            print_metrics=True, metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}"):
        """
        Contrastive Divergence training with optional rotated-basis positive phase.
        NumPy-free in the hot path (pure-Python checks + masks).
        """
        # --- ingest ---
        neg_batch_size = neg_batch_size or pos_batch_size
        optimizer_args = {} if optimizer_args is None else optimizer_args

        if isinstance(data, TomographyData):
            train_samples = data.train_samples.to(self.device, dtype=DTYPE)
            input_bases_obj = data.train_bases if input_bases is None else input_bases
            target = data.target_state if target is None else target
            bases = data.bases if bases is None else bases
            z_samples = data.z_samples().to(self.device, dtype=DTYPE)
        else:
            if input_bases is None:
                raise ValueError("input_bases must be provided to train ComplexWaveFunction!")
            train_samples = data.clone().detach().to(self.device, dtype=DTYPE) if isinstance(data, torch.Tensor) \
                else torch.tensor(data, device=self.device, dtype=DTYPE)
            input_bases_obj = input_bases
            # Pure-Python Z-mask (no NumPy)
            try:
                bases_seq_tmp = [tuple(row) for row in input_bases_obj]
            except Exception as e:
                raise ValueError("fit: `input_bases` must be an iterable of rows (strings).") from e
            z_mask_list = [all(ch == "Z" for ch in row) for row in bases_seq_tmp]
            z_mask = torch.tensor(z_mask_list, device=train_samples.device, dtype=torch.bool)
            if z_mask.numel() != train_samples.shape[0]:
                raise ValueError("fit: Z-mask length != number of training samples.")
            z_samples = train_samples[z_mask]

        # --- pure-Python shape guards for input_bases ---
        try:
            bases_seq = [tuple(row) for row in input_bases_obj]
        except Exception as e:
            raise ValueError("fit: `input_bases` must be an iterable of rows (strings).") from e
        if len(bases_seq) == 0:
            raise ValueError("fit: empty `input_bases`.")
        B = len(bases_seq)
        n = len(bases_seq[0])
        if any(len(row) != n for row in bases_seq):
            raise ValueError("fit: all basis rows must have the same width.")
        if n != self.num_visible:
            raise ValueError(f"fit: basis width {n} != num_visible {self.num_visible}.")
        if train_samples.shape[0] != B:
            raise ValueError(f"fit: input_bases rows {B} != training samples {train_samples.shape[0]}.")
        if z_samples.numel() == 0:
            raise ValueError("fit: no Z-basis rows found for negative sampling.")

        if self.stop_training:
            return {"epoch": []}

        # --- opt setup ---
        all_params = list(chain.from_iterable(getattr(self, n).parameters() for n in ["rbm_am", "rbm_ph"]))
        opt = optimizer(all_params, lr=lr, **optimizer_args)

        history = {"epoch": []}
        if target is not None:
            history["Fidelity"], history["KL"] = [], []
        if space is None:
            space = self.generate_hilbert_space()

        num_batches = ceil(train_samples.shape[0] / pos_batch_size)

        # --- training ---
        for ep in range(starting_epoch, epochs + 1):
            data_iter = self._shuffle_data(pos_batch_size, neg_batch_size, num_batches,
                                           train_samples, input_bases_obj, z_samples)
            for batch in data_iter:
                grads = self.compute_batch_gradients(k, *batch)
                opt.zero_grad()
                for i, net in enumerate(["rbm_am", "rbm_ph"]):
                    rbm = getattr(self, net)
                    vector_to_grads(grads[i], rbm.parameters())
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




@torch.no_grad()
def fidelity(nn_state, target, space=None, **kwargs):
    """
    Squared overlap of normalized states: |<target|psi>|^2.
    """
    space = nn_state.generate_hilbert_space() if space is None else space
    psi = nn_state.psi_cd_normalized(space).reshape(-1).contiguous()

    tgt = target.to(nn_state.device)
    tgt = (tgt if torch.is_complex(tgt) else DataProvider.from_2row(tgt)).reshape(-1).contiguous()

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
    """
    Average KL(p||q) across provided measurement bases.
    Rotate normalized complex state; unitaries preserve norm.
    """
    if bases is None:
        raise ValueError("KL needs `bases` (e.g. from TomographyData.bases)")

    space = nn_state.generate_hilbert_space() if space is None else space

    tgt = target.to(nn_state.device)
    tgt_cd = tgt if torch.is_complex(tgt) else DataProvider.from_2row(tgt)

    psi_norm_cd = nn_state.psi_cd_normalized(space)  # normalized model ψ in computational basis

    KL_val = 0.0
    for basis in bases:
        tgt_psi_r = rotate_psi(nn_state, basis, space, psi=tgt_cd)   # complex
        psi_r     = rotate_psi(nn_state, basis, space, psi=psi_norm_cd)

        nn_probs_r  = (psi_r.abs().to(DTYPE)) ** 2
        tgt_probs_r = (tgt_psi_r.abs().to(DTYPE)) ** 2

        p = tgt_probs_r.clamp_min(1e-12)
        q = nn_probs_r.clamp_min(1e-12)
        KL_val += torch.sum(p * (torch.log(p) - torch.log(q)))

    return (KL_val / len(bases)).item()


# Aliases for metrics

def state_fidelity(nn_state, target, **kwargs):
    return fidelity(nn_state, target, **kwargs)


def average_kl_divergence(nn_state, target, **kwargs):
    return KL(nn_state, target, **kwargs)

# -------------------------------
# Example script (no plotting)
# -------------------------------
if __name__ == "__main__":
    # File names (adjust paths as needed)
    train_path = "w_state_meas.txt"
    train_bases_path = "w_state_basis.txt"
    psi_path = "w_state_aug.txt"
    bases_path = "w_state_bases.txt"

    # Use the pythonic dataloader
    data_io = TomographyData(train_path, psi_path, train_bases_path, bases_path, device=DEVICE)

    torch.manual_seed(1234)
    unitary_dict = create_dict()

    nv = data_io.n_visible()
    nh = nv
    nn_state = ComplexWaveFunction(nv, nh, unitary_dict)

    # Typical CD settings (tune as needed)
    epochs = 70
    pbs = 100
    nbs = 100
    lr = 1e-1
    k = 10
    log_every = 5
    space = nn_state.generate_hilbert_space()

    # Fit using the dataloader; no need to pass input_bases/target/bases
    history = nn_state.fit(
        data_io, epochs=epochs, pos_batch_size=pbs,
        neg_batch_size=nbs, lr=lr, k=k,
        log_every=log_every, space=space, print_metrics=True,
        metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}"
    )

    # History now contains only logged epochs and metrics. No plots.

