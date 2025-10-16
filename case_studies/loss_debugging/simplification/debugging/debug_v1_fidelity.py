# nqt_debug_suite_v2.py
# Deep-dive diagnostics for the 2-row complex RBM tomography stack.
# Focus: explain F=0 by checking target norm/shape, bit-order, overlap, KL, Uψ denominators, dataset integrity.
# Patch: canonical overlap (no torch.vdot) + optional amplitude-only L1 readout.

import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.convert_parameters import _check_param_device
from torch.nn.utils import parameters_to_vector
from itertools import chain, product

# =========================
# Core config
# =========================
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE   = torch.double
CDTYPE  = torch.cdouble
INV_EPS = 1e-6
SEED    = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================
# 2-row complex helpers
# =========================
def make_complex(x, y=None):
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
def _to_cd(x):  # 2-row -> complex
    return real(x).to(dtype=CDTYPE, device=DEVICE) + 1j * imag(x).to(dtype=CDTYPE, device=DEVICE)
def _from_cd(z): return make_complex(z.real.to(DTYPE), z.imag.to(DTYPE))
def absolute_value(x): return _to_cd(x).abs().to(DTYPE)

# =========================
# Unitaries & rotations
# =========================
def create_dict(**kwargs):
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
    dictionary.update({
        name: (matrix.clone().detach() if isinstance(matrix, torch.Tensor) else torch.tensor(matrix)
               ).to(dtype=DTYPE, device=DEVICE)
        for name, matrix in kwargs.items()
    })
    return dictionary

def matmul(x, y):
    X = _to_cd(x); Y = _to_cd(y)
    Z = torch.einsum('ab,b...->a...', X, Y)
    return _from_cd(Z)

def generate_hilbert_space(size, device=DEVICE):
    if size <= 0: return torch.zeros(0, 0, dtype=DTYPE, device=device)
    ar = torch.arange(2**size, device=device, dtype=torch.long)
    shifts = torch.arange(size-1, -1, -1, device=device, dtype=torch.long)  # MSB first
    bits = ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)
    return bits

def _rotate_basis_state(unitary_dict, basis, states, device):
    basis_s = "".join(basis) if isinstance(basis, (list, tuple)) else str(basis)
    basis_arr = np.array(list(basis_s))
    sites = np.where(basis_arr != "Z")[0]

    if sites.size == 0:
        v = states.unsqueeze(0)  # (1,B,n)
        Ut = torch.ones(v.shape[:-1], dtype=CDTYPE, device=device)  # (1,B)
        return Ut, v

    Us = torch.stack([unitary_dict[b] for b in basis_arr[sites]], dim=0).to(device=device)  # (S,2,2,2)
    Uc = Us[:, 0, ...].to(dtype=CDTYPE) + 1j * Us[:, 1, ...].to(dtype=CDTYPE)             # (S,2,2)
    Uio = Uc  # <out|U|in>

    S = len(sites)
    B = states.shape[0]
    C = 2 ** S

    combos = generate_hilbert_space(size=S, device=device)   # (C,S)
    v = states.unsqueeze(0).repeat(C, 1, 1)                  # (C,B,n)
    v[:, :, sites] = combos.unsqueeze(1)
    v = v.contiguous()

    inp  = states[:, sites].round().long().T                 # (S,B)   OUT indices
    outp = v[:, :, sites].round().long().permute(0, 2, 1)    # (C,S,B) IN indices

    Uio_exp = Uio.unsqueeze(0).unsqueeze(2).expand(C, S, B, 2, 2)   # (C,S,B,2_in,2_out)
    inp_idx = inp.unsqueeze(0).expand(C, S, B).unsqueeze(-1).unsqueeze(-1)
    sel_in  = torch.gather(Uio_exp, 3, inp_idx.expand(C, S, B, 1, 2))
    out_idx = outp.unsqueeze(-1).unsqueeze(-1)
    sel_out = torch.gather(sel_in, 4, out_idx)

    Ut = sel_out.squeeze(-1).squeeze(-1).permute(0, 2, 1).prod(dim=-1)  # (C,B)
    return Ut, v

def _convert_basis_element_to_index(states):
    n = states.shape[-1]
    powers = (2 ** (torch.arange(n, 0, -1, device=states.device) - 1)).to(states)
    return torch.matmul(states, powers)

def rotate_psi(nn_state, basis, space, psi=None):
    psi2 = nn_state.psi(space) if psi is None else psi.to(device=DEVICE, dtype=DTYPE)
    unitaries = {k: v.to(device=DEVICE) for k, v in nn_state.unitary_dict.items()}
    us = [unitaries[b] for b in (list(basis) if not isinstance(basis, (list, tuple)) else basis)]
    return _kron_mult(us, psi2)

def rotate_psi_inner_prod(nn_state, basis, states, psi=None, include_extras=False):
    Ut, v = _rotate_basis_state(nn_state.unitary_dict, basis, states, nn_state.device)  # Ut: (C,B)
    if psi is None:
        psi_sel = _to_cd(nn_state.psi(v))                    # (C,B)
    else:
        idx = _convert_basis_element_to_index(v).long()      # (C,B)
        psi_sel = _to_cd(psi)[idx]                           # (C,B)
    Upsi_v_c = Ut * psi_sel                                  # (C,B)
    Upsi_c   = Upsi_v_c.sum(dim=0)                           # (B,)
    Upsi     = _from_cd(Upsi_c)                              # (2,B)
    if include_extras:
        return Upsi, _from_cd(Upsi_v_c), v
    return Upsi

def _kron_mult(matrices, x):
    n = [m.size(1) for m in matrices]  # matrix is (2,2,2): channel, row, col
    l, r = int(np.prod(n)), 1
    if l != x.shape[1]: raise ValueError("Incompatible sizes!")
    y = x.clone()
    for s in reversed(range(len(n))):
        l //= n[s]; m = matrices[s]
        for k in range(l):
            for i in range(r):
                slc = slice(k * n[s] * r + i, (k + 1) * n[s] * r + i, r)
                y[:, slc, ...] = matmul(m, y[:, slc, ...])
        r *= n[s]
    return y

# =========================
# RBM (real)
# =========================
class BinaryRBM(nn.Module):
    def __init__(self, num_visible, num_hidden=None, device: torch.device = DEVICE):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden) if num_hidden else self.num_visible
        self.num_pars = (self.num_visible * self.num_hidden) + self.num_visible + self.num_hidden
        self.device = device
        self.initialize_parameters()

    def initialize_parameters(self):
        self.weights = nn.Parameter(torch.randn(self.num_hidden, self.num_visible, device=self.device, dtype=DTYPE)
                                    / np.sqrt(self.num_visible), requires_grad=False)
        self.visible_bias = nn.Parameter(torch.zeros(self.num_visible, device=self.device, dtype=DTYPE),
                                         requires_grad=False)
        self.hidden_bias = nn.Parameter(torch.zeros(self.num_hidden, device=self.device, dtype=DTYPE),
                                        requires_grad=False)

    def effective_energy(self, v):
        unsq = False
        if v.dim() < 2: v = v.unsqueeze(0); unsq = True
        v = v.to(self.weights)
        visible_bias_term = torch.matmul(v, self.visible_bias)
        hid_bias_term = F.softplus(F.linear(v, self.weights, self.hidden_bias)).sum(-1)
        out = -(visible_bias_term + hid_bias_term)
        return out.squeeze(0) if unsq else out

    def prob_v_given_h(self, h, out=None):
        unsq = False
        if h.dim() < 2: h = h.unsqueeze(0); unsq = True
        res = torch.matmul(h, self.weights.data).add_(self.visible_bias.data).sigmoid_().clamp_(0, 1)
        if out is not None:
            out.copy_(res.squeeze(0) if unsq and out.dim() == 1 else res); return out
        return res.squeeze(0) if unsq else res

    def prob_h_given_v(self, v, out=None):
        unsq = False
        if v.dim() < 2: v = v.unsqueeze(0); unsq = True
        res = torch.matmul(v, self.weights.data.t()).add_(self.hidden_bias.data).sigmoid_().clamp_(0, 1)
        if out is not None:
            out.copy_(res.squeeze(0) if unsq and out.dim() == 1 else res); return out
        return res.squeeze(0) if unsq else res

    def effective_energy_gradient(self, v, reduce=True):
        v = (v.unsqueeze(0) if v.dim() < 2 else v).to(self.weights)
        prob = self.prob_h_given_v(v)
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

    def gibbs_steps(self, k, initial_state, overwrite=False):
        v = (initial_state if overwrite else initial_state.clone()).to(self.weights)
        h = torch.zeros(*v.shape[:-1], self.num_hidden, device=self.device, dtype=DTYPE)
        for _ in range(k):
            self.prob_h_given_v(v, out=h); torch.bernoulli(h, out=h)
            self.prob_v_given_h(h, out=v); torch.bernoulli(v, out=v)
        return v

# =========================
# Wavefunction (amp+phase RBMs)
# =========================
class ComplexWaveFunction:
    def __init__(self, num_visible, num_hidden=None, unitary_dict=None):
        self.device = DEVICE
        self.rbm_am = BinaryRBM(num_visible, num_hidden, device=self.device)
        self.rbm_ph = BinaryRBM(num_visible, num_hidden, device=self.device)
        self.num_visible = self.rbm_am.num_visible
        self.unitary_dict = (unitary_dict if unitary_dict is not None else create_dict())

    def amplitude(self, v):
        v = v.to(self.device, DTYPE)
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()

    def phase(self, v):
        v = v.to(self.device, DTYPE)
        return -0.5 * self.rbm_ph.effective_energy(v)

    def psi(self, v):
        amp, ph = self.amplitude(v), self.phase(v)
        return make_complex(amp * ph.cos(), amp * ph.sin())

    def psi_normalized(self, v):
        v = v.to(self.device, DTYPE)
        E = self.rbm_am.effective_energy(v)
        log_amp = -0.5 * E
        logZ = torch.logsumexp(-E, dim=0)
        a = torch.exp(log_amp - 0.5 * logZ)
        ph = self.phase(v)
        return make_complex(a * ph.cos(), a * ph.sin())

    def generate_hilbert_space(self, size=None, device=None):
        device = self.device if device is None else device
        size = self.num_visible if size is None else size
        return generate_hilbert_space(size, device=device)

# =========================
# Data loading / fallback
# =========================
def try_load_files():
    files = {
        "train_path": "w_state_meas.txt",
        "train_bases_path": "w_state_basis.txt",
        "psi_path": "w_state_aug.txt",
        "bases_path": "w_state_bases.txt",
    }
    if all(os.path.exists(v) for v in files.values()):
        train_samples = torch.tensor(np.loadtxt(files["train_path"], dtype="float64"), dtype=DTYPE, device=DEVICE)
        target_psi_data = np.loadtxt(files["psi_path"], dtype="float64")
        target_psi = torch.zeros(2, len(target_psi_data), dtype=DTYPE, device=DEVICE)
        target_psi[0] = torch.tensor(target_psi_data[:, 0], dtype=DTYPE, device=DEVICE)
        target_psi[1] = torch.tensor(target_psi_data[:, 1], dtype=DTYPE, device=DEVICE)
        train_bases = np.loadtxt(files["train_bases_path"], dtype=str)
        bases = np.loadtxt(files["bases_path"], dtype=str, ndmin=1)
        return train_samples, target_psi, train_bases, bases
    return None

def synth_w_state(n=3):
    # |W>_3 = (|001>+|010>+|100>)/sqrt(3), MSB-first indexing -> indices [1,2,4]
    hs = generate_hilbert_space(size=n)
    N = hs.shape[0]
    idx = [1, 2, 4]
    re = torch.zeros(N, dtype=DTYPE, device=DEVICE); im = torch.zeros_like(re)
    re[idx] = 1/np.sqrt(3)
    psi = make_complex(re, im)
    samples = torch.bernoulli(torch.full((500, n), 0.5, dtype=DTYPE, device=DEVICE))
    bases_mat = np.random.choice(list("XYZ"), size=(samples.shape[0], n))
    bases_eval = np.array(["X","Y","Z"], dtype=str)
    return samples, psi, bases_eval, bases_eval  # (compat)

# =========================
# ======= NEW DIAGNOSTICS =====
# =========================
def _unitaries_to_cdouble(unitary_dict):
    Uc = {}
    for k, U2 in unitary_dict.items():
        Uc[k] = (U2[0].to(dtype=CDTYPE) + 1j * U2[1].to(dtype=CDTYPE)).to(U2.device)
    return Uc

@torch.no_grad()
def check_unitaries(unitary_dict, device_expected, dtype_expected, atol=1e-12):
    print("== UNITARY CHECKS ==")
    ok = True
    Uc = _unitaries_to_cdouble(unitary_dict)
    for name, U in Uc.items():
        I = torch.eye(2, dtype=CDTYPE, device=U.device)
        dev = (U.conj().T @ U - I).abs().max().item()
        print(f"  {name}: max|U†U-I| = {dev:.3e}")
        ok &= (dev < atol)
    for name, U2 in unitary_dict.items():
        if U2.device != device_expected or U2.dtype != dtype_expected:
            print(f"  WARN: {name} at {U2.device}/{U2.dtype}, expected {device_expected}/{dtype_expected}")
    print("[PASS] unitarity" if ok else "[FAIL] unitarity")
    return ok

@torch.no_grad()
def check_dataset_consistency(train_samples, train_bases, n_expected):
    print("== DATASET CONSISTENCY ==")
    B, n = train_samples.shape
    print(f"  train_samples shape = ({B}, {n})  expected n={n_expected}")
    if n != n_expected:
        print(f"  [FAIL] visibles mismatch")
    uniq_rows = {}
    has_allZ = 0
    for row in train_bases:
        key = "".join(row) if not isinstance(row, str) else row
        uniq_rows[key] = uniq_rows.get(key, 0) + 1
        if all(ch == "Z" for ch in key): has_allZ += 1
    print(f"  unique basis-rows: {len(uniq_rows)}  sample: {list(list(uniq_rows.keys())[:5])}")
    print(f"  Z-only rows: {has_allZ} / {B} ({100*has_allZ/B:.2f}%)")
    bad = [k for k in uniq_rows if any(ch not in "XYZ" for ch in k) or len(k)!=n_expected]
    if bad:
        print(f"  [FAIL] malformed basis rows detected: {bad[:3]}")

@torch.no_grad()
def check_hilbert_space_msb():
    print("== HILBERT SPACE MSB CHECK ==")
    hs3 = generate_hilbert_space(3)
    ok = (hs3[1].tolist() == [0.,0.,1.]) and (hs3[2].tolist() == [0.,1.,0.]) and (hs3[4].tolist() == [1.,0.,0.])
    print("[PASS] MSB-first bit order (n=3)" if ok else "[FAIL] MSB order (n=3)")

@torch.no_grad()
def compare_rotation_paths(nn_state, basis, space, psi_override_2row=None):
    if psi_override_2row is None:
        psi_2 = nn_state.psi(space)
    else:
        psi_2 = psi_override_2row

    rot_A = _kron_mult([nn_state.unitary_dict[b] for b in list(basis)], psi_2)  # (2,N)
    rot_B = rotate_psi_inner_prod(nn_state, basis, space, psi=psi_2)            # (2,N)
    psi_cd = _to_cd(psi_2)
    Uc = _unitaries_to_cdouble(nn_state.unitary_dict)
    # tensordot path
    n = len(basis)
    t = psi_cd.reshape(*(2 for _ in range(n)))
    for s, b in enumerate(basis):
        U = Uc[b]
        t = torch.movedim(t, s, 0)
        t = torch.tensordot(U, t, dims=([1], [0]))
        t = torch.movedim(t, 0, s)
    rot_C = _from_cd(t.reshape(-1))

    err_AB = (rot_A[0]-rot_B[0]).abs().max().item(), (rot_A[1]-rot_B[1]).abs().max().item()
    err_AC = (rot_A[0]-rot_C[0]).abs().max().item(), (rot_A[1]-rot_C[1]).abs().max().item()
    err_BC = (rot_B[0]-rot_C[0]).abs().max().item(), (rot_B[1]-rot_C[1]).abs().max().item()
    print(f"  basis={basis}  max|A−B| re/imag = {err_AB[0]:.3e}/{err_AB[1]:.3e}   "
          f"max|A−C| = {err_AC[0]:.3e}/{err_AC[1]:.3e}   max|B−C| = {err_BC[0]:.3e}/{err_BC[1]:.3e}")

@torch.no_grad()
def check_rotation_equivalence(nn_state):
    print("== ROTATION EQUIVALENCE (3 paths) ==")
    n = nn_state.num_visible
    hs = nn_state.generate_hilbert_space()
    psi = nn_state.psi_normalized(hs)
    bases = ["Z"*n, "X"*n, "Y"*n, "".join(np.random.default_rng(0).choice(list("XYZ"), size=n))]
    for b in bases:
        compare_rotation_paths(nn_state, b, hs, psi_override_2row=psi)

# -------- Fidelity/Overlap triage --------
@torch.no_grad()
def _cd_norm_sq(z): return (z.conj()*z).real.sum().item()

@torch.no_grad()
def fidelity_direct(target_2row, psi_2row):
    # Robust canonical overlap: <t|psi> = sum conj(t) * psi
    t = _to_cd(target_2row).reshape(-1).contiguous()
    p = _to_cd(psi_2row).reshape(-1).contiguous()
    ov = torch.sum(t.conj() * p)
    return (ov.abs()**2).item(), ov

@torch.no_grad()
def bit_reverse_permutation(nbits, device=DEVICE):
    N = 2**nbits
    ar = torch.arange(N, device=device, dtype=torch.long)
    rev = torch.zeros_like(ar)
    for i in range(nbits):
        rev |= ((ar >> i) & 1) << (nbits-1-i)
    return rev

@torch.no_grad()
def check_fidelity_suite(nn_state, target_psi_2row, space):
    print("== FIDELITY TRIAGE ==")
    # Basic norms
    t_cd = _to_cd(target_psi_2row)
    p_cd = _to_cd(nn_state.psi_normalized(space))
    nt = _cd_norm_sq(t_cd)
    np_ = _cd_norm_sq(p_cd)
    print(f"  ‖target‖^2 = {nt:.6f}   ‖model(ψ_norm)‖^2 = {np_:.6f}")

    # Normalize target for fair comparison
    if nt > 0:
        t_cd_n = t_cd / np.sqrt(nt)
    else:
        print("  [FAIL] target norm is zero!")
        t_cd_n = t_cd

    # Direct fidelities (canonical)
    F_tt, ov_tt = fidelity_direct(_from_cd(t_cd_n), _from_cd(t_cd_n))
    F_pp, ov_pp = fidelity_direct(_from_cd(p_cd), _from_cd(p_cd))
    F_tp, ov_tp = fidelity_direct(_from_cd(t_cd_n), _from_cd(p_cd))
    print(f"  F[target,target]= {F_tt:.12f}   F[psi,psi]= {F_pp:.12f}   F[target,psi]= {F_tp:.12e}")
    print(f"  <t|psi> magnitude = {ov_tp.abs().item():.6e}  angle(rad) = {torch.angle(ov_tp).item():.6f}")

    # Amplitude-only mismatch (Z basis)
    l1_amp = (t_cd_n.abs() - p_cd.abs()).abs().sum().item()
    print(f"  L1(|amplitude|, Z-basis) = {l1_amp:.6e}")

    # Bit-order mismatch probe (MSB↔LSB)
    nbits = space.shape[1]
    perm = bit_reverse_permutation(nbits)
    t_rev = t_cd_n[perm]
    F_trev, ov_trev = fidelity_direct(_from_cd(t_rev), _from_cd(p_cd))
    print(f"  F[bit-reversed(target), psi]= {F_trev:.12e}  (if this >> F_tp, your bit-order is flipped)")

    # Top-k amplitudes to see if supports are disjoint
    k = min(10, t_cd.numel())
    t_probs = (t_cd_n.abs()**2)
    p_probs = (p_cd.abs()**2)
    t_top = torch.topk(t_probs, k)
    p_top = torch.topk(p_probs, k)
    print(f"  top-{k} target idx: {t_top.indices.tolist()}")
    print(f"  top-{k} model  idx: {p_top.indices.tolist()}")
    print(f"  overlap on top-{k} sets: {len(set(t_top.indices.tolist()) & set(p_top.indices.tolist()))}")

@torch.no_grad()
def check_prob_sums(nn_state, bases, space):
    print("== PROB SUMS (normalized ψ) ==")
    psi_n = nn_state.psi_normalized(space)
    for basis in bases:
        psi_r = rotate_psi_inner_prod(nn_state, basis, space, psi=psi_n)
        p_sum = (absolute_value(psi_r)**2).sum().item()
        print(f"  basis={''.join(basis)}  sum p = {p_sum:.12f}")

@torch.no_grad()
def per_basis_KL_vs_target(nn_state, target_psi_2row, bases, space):
    print("== PER-BASIS KL(target || model) ==")
    psi_n = nn_state.psi_normalized(space)
    t_cd = _to_cd(target_psi_2row)
    t_norm = t_cd / (torch.sqrt((t_cd.conj()*t_cd).real.sum()) + 1e-300)
    for basis in bases:
        t_r = rotate_psi_inner_prod(nn_state, basis, space, psi=_from_cd(t_norm))
        p_r = rotate_psi_inner_prod(nn_state, basis, space, psi=psi_n)
        tP = (absolute_value(t_r)**2).clamp_min(1e-300)
        pP = (absolute_value(p_r)**2).clamp_min(1e-300)
        kl = (tP * (torch.log(tP) - torch.log(pP))).sum().item()
        l1 = (tP - pP).abs().sum().item()
        print(f"  basis={''.join(basis)}  KL={kl:.6e}  L1={l1:.6e}")

@torch.no_grad()
def diagnose_Upsi_denominators(nn_state, bases, space, B=128, thresholds=(1e-12, 1e-9, 1e-6, 1e-3)):
    print("== DENOMINATOR HEALTH: |Uψ| over random outputs ==")
    idx = torch.randperm(space.shape[0], device=space.device)[:B]
    samples = space[idx].to(nn_state.device, dtype=DTYPE)
    psi_n = nn_state.psi_normalized(space)
    for basis in bases:
        Upsi = rotate_psi_inner_prod(nn_state, basis, samples, psi=psi_n)  # (2,B)
        mags = _to_cd(Upsi).abs()
        mn, mx = mags.min().item(), mags.max().item()
        print(f"  basis={''.join(basis)}  min|Uψ| = {mn:.3e}   max|Uψ| = {mx:.3e}")
        for t in thresholds:
            frac = (mags < t).float().mean().item()
            print(f"    frac(|Uψ| < {t:g}) = {100*frac:.2f}%")

@torch.no_grad()
def gradient_smoke_test(nn_state, space):
    print("== POS-PHASE GRADS SMOKE TEST ==")
    B = min(64, space.shape[0])
    samp = space[torch.randperm(space.shape[0])[:B]].to(nn_state.device, dtype=DTYPE)
    g_am = nn_state.rbm_am.effective_energy_gradient(samp)
    g_ph = nn_state.rbm_ph.effective_energy_gradient(samp)
    print(f"  ‖grad_am‖ = {g_am.norm().item():.3e}   NaN? {bool(torch.isnan(g_am).any().item())}")
    print(f"  ‖grad_ph‖ = {g_ph.norm().item():.3e}   NaN? {bool(torch.isnan(g_ph).any().item())}")

@torch.no_grad()
def print_topk_states(psi_2row, k=12):
    print("== TOP-K STATES BY PROBABILITY ==")
    z = _to_cd(psi_2row)
    probs = (z.abs()**2)
    top = torch.topk(probs, min(k, probs.numel()))
    print(f"  idx: {top.indices.tolist()}")
    print(f"  prob: {[float(x) for x in top.values.tolist()]}")

# =========================
# Runner
# =========================
def run_all_debug_checks(nn_state, train_samples, train_bases, bases_eval, target_psi_2row):
    print("===== NQT DEBUG SUITE v2 =====")
    print("== ENV ==")
    print(f"torch: {torch.__version__}")
    print(f"device: {DEVICE}, cuda_available={torch.cuda.is_available()}")
    print(f"DTYPE: {DTYPE}, CDTYPE: {CDTYPE}, inverse eps: {INV_EPS}")
    print("[PASS] env/versions")

    n = train_samples.shape[1]
    space = nn_state.generate_hilbert_space(size=n)
    unitary_ok = check_unitaries(nn_state.unitary_dict, nn_state.device, DTYPE)
    check_hilbert_space_msb()
    check_dataset_consistency(train_samples, train_bases, n)

    # Normalize sanity & direct fidelity probes
    check_fidelity_suite(nn_state, target_psi_2row, space)

    # Rotation math equivalence (if unitary OK)
    if unitary_ok:
        check_rotation_equivalence(nn_state)

    # Probability sums and KL mismatch by basis
    bases_pref = ["Z"*n, "X"*n, "Y"*n]
    bases_list = list(bases_pref)
    if isinstance(bases_eval, (list, np.ndarray)):
        # add up to 3 from file for variety
        for s in list(bases_eval)[:3]:
            b = "".join(s) if not isinstance(s, str) else s
            if len(b) == n and all(ch in "XYZ" for ch in b):
                bases_list.append(b)

    check_prob_sums(nn_state, bases_list, space)
    per_basis_KL_vs_target(nn_state, target_psi_2row, bases_list, space)

    # Denominator health & gradient sanity
    diagnose_Upsi_denominators(nn_state, bases_list, space)
    gradient_smoke_test(nn_state, space)

    # Where is mass? Print top-k states for model & target
    print_topk_states(nn_state.psi_normalized(space), k=12)
    print_topk_states(target_psi_2row, k=12)

    print("===== DONE =====")

# =========================
# Main
# =========================
if __name__ == "__main__":
    data = try_load_files()
    if data is None:
        print("Data files not found -> using synthetic W-state fallback.")
        train_samples, true_psi, train_bases, bases = synth_w_state(n=3)
    else:
        train_samples, true_psi, train_bases, bases = data

    nv = train_samples.shape[-1]
    unitary_dict = create_dict()
    nn_state = ComplexWaveFunction(nv, nv, unitary_dict)

    run_all_debug_checks(nn_state, train_samples, train_bases, bases, true_psi)
