# nqt_probe_unified.py
# One-shot, conclusive diagnostics for "fidelity not increasing" in complex NQS tomography
# - Pure complex forward (torch.cdouble); RBM params are real (torch.double)
# - Splits gradients into positive (rotated data) and negative (model/CD) parts
# - Tests multiple step variants (+g, -g, amp-only, phase-only, pos-only, neg-only)
# - Phase-sign grid: tries {+i/-i} × {Real/Imag} mappings for the phase block and recommends the best
# - Finite-difference directional derivatives along training direction
# - Line-scan step sweeps for +g / amp-only / phase-only
# - Gradient geometry (norms & cosines among pos/neg/model)
# - Uψ conditioning: tiny |Uψ| fractions and inv-floor sensitivity
# - Exact model negative gradient for amplitude (small n only; nv<=12)
# - Prints a single "CONCLUSION" with a reasoned verdict, plus a phase-sign recommendation
#
# Optional add-on (toggle with NQT_RUN_PHASE_MIN=1):
# - Minimal phase-only regression: compares 'old' phase mapping (A ≡ -Imag(S)) vs 'C' (Real(S))
#   with directional derivatives and line-scans on a random uniform-phase target.
#
# Usage (defaults on env vars):
#   NQT_TRIALS=5 NQT_CD_K=10 NQT_POS_BS=100 NQT_NEG_BS=100 NQT_STEP=1e-3 NQT_EPS=5e-4 python nqt_probe_unified.py
#   NQT_RUN_PHASE_MIN=1 to run the minimal phase-only regression after the main probe.

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn  # robust to F-shadowing
from torch.distributions.utils import probs_to_logits
from dataclasses import dataclass
from typing import Tuple, List, Dict

# --------------------- core setup ---------------------
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RTYPE   = torch.double
CDTYPE  = torch.cdouble
INV_EPS = float(os.getenv("NQT_INV_EPS", "1e-6"))
SEED    = int(os.getenv("NQT_SEED", "1234"))
torch.set_default_dtype(RTYPE)
torch.manual_seed(SEED)
np.random.seed(SEED)

# --------------------- utils ---------------------
def cinner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Complex inner product ⟨x|y⟩ (conjugate on x)."""
    return torch.sum(x.conj() * y)

def generate_hilbert_space(size: int, device=DEVICE) -> torch.Tensor:
    """All bitstrings of length `size`, as {0,1}-valued rows. float64, shape (2**size, size)."""
    if size <= 0: return torch.zeros(0, 0, dtype=RTYPE, device=device)
    ar = torch.arange(2**size, device=device, dtype=torch.long)
    shifts = torch.arange(size-1, -1, -1, device=device, dtype=torch.long)
    return ((ar.unsqueeze(1) >> shifts) & 1).to(RTYPE)

def probs_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p||q) using logits for stability. p,q normalized probs, shape (N,)."""
    return torch.sum(p * probs_to_logits(p)) - torch.sum(p * probs_to_logits(q))

def split_vec(vec: torch.Tensor, rbm: "BinaryRBM"):
    nh, nv = rbm.num_hidden, rbm.num_visible
    nW = nh * nv
    W = vec[:nW].view(nh, nv)
    vb = vec[nW:nW+nv]
    hb = vec[nW+nv:]
    return W, vb, hb

def vec_norm(x: torch.Tensor) -> float:
    return float(x.norm().item())

def clone_params(rbms: List["BinaryRBM"]):
    return [tuple(p.detach().clone() for p in rbm.parameters()) for rbm in rbms]

def restore_params(rbms: List["BinaryRBM"], snapshot):
    with torch.no_grad():
        for rbm, pack in zip(rbms, snapshot):
            for p, p_old in zip(rbm.parameters(), pack):
                p.copy_(p_old)

def apply_step_into_params(rbm: "BinaryRBM", gvec: torch.Tensor, step_scale: float):
    Wg, vbg, hbg = split_vec(gvec, rbm)
    with torch.no_grad():
        for p, g in zip(rbm.parameters(), [Wg, vbg, hbg]):
            p.add_(-step_scale * g)

# --------------------- unitaries (U[out,in]) ---------------------
def create_unitaries(device=DEVICE):
    """Return dict {'X','Y','Z'} with 2x2 cdouble matrices. Y has +i in (0,1) and -i in (1,1)."""
    s2 = math.sqrt(2.0)
    X = torch.tensor([[1,  1],
                      [1, -1]], dtype=CDTYPE, device=device) / s2
    Y = torch.tensor([[1,  1j],
                      [1, -1j]], dtype=CDTYPE, device=device) / s2
    Z = torch.eye(2, dtype=CDTYPE, device=device)
    return {"X": X, "Y": Y, "Z": Z}

# --------------------- RBM (real params) ---------------------
class BinaryRBM(nn.Module):
    """Binary RBM used for amplitude and phase energies (real parameters only)."""
    def __init__(self, num_visible, num_hidden=None, zero_weights=False, device: torch.device = DEVICE):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden  = int(num_hidden) if num_hidden else self.num_visible
        self.num_pars    = (self.num_visible * self.num_hidden) + self.num_visible + self.num_hidden
        self.device      = device
        self.initialize_parameters(zero_weights=zero_weights)

    def initialize_parameters(self, zero_weights=False):
        gen = torch.zeros if zero_weights else torch.randn
        self.weights = nn.Parameter((gen(self.num_hidden, self.num_visible, device=self.device, dtype=RTYPE)
                                     / math.sqrt(max(1, self.num_visible))), requires_grad=False)
        self.visible_bias = nn.Parameter(torch.zeros(self.num_visible, device=self.device, dtype=RTYPE),
                                         requires_grad=False)
        self.hidden_bias  = nn.Parameter(torch.zeros(self.num_hidden,  device=self.device, dtype=RTYPE),
                                         requires_grad=False)

    def effective_energy(self, v):
        """-log ψ_amp^2 proxy: -[v·vb + sum softplus(vW+hb)]. shape (...,)"""
        unsq = False
        if v.dim() < 2:
            v = v.unsqueeze(0); unsq = True
        v = v.to(self.weights)
        vb = torch.matmul(v, self.visible_bias)
        hb = Fnn.softplus(Fnn.linear(v, self.weights, self.hidden_bias)).sum(-1)
        out = -(vb + hb)
        return out.squeeze(0) if unsq else out

    def prob_v_given_h(self, h, out=None):
        unsq = False
        if h.dim() < 2:
            h = h.unsqueeze(0); unsq = True
        res = torch.matmul(h, self.weights.data).add_(self.visible_bias.data).sigmoid_().clamp_(0,1)
        if out is not None:
            out.copy_(res.squeeze(0) if unsq and out.dim()==1 else res); return out
        return res.squeeze(0) if unsq else res

    def prob_h_given_v(self, v, out=None):
        unsq = False
        if v.dim() < 2:
            v = v.unsqueeze(0); unsq = True
        res = torch.matmul(v, self.weights.data.t()).add_(self.hidden_bias.data).sigmoid_().clamp_(0,1)
        if out is not None:
            out.copy_(res.squeeze(0) if unsq and out.dim()==1 else res); return out
        return res.squeeze(0) if unsq else res

    def sample_v_given_h(self, h, out=None):
        probs = self.prob_v_given_h(h)
        return torch.bernoulli(probs, out=out) if out is not None else torch.bernoulli(probs)

    def sample_h_given_v(self, v, out=None):
        probs = self.prob_h_given_v(v)
        return torch.bernoulli(probs, out=out) if out is not None else torch.bernoulli(probs)

    def gibbs_steps(self, k, initial_state, overwrite=False):
        """CD-k Gibbs chain starting from given v. Returns v_k, shape like initial_state."""
        v = (initial_state if overwrite else initial_state.clone()).to(self.weights)
        h = torch.zeros(*v.shape[:-1], self.num_hidden, device=self.device, dtype=RTYPE)
        for _ in range(k):
            self.sample_h_given_v(v, out=h)
            self.sample_v_given_h(h, out=v)
        return v

    def effective_energy_gradient(self, v, reduce=True):
        """∂(-E)/∂θ for θ=(W,vb,hb). If reduce=False, returns per-sample grads, shape (..., G)."""
        v = (v.unsqueeze(0) if v.dim() < 2 else v).to(self.weights)
        prob = self.prob_h_given_v(v)
        if reduce:
            Wg  = -torch.matmul(prob.transpose(0,-1), v)
            vbg = -torch.sum(v, 0)
            hbg = -torch.sum(prob, 0)
            return torch.cat([Wg.reshape(-1), vbg, hbg])
        else:
            Wg  = -torch.einsum("...j,...k->...jk", prob, v)
            vbg = -v
            hbg = -prob
            return torch.cat([Wg.view(*v.shape[:-1], -1), vbg, hbg], dim=-1)

# --------------------- ComplexWaveFunction ---------------------
class ComplexWaveFunction:
    """ψ(v) = exp(-E_amp(v)/2) * exp(+i * θ(v)), with θ(v) = -0.5*E_phase(v)."""
    def __init__(self, num_visible, num_hidden=None, unitary_dict=None):
        self.device = DEVICE
        self.rbm_am = BinaryRBM(num_visible, num_hidden, device=self.device)
        self.rbm_ph = BinaryRBM(num_visible, num_hidden, device=self.device)
        self.num_visible = self.rbm_am.num_visible
        self.num_hidden  = self.rbm_am.num_hidden
        self.unitary_dict = unitary_dict if unitary_dict is not None else create_unitaries(self.device)

    # Hilbert utility
    def generate_hilbert_space(self, size=None, device=None):
        device = self.device if device is None else device
        size = self.rbm_am.num_visible if size is None else size
        return generate_hilbert_space(size, device=device)

    # ψ and normalized ψ
    def psi(self, v):
        vv = v.to(self.device, dtype=RTYPE)
        a  = (-self.rbm_am.effective_energy(vv)).exp().sqrt()
        ph = -0.5 * self.rbm_ph.effective_energy(vv)
        return torch.polar(a.to(RTYPE), ph.to(RTYPE)).to(CDTYPE)

    def psi_normalized(self, v):
        vv = v.to(self.device, dtype=RTYPE)
        E  = self.rbm_am.effective_energy(vv)
        a  = torch.exp(-0.5*E - 0.5*torch.logsumexp(-E, dim=0))
        ph = -0.5 * self.rbm_ph.effective_energy(vv)
        return torch.polar(a.to(RTYPE), ph.to(RTYPE)).to(CDTYPE)

    # rotations (vectorized; U[out,in])
    def _rotate_basis_state(self, basis, states):
        device = self.device
        basis_arr = np.array(list(basis))
        sites = np.where(basis_arr != "Z")[0]
        if sites.size == 0:
            v = states.unsqueeze(0)
            return torch.ones(v.shape[:-1], dtype=CDTYPE, device=device), v
        Uoi = torch.stack([self.unitary_dict[b] for b in basis_arr[sites]], dim=0).to(device=device)
        S = len(sites); B = states.shape[0]; C = 2**S
        combos_in = generate_hilbert_space(size=S, device=device)
        v = states.unsqueeze(0).repeat(C,1,1)
        v[:, :, sites] = combos_in.unsqueeze(1)
        out_idx = states[:, sites].round().long().T
        in_idx  = v[:, :, sites].round().long().permute(0,2,1)
        Uoi_exp = Uoi.unsqueeze(0).unsqueeze(2).expand(C,S,B,2,2)
        sel_out = torch.gather(Uoi_exp, 3, out_idx.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(C,S,B,1,2))
        sel     = torch.gather(sel_out, 4, in_idx.unsqueeze(-1).unsqueeze(-1))
        Ut = sel.squeeze(-1).squeeze(-1).prod(dim=1)
        return Ut, v

    @staticmethod
    def _states_to_index(states):
        powers = (2 ** (torch.arange(states.shape[-1], 0, -1, device=states.device) - 1)).to(states)
        return torch.matmul(states, powers)

    def rotate_psi_inner_prod(self, basis, states, psi=None, include_extras=False):
        Ut, v = self._rotate_basis_state(basis, states)
        if psi is None:
            psi_sel = self.psi(v.view(-1, v.shape[-1])).view(v.shape[0], v.shape[1])
        else:
            idx = self._states_to_index(v).long()
            psi_sel = psi.to(self.device, dtype=CDTYPE)[idx]
        Upsi_v = Ut * psi_sel
        Upsi   = Upsi_v.sum(dim=0)
        return (Upsi, Upsi_v, v) if include_extras else Upsi


    def rotate_psi(self, basis, states, psi=None):
        # Back-compat alias: returns ⟨basis|ψ⟩ over the given states
        return self.rotate_psi_inner_prod(basis, states, psi=psi)


    # grads (complex combo -> real params)
    def am_grads_complex(self, v):
        return self.rbm_am.effective_energy_gradient(v, reduce=False).to(CDTYPE)

    def ph_grads_complex(self, v):
        # multiply by +i → we project later; keeping i here is useful for clarity
        return (1j * self.rbm_ph.effective_energy_gradient(v, reduce=False)).to(CDTYPE)

    def rotated_gradient(self, basis, sample, inv_floor: float = INV_EPS):
        """Positive (rotated) gradients for amplitude and phase, mapping: phase uses Real after +i."""
        Upsi, Upsi_v, v = self.rotate_psi_inner_prod(basis, sample, include_extras=True)
        invU = Upsi.conj() / (Upsi.abs().pow(2).clamp_min(inv_floor))
        rg_am = torch.einsum("ib,ibg->bg", Upsi_v, self.am_grads_complex(v))
        rg_ph = torch.einsum("ib,ibg->bg", Upsi_v, self.ph_grads_complex(v))
        g_am = torch.real(torch.einsum("b,bg->g", invU, rg_am)).to(RTYPE)
        g_ph = torch.real(torch.einsum("b,bg->g", invU, rg_ph)).to(RTYPE)  # == PHASE-C if ph_grads_complex used raw
        return [g_am, g_ph]

    def rotated_phase_variants(self, basis, sample, inv_floor: float = INV_EPS):
        """Return four candidate *real* phase gradients from raw (non-i) phase energy grads.
        We compute S = einsum(invU, einsum(Upsi_v, raw_phase_grads)) and derive:
          A: +i, Real  ->  -Im(S)
          B: -i, Real  ->  +Im(S)
          C: +i, Imag  ->  +Re(S)
          D: -i, Imag  ->  -Re(S)
        """
        Upsi, Upsi_v, v = self.rotate_psi_inner_prod(basis, sample, include_extras=True)
        invU = Upsi.conj() / (Upsi.abs().pow(2).clamp_min(inv_floor))
        raw_ph = self.rbm_ph.effective_energy_gradient(v, reduce=False).to(CDTYPE)  # real → complex
        S_bg = torch.einsum("ib,ibg->bg", Upsi_v, raw_ph)  # (B,G)
        S_g = torch.einsum("b,bg->g", invU, S_bg)          # (G,)
        A = (-torch.imag(S_g)).to(RTYPE)  # +i, Real
        B = ( torch.imag(S_g)).to(RTYPE)  # -i, Real
        C = ( torch.real(S_g)).to(RTYPE)  # +i, Imag
        D = (-torch.real(S_g)).to(RTYPE)  # -i, Imag
        return {"PHASE A: +i, Real": A, "PHASE B: -i, Real": B, "PHASE C: +i, Imag": C, "PHASE D: -i, Imag": D}

    def gradient(self, samples, bases=None):
        grad = [torch.zeros(self.rbm_am.num_pars, dtype=RTYPE, device=self.device),
                torch.zeros(self.rbm_ph.num_pars, dtype=RTYPE, device=self.device)]
        if bases is None:
            grad[0] = self.rbm_am.effective_energy_gradient(samples)
            return grad
        if samples.dim() < 2:
            samples = samples.unsqueeze(0); bases = np.array(list(bases)).reshape(1,-1)
        uniq, idxs = np.unique(bases, axis=0, return_inverse=True)
        idxs = torch.tensor(idxs, device=samples.device)
        for i in range(uniq.shape[0]):
            basis = uniq[i, :]
            if np.any(basis != "Z"):
                g_am, g_ph = self.rotated_gradient(basis, samples[idxs==i, :])
                grad[0] += g_am; grad[1] += g_ph
            else:
                grad[0] += self.rbm_am.effective_energy_gradient(samples[idxs==i, :])
        return grad

    def positive_phase_gradients(self, samples_batch, bases_batch=None):
        grad = self.gradient(samples_batch, bases=bases_batch)
        return [g / float(samples_batch.shape[0]) for g in grad]

    def compute_batch_gradients(self, k, samples_batch, neg_batch, bases_batch=None):
        """Return:
        ([grad_am, grad_pos_ph], grad_pos_am, grad_pos_ph, (-grad_model_am))
        All grads are averaged by batch sizes, shape (G,)."""
        grad_pos_am, grad_pos_ph = self.positive_phase_gradients(samples_batch, bases_batch=bases_batch)
        vk = self.rbm_am.gibbs_steps(k, neg_batch)
        grad_model = self.rbm_am.effective_energy_gradient(vk) / float(neg_batch.shape[0])
        grad_am = grad_pos_am - grad_model
        return [grad_am, grad_pos_ph], grad_pos_am, grad_pos_ph, (-grad_model)

# --------------------- data I/O ---------------------
def load_data(tr_samples_path, tr_psi_path=None, tr_bases_path=None, bases_path=None):
    data = []
    data.append(torch.tensor(np.loadtxt(tr_samples_path, dtype="float32"), dtype=RTYPE, device=DEVICE))
    if tr_psi_path is not None:
        arr = np.loadtxt(tr_psi_path, dtype="float64")
        target_psi = torch.from_numpy(arr[:,0] + 1j*arr[:,1]).to(device=DEVICE, dtype=CDTYPE)
        data.append(target_psi)
    if tr_bases_path is not None:
        data.append(np.loadtxt(tr_bases_path, dtype=str))
    if bases_path is not None:
        data.append(np.loadtxt(bases_path, dtype=str, ndmin=1))
    return data

def try_load_or_synth():
    """Load Tonni’s w_state_* files if present, else synthesize a 3-qubit W."""
    files = {
        "train_path": "w_state_meas.txt",
        "train_bases_path": "w_state_basis.txt",
        "psi_path": "w_state_aug.txt",
        "bases_path": "w_state_bases.txt",
    }
    if all(os.path.exists(v) for v in files.values()):
        return load_data(files["train_path"], files["psi_path"], files["train_bases_path"], files["bases_path"])
    # fallback: 3-qubit W
    n=3
    hs = generate_hilbert_space(size=n)
    N  = hs.shape[0]
    idx = [1,2,4]
    psi = torch.zeros(N, dtype=CDTYPE, device=DEVICE); psi[idx] = (1/math.sqrt(3))
    train_samples = torch.bernoulli(torch.full((2000,n), 0.5, dtype=RTYPE, device=DEVICE))
    train_bases   = np.random.choice(list("XYZ"), size=(train_samples.shape[0], n))
    bases_eval    = np.array(["X"*n, "Y"*n, "Z"*n], dtype=str)
    return [train_samples, psi, train_bases, bases_eval]

# --------------------- metrics ---------------------
def fidelity(nn_state: ComplexWaveFunction, target: torch.Tensor, space=None, **kwargs) -> float:
    """F(ψ̂,ψ) = |⟨ψ|ψ̂⟩|^2. Returns python float."""
    space = nn_state.generate_hilbert_space() if space is None else space
    psi  = nn_state.psi_normalized(space).reshape(-1)
    targ = target.to(nn_state.device, dtype=CDTYPE).reshape(-1)
    return (cinner(targ, psi).abs()**2).item()

def KL(nn_state: ComplexWaveFunction, target: torch.Tensor, space=None, bases=None, **kwargs) -> float:
    """Mean KL over listed bases between rotated probs of target vs model."""
    space = nn_state.generate_hilbert_space() if space is None else space
    targ  = target.to(nn_state.device, dtype=CDTYPE)
    psi_n = nn_state.psi_normalized(space)
    val = 0.0
    basis_list = bases if isinstance(bases, (list, np.ndarray)) else [bases]
    for basis in basis_list:
        t_r = nn_state.rotate_psi(basis, space, psi=targ)
        p_r = nn_state.rotate_psi(basis, space, psi=psi_n)
        val += probs_kl((t_r.abs()**2), (p_r.abs()**2))
    return float((val / len(basis_list)).item())

# --------------------- batching ---------------------
def first_batch(nn_state: ComplexWaveFunction, train_samples: torch.Tensor, train_bases, pos_bs: int, neg_bs: int):
    B = train_samples.shape[0]
    pos_perm = torch.randperm(B, device=nn_state.device)
    pos_samples = train_samples[pos_perm]
    tb = np.asarray(train_bases)
    zmask = (tb == "Z").all(axis=1)
    z_samples = train_samples[torch.as_tensor(zmask, device=nn_state.device)]
    if len(z_samples) == 0:
        neg_perm = torch.randint(B, size=(neg_bs,), device=nn_state.device)
        neg_samples = train_samples[neg_perm]
    else:
        neg_perm = torch.randint(z_samples.shape[0], size=(neg_bs,), device=nn_state.device)
        neg_samples = z_samples[neg_perm]
    pos_batch = pos_samples[:pos_bs]
    neg_batch = neg_samples[:neg_bs]
    pos_bases = np.asarray(train_bases)[pos_perm.cpu().numpy()]
    pos_bases_batch = pos_bases[:pos_bs]
    return pos_batch, neg_batch, pos_bases_batch

# --------------------- aggregated probe ---------------------
@dataclass
class TrialResult:
    dF_dir: float
    dKL_dir: float
    variants: Dict[str, Tuple[float, float]]  # name -> (dF, dKL)
    phase_table: Dict[str, Tuple[float, float]]  # phase-variant -> (dF, dKL)

def run_trial(nn_state: ComplexWaveFunction, true_psi, space, bases_eval, train_samples, train_bases,
              k=10, pos_bs=100, neg_bs=100, step_scale=1e-3, eps=5e-4) -> TrialResult:
    pos_batch, neg_batch, bases_batch = first_batch(nn_state, train_samples.to(RTYPE), train_bases, pos_bs, neg_bs)

    (g_am, g_ph), g_pos_am, g_pos_ph, g_neg_am = nn_state.compute_batch_gradients(
        k, pos_batch, neg_batch, bases_batch=bases_batch
    )

    base_snapshot = clone_params([nn_state.rbm_am, nn_state.rbm_ph])

    with torch.no_grad():
        F0  = fidelity(nn_state, true_psi, space=space)
        KL0 = KL(nn_state, true_psi, space=space, bases=bases_eval if isinstance(bases_eval, (list,np.ndarray)) else [bases_eval])

    def eval_metrics():
        with torch.no_grad():
            return (fidelity(nn_state, true_psi, space=space),
                    KL(nn_state, true_psi, space=space, bases=bases_eval if isinstance(bases_eval, (list,np.ndarray)) else [bases_eval]))

    variants = [
        ("+g (train combo)", dict(am=g_am, ph=g_ph, scale=step_scale)),
        ("-g (sign flip)",   dict(am=-g_am, ph=-g_ph, scale=step_scale)),
        ("amp-only +",       dict(am=g_am, ph=None, scale=step_scale)),
        ("phase-only +",     dict(am=None, ph=g_ph, scale=step_scale)),
        ("pos-only +",       dict(am=g_pos_am, ph=g_pos_ph, scale=step_scale)),
        ("neg-only +",       dict(am=g_neg_am, ph=None,    scale=step_scale)),
    ]

    out_table: Dict[str, Tuple[float, float]] = {}
    for name, cfg in variants:
        restore_params([nn_state.rbm_am, nn_state.rbm_ph], base_snapshot)
        if cfg["am"] is not None: apply_step_into_params(nn_state.rbm_am, cfg["am"], cfg["scale"])
        if cfg["ph"] is not None: apply_step_into_params(nn_state.rbm_ph, cfg["ph"], cfg["scale"])
        F1, KL1 = eval_metrics()
        out_table[name] = (F1 - F0, KL1 - KL0)

    # phase sign grid (phase-only tiny steps)
    phase_table: Dict[str, Tuple[float, float]] = {}
    if isinstance(bases_batch, np.ndarray):
        phase_variants = nn_state.rotated_phase_variants(bases_batch[0], pos_batch)
        for label, gph in phase_variants.items():
            restore_params([nn_state.rbm_am, nn_state.rbm_ph], base_snapshot)
            apply_step_into_params(nn_state.rbm_ph, gph, step_scale)
            F1, KL1 = eval_metrics()
            phase_table[label] = (F1 - F0, KL1 - KL0)

    # directional finite-difference along normalized +g
    restore_params([nn_state.rbm_am, nn_state.rbm_ph], base_snapshot)
    am_dir = g_am / (g_am.norm() + 1e-12)
    ph_dir = g_ph / (g_ph.norm() + 1e-12)
    apply_step_into_params(nn_state.rbm_am, am_dir, +eps)
    apply_step_into_params(nn_state.rbm_ph, ph_dir, +eps)
    Fp, KLp = eval_metrics()
    restore_params([nn_state.rbm_am, nn_state.rbm_ph], base_snapshot)
    apply_step_into_params(nn_state.rbm_am, am_dir, -eps)
    apply_step_into_params(nn_state.rbm_ph, ph_dir, -eps)
    Fm, KLm = eval_metrics()
    dF_dir  = (Fp - Fm) / (2*eps)
    dKL_dir = (KLp - KLm) / (2*eps)
    restore_params([nn_state.rbm_am, nn_state.rbm_ph], base_snapshot)

    return TrialResult(float(dF_dir), float(dKL_dir), out_table, phase_table)

# --------------------- pretty print ---------------------
def fmt_float(x: float) -> str:
    return f"{x:+.6f}"

# ===================== EXTRA TESTS =====================
def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    an = a.norm(); bn = b.norm()
    if an.item() == 0.0 or bn.item() == 0.0: return float('nan')
    return float(torch.dot(a, b) / (an * bn))

@torch.no_grad()
def _metrics(nn_state, true_psi, space, bases_eval):
    F  = fidelity(nn_state, true_psi, space=space)
    KLv = KL(nn_state, true_psi, space=space, bases=bases_eval if isinstance(bases_eval, (list,np.ndarray)) else [bases_eval])
    return F, KLv

def _exact_model_grad_am(rbm_am: BinaryRBM):
    """Exact model <∂(-E)/∂θ> under p_model(v) ∝ exp(-E(v)) — small n only."""
    nv = rbm_am.num_visible
    if nv > 12:
        return None
    space = generate_hilbert_space(nv, device=rbm_am.device)
    E = rbm_am.effective_energy(space)
    logZ = torch.logsumexp(-E, dim=0)
    p = torch.exp(-E - logZ)
    geach = rbm_am.effective_energy_gradient(space, reduce=False)
    g_model = (p.unsqueeze(-1) * geach).sum(0)
    return g_model

def _conditioning_check(nn_state: ComplexWaveFunction, basis: str, pos_batch: torch.Tensor, inv_floor=INV_EPS):
    Upsi, Upsi_v, _ = nn_state.rotate_psi_inner_prod(basis, pos_batch, include_extras=True)
    a = Upsi.abs()
    tiny = torch.tensor([1e-8, 1e-6, 1e-4, 1e-3, 1e-2], device=a.device, dtype=a.dtype)
    stats = {
        "min|Uψ|": float(a.min().item()),
        "p1": float(torch.quantile(a, 0.01).item()),
        "median": float(torch.quantile(a, 0.5).item()),
        "p99": float(torch.quantile(a, 0.99).item()),
        "max": float(a.max().item()),
    }
    w = 1.0 / a.clamp_min(inv_floor)
    return stats, {"E[w]": float(w.mean().item()), "max w": float(w.max().item())}

def _linescan(nn_state: ComplexWaveFunction, base_snapshot, variants, eval_metrics, scales):
    """Return: dict[name] -> list[(scale, dF, dKL)], and best by ΔF with ΔKL≤0 tie-break."""
    F0, KL0 = eval_metrics()
    out = {}; best = {}
    for name, cfg in variants:
        curve = []
        for s in scales:
            restore_params([nn_state.rbm_am, nn_state.rbm_ph], base_snapshot)
            if cfg.get("am") is not None: apply_step_into_params(nn_state.rbm_am, cfg["am"], s)
            if cfg.get("ph") is not None: apply_step_into_params(nn_state.rbm_ph, cfg["ph"], s)
            F1, KL1 = eval_metrics()
            curve.append((s, F1 - F0, KL1 - KL0))
        pick = max(curve, key=lambda t: (t[1], -max(0.0, t[2])))
        out[name] = curve; best[name] = pick
    restore_params([nn_state.rbm_am, nn_state.rbm_ph], base_snapshot)
    return out, best

def run_extra_tests(nn_state: ComplexWaveFunction, true_psi, space, bases_eval, train_samples, train_bases,
                    k=10, pos_bs=100, neg_bs=100):
    print("\n===== EXTRA TESTS =====")
    pos_batch, neg_batch, bases_batch = first_batch(nn_state, train_samples.to(RTYPE), train_bases, pos_bs, neg_bs)

    (g_am, g_ph), g_pos_am, g_pos_ph, g_neg_am = nn_state.compute_batch_gradients(
        k, pos_batch, neg_batch, bases_batch=bases_batch
    )
    print("[GEOMETRY] norms:")
    print(f"  ||g_am||={g_am.norm().item():.4e}  ||g_ph||={g_ph.norm().item():.4e}  "
          f"||g_pos_am||={g_pos_am.norm().item():.4e}  ||g_pos_ph||={g_pos_ph.norm().item():.4e}  "
          f"||g_neg_am||={g_neg_am.norm().item():.4e}")
    print("[GEOMETRY] cosines:")
    print(f"  cos(g_am, g_pos_am)={_cos(g_am, g_pos_am):+.4f}  cos(g_am, g_neg_am)={_cos(g_am, g_neg_am):+.4f}  "
          f"cos(g_pos_am, g_neg_am)={_cos(g_pos_am, g_neg_am):+.4f}")

    # Uψ conditioning on first basis of this batch
    if isinstance(bases_batch, np.ndarray):
        basis_str = "".join(bases_batch[0])
        stats, wstats = _conditioning_check(nn_state, basis_str, pos_batch)
        print(f"[COND] basis='{basis_str}'  |Uψ| stats: {stats}   weights: {wstats}")

    # exact model negative gradient for amplitude (if feasible)
    g_model_exact = _exact_model_grad_am(nn_state.rbm_am)
    if g_model_exact is not None:
        F0, KL0 = _metrics(nn_state, true_psi, space, bases_eval)
        snap = clone_params([nn_state.rbm_am, nn_state.rbm_ph])
        g_am_exact = g_pos_am - g_model_exact
        restore_params([nn_state.rbm_am, nn_state.rbm_ph], snap)
        apply_step_into_params(nn_state.rbm_am, g_am_exact, 1e-3)
        F1, KL1 = _metrics(nn_state, true_psi, space, bases_eval)
        print(f"[EXACT-MODEL] step 1e-3 :: ΔF={F1-F0:+.6f}  ΔKL={KL1-KL0:+.6f}")
        print(f"[EXACT-MODEL] cos(g_am (CD), g_am (exact))={_cos(g_pos_am - g_neg_am, g_am_exact):+.4f}")
        restore_params([nn_state.rbm_am, nn_state.rbm_ph], snap)
    else:
        print("[EXACT-MODEL] skipped (nv>12)")

    # line scans
    snap = clone_params([nn_state.rbm_am, nn_state.rbm_ph])
    eval_metrics = lambda: _metrics(nn_state, true_psi, space, bases_eval)
    scales = [10.0**e for e in np.linspace(-5, -1, 9)]
    variants = [
        ("+g (train combo)", dict(am=g_am, ph=g_ph)),
        ("amp-only +",       dict(am=g_am, ph=None)),
        ("phase-only +",     dict(am=None, ph=g_ph)),
    ]
    curve, best = _linescan(nn_state, snap, variants, eval_metrics, scales)
    print("\n[LINESCAN] ΔF/ΔKL across step sizes:")
    for (name, _) in variants:
        rows = ", ".join([f"s={s:.0e}:ΔF={dF:+.4f}/ΔKL={dKL:+.4f}" for (s, dF, dKL) in curve[name]])
        print(f"  {name:16s} :: {rows}")
    print("[LINESCAN] best by ΔF (tie-break ΔKL≤0):")
    for nm, (s, dF, dKL) in best.items():
        print(f"  {nm:16s} :: s={s:.0e}  ΔF={dF:+.6f}  ΔKL={dKL:+.6f}")

# ===================== OPTIONAL: minimal phase-only regression =====================
def _phase_error(nn: ComplexWaveFunction, phi_true: torch.Tensor, space: torch.Tensor) -> float:
    theta_pred = (-0.5*nn.rbm_ph.effective_energy(space)).to(RTYPE)
    delta = theta_pred - phi_true
    delta = delta - delta[0]  # global phase
    sin_d = torch.sin(delta); cos_d = torch.cos(delta)
    angle = torch.atan2(sin_d, cos_d)
    return float(torch.mean(angle**2).item())

def _compute_phase_grad_mapping(nn: ComplexWaveFunction, basis: str, batch: torch.Tensor, mapping='A', inv_eps: float = INV_EPS):
    # S_g as in rotated_phase_variants; then pick projection
    Upsi, Upsi_v, v = nn.rotate_psi_inner_prod(basis, batch, include_extras=True)
    invU = Upsi.conj() / (Upsi.abs().pow(2).clamp_min(inv_eps))
    base = nn.rbm_ph.effective_energy_gradient(v, reduce=False).to(CDTYPE)
    S_bg = torch.einsum("ib,ibg->bg", Upsi_v, base)
    S_g  = torch.einsum("b,bg->g", invU, S_bg)
    if mapping.upper() == 'A':   # old: -Imag
        return (-torch.imag(S_g)).to(RTYPE)
    elif mapping.upper() == 'C': # Real
        return ( torch.real(S_g)).to(RTYPE)
    raise ValueError("mapping must be 'A' or 'C'.")

def _phase_dir_deriv(nn: ComplexWaveFunction, gvec: torch.Tensor, eps: float, phi_true: torch.Tensor, space: torch.Tensor) -> float:
    snap = clone_params([nn.rbm_am, nn.rbm_ph])
    d = gvec/(gvec.norm() + 1e-12)
    apply_step_into_params(nn.rbm_ph, d, +eps); fp = _phase_error(nn, phi_true, space)
    restore_params([nn.rbm_am, nn.rbm_ph], snap)
    apply_step_into_params(nn.rbm_ph, d, -eps); fm = _phase_error(nn, phi_true, space)
    restore_params([nn.rbm_am, nn.rbm_ph], snap)
    return float((fp - fm) / (2*eps))

def run_phase_minimal(nn: ComplexWaveFunction, nv: int):
    print("\n===== PHASE-ONLY MINIMAL REGRESSION =====")
    space = nn.generate_hilbert_space(nv)
    # Random uniform-phase target (amp = 1/√N)
    N = 2**nv
    phases = 2 * math.pi * torch.rand(N, device=DEVICE, dtype=RTYPE)
    phi_true = phases
    # sanity: rotated prob sums
    with torch.no_grad():
        psi_n = nn.psi_normalized(space)
        for b in ["X"*nv, "Y"*nv, "Z"*nv]:
            Ut, v = nn._rotate_basis_state(b, space)
            p = (Ut * psi_n[nn._states_to_index(v).long()]).abs().pow(2).sum(dim=0)
            print(f"[SANITY] {b}: sum p = {float(p.sum().item()):.12f}")

    B = int(os.getenv("NQT_BATCH", "128"))
    trials = int(os.getenv("NQT_TRIALS", "5"))
    eps = float(os.getenv("NQT_EPS", "1e-3"))
    steps = [float(x) for x in os.getenv("NQT_STEPS", "1e-05,3e-05,1e-04,3e-04,1e-03,3e-03,1e-02").split(",")]

    def random_basis(n):
        letters = np.array(list("XYZ"))
        basis = "".join(np.random.choice(letters, size=n))
        if set(basis) == {"Z"}:
            i = np.random.randint(n); basis = list(basis); basis[i] = np.random.choice(list("XY")); basis = "".join(basis)
        return basis

    cos_AC = []; dA=[]; dC=[]; bestA=[]; bestC=[]
    for t in range(trials):
        basis = random_basis(nv)
        idx = torch.randint(space.shape[0], (B,), device=DEVICE)
        batch = space[idx]
        gA = _compute_phase_grad_mapping(nn, basis, batch, mapping='A', inv_eps=INV_EPS)
        gC = _compute_phase_grad_mapping(nn, basis, batch, mapping='C', inv_eps=INV_EPS)
        cos = float(torch.dot(gA, gC)/(gA.norm()*gC.norm() + 1e-12)); cos_AC.append(cos)
        dA_t = _phase_dir_deriv(nn, gA, eps, phi_true, space); dA.append(dA_t)
        dC_t = _phase_dir_deriv(nn, gC, eps, phi_true, space); dC.append(dC_t)

        # line scans
        base_err = _phase_error(nn, phi_true, space)
        snap = clone_params([nn.rbm_am, nn.rbm_ph])
        deltasA=[]; deltasC=[]
        for s in steps:
            restore_params([nn.rbm_am, nn.rbm_ph], snap); apply_step_into_params(nn.rbm_ph, gA/(gA.norm()+1e-12), s)
            deltasA.append(_phase_error(nn, phi_true, space) - base_err)
            restore_params([nn.rbm_am, nn.rbm_ph], snap); apply_step_into_params(nn.rbm_ph, gC/(gC.norm()+1e-12), s)
            deltasC.append(_phase_error(nn, phi_true, space) - base_err)
        bestA.append(min(deltasA)); bestC.append(min(deltasC))
        print(f"[Trial {t+1}/{trials}] basis='{basis}'  cos(A,C)={cos:+.4f}  dA={dA_t:+.3e}  dC={dC_t:+.3e}")

    print("\n== PHASE MINIMAL AGGREGATES ==")
    mean_cos = float(np.mean(cos_AC))
    mean_dA  = float(np.mean(dA));  pA = float(np.mean([1.0 if x < 0 else 0.0 for x in dA]))
    mean_dC  = float(np.mean(dC));  pC = float(np.mean([1.0 if x < 0 else 0.0 for x in dC]))
    mean_bestA = float(np.mean(bestA)); mean_bestC = float(np.mean(bestC))
    print(f"⟨cos(A,C)⟩={mean_cos:+.4f}")
    print(f"⟨d(err)/dα⟩: A={mean_dA:+.3e} (P<0={pA:.2f})   C={mean_dC:+.3e} (P<0={pC:.2f})")
    print(f"Best Δ(err):  A={mean_bestA:+.3e}   C={mean_bestC:+.3e}")
    if mean_dC < 0 and (mean_bestC < mean_bestA or pC > pA + 0.2):
        print(" - PHASE-C (Real(S)) more often reduces phase error than A.")
    elif mean_dA < 0 and (mean_bestA < mean_bestC or pA > pC + 0.2):
        print(" - PHASE-A (-Imag(S)) more often reduces phase error than C; check rotation orientation or safeguards.")
    else:
        print(" - Mixed; both comparable under this minimal test.")

# =================== MAIN ===================
def main():
    print("===== NQT UNIFIED PROBE =====")
    print(f"torch: {torch.__version__} | device: {DEVICE} | RTYPE: {RTYPE} | CDTYPE: {CDTYPE}")

    # data
    train_samples, true_psi, train_bases, bases_eval = try_load_or_synth()
    nv = train_samples.shape[-1]; nh = nv
    nn_state = ComplexWaveFunction(nv, nh, create_unitaries(DEVICE))
    space = nn_state.generate_hilbert_space()

    # sanity: normalization & rotation sums
    with torch.no_grad():
        psi_n = nn_state.psi_normalized(space)
        sums = {
            "Z"*nv: float((psi_n.abs()**2).sum().item()),
            "X"*nv: float((nn_state.rotate_psi("X"*nv, space, psi=psi_n).abs()**2).sum().item()),
            "Y"*nv: float((nn_state.rotate_psi("Y"*nv, space, psi=psi_n).abs()**2).sum().item()),
        }
    print("[SANITY] prob sums (norm ψ):", {k: f"{v:.12f}" for k,v in sums.items()})

    # baseline metrics
    with torch.no_grad():
        F0  = fidelity(nn_state, true_psi, space=space)
        KL0 = KL(nn_state, true_psi, space=space, bases=bases_eval if isinstance(bases_eval, (list,np.ndarray)) else [bases_eval])
    print(f"[METRICS] BASE  :: Fidelity={F0:.6f}  KL={KL0:.6f}")

    # hyperparams
    trials = int(os.getenv("NQT_TRIALS", "5"))
    k = int(os.getenv("NQT_CD_K", "10"))
    pos_bs = int(os.getenv("NQT_POS_BS", "100"))
    neg_bs = int(os.getenv("NQT_NEG_BS", "100"))
    step_scale = float(os.getenv("NQT_STEP", "1e-3"))
    eps = float(os.getenv("NQT_EPS", "5e-4"))

    results: List[TrialResult] = []

    print("\n== PER-TRIAL STEP TABLES (ΔF, ΔKL) ==")
    for t in range(trials):
        tr = run_trial(nn_state, true_psi, space, bases_eval, train_samples, train_bases,
                       k=k, pos_bs=pos_bs, neg_bs=neg_bs, step_scale=step_scale, eps=eps)
        results.append(tr)
        print(f"\n[Trial {t+1}/{trials}] Directional: dF/dα={tr.dF_dir:+.6e}  dKL/dα={tr.dKL_dir:+.6e}")
        for name, (dF, dKL) in tr.variants.items():
            print(f"{name:16s} :: ΔF={fmt_float(dF)}   ΔKL={fmt_float(dKL)}")
        if tr.phase_table:
            print("-- PHASE SIGN GRID --")
            for label, (dF, dKL) in tr.phase_table.items():
                print(f"{label:18s} :: ΔF={fmt_float(dF)}   ΔKL={fmt_float(dKL)}")

    # aggregate
    mean_dF = np.mean([r.dF_dir for r in results])
    mean_dKL = np.mean([r.dKL_dir for r in results])
    signF = np.mean([1.0 if r.dF_dir < 0 else 0.0 for r in results])
    signKL = np.mean([1.0 if r.dKL_dir < 0 else 0.0 for r in results])

    keys = list(results[0].variants.keys())
    agg = {k: (np.mean([r.variants[k][0] for r in results]),
               np.mean([r.variants[k][1] for r in results])) for k in keys}

    phase_agg = {}
    best_phase = None
    if results[0].phase_table:
        phase_keys = list(results[0].phase_table.keys())
        phase_agg = {k: (np.mean([r.phase_table[k][0] for r in results]),
                         np.mean([r.phase_table[k][1] for r in results])) for k in phase_keys}
        def phase_score(item):
            dF, dKL = item[1]
            return (dF, -abs(max(0.0, dKL)))
        best_phase = max(phase_agg.items(), key=phase_score)

    print("\n== AGGREGATES OVER TRIALS ==")
    print(f"<dir>  mean dF/dα={mean_dF:+.6e}   mean dKL/dα={mean_dKL:+.6e}   P[dF<0]={signF:.2f}  P[dKL<0]={signKL:.2f}")
    for kkey in keys:
        dF, dKL = agg[kkey]
        print(f"{kkey:16s} :: ⟨ΔF⟩={fmt_float(dF)}   ⟨ΔKL⟩={fmt_float(dKL)}")
    if phase_agg:
        print("-- PHASE SIGN GRID (aggregated) --")
        for kkey, (dF, dKL) in phase_agg.items():
            print(f"{kkey:18s} :: ⟨ΔF⟩={fmt_float(dF)}   ⟨ΔKL⟩={fmt_float(dKL)}")

    # verdict
    tolF  = float(os.getenv("NQT_TOLF", "1e-4"))
    tolKL = float(os.getenv("NQT_TOLKL", "1e-3"))

    conclusion = []
    if mean_dF < -tolF and mean_dKL < -tolKL and signF >= 0.7 and signKL >= 0.7:
        conclusion.append("Training direction consistently reduces KL but *also* reduces Fidelity → your loss aligns with measurement-KL, not state fidelity.")
    elif mean_dF >  tolF and mean_dKL >  tolKL and (1.0 - signF) >= 0.7 and (1.0 - signKL) >= 0.7:
        conclusion.append("Training direction consistently worsens both Fidelity and KL → likely gradient sign/mapping bug.")
    elif mean_dF >  tolF and mean_dKL < -tolKL and (1.0 - signF) >= 0.7 and signKL >= 0.7:
        conclusion.append("Training direction tends to improve both metrics → ok.")
    else:
        conclusion.append("Mixed signals; inspect amp-only vs phase-only and pos-only vs neg-only aggregates above.")

    amp_dF, _ = agg["amp-only +"]
    phs_dF, _ = agg["phase-only +"]
    pos_dF, _ = agg["pos-only +"]
    neg_dF, _ = agg["neg-only +"]

    detail = []
    if amp_dF > tolF and phs_dF < -tolF:
        detail.append("phase-only step drives ↓F; amplitude-only benign/positive → check rotated phase mapping.")
    elif amp_dF < -tolF and phs_dF > tolF:
        detail.append("amplitude-only step drives ↓F; phase-only benign/positive → check amplitude/neg-phase coupling.")
    if pos_dF < -tolF and neg_dF >= -tolF:
        detail.append("positive (data/rotated) term lowers F; negative (model/CD) is not the culprit.")
    elif neg_dF < -tolF and pos_dF >= -tolF:
        detail.append("negative (model/CD) term lowers F; contrastive estimator too strong or k too small.")

    print("\nCONCLUSION:")
    print(" - " + " ".join(conclusion))
    for d in detail:
        print(" - " + d)
    if best_phase is not None:
        label, (dF, dKL) = best_phase
        print(f" - Phase-sign recommendation: {label} (⟨ΔF⟩={fmt_float(dF)}, ⟨ΔKL⟩={fmt_float(dKL)})")

    # extra diagnostics (always useful)
    run_extra_tests(nn_state, true_psi, space, bases_eval, train_samples, train_bases,
                    k=k, pos_bs=pos_bs, neg_bs=neg_bs)

    # optional minimal phase-only regression
    if int(os.getenv("NQT_RUN_PHASE_MIN", "0")) == 1:
        run_phase_minimal(nn_state, nv)

    print("===== DONE =====")

if __name__ == "__main__":
    main()
