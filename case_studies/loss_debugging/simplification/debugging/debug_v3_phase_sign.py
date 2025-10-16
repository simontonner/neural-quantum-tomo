# nqt_conclusive_probe.py
# One-shot, conclusive diagnostic for "fidelity not increasing" in complex NQS tomography
# - Pure complex forward (torch.cdouble); RBM params are real (torch.double)
# - Splits gradients into positive (rotated data) and negative (model/CD) parts
# - Tests multiple step variants (+g, -g, amp-only, phase-only, pos-only, neg-only)
# - **NEW** Phase-sign grid: tries {+i/-i} × {Real/Imag} mappings for the phase block and recommends the best
# - **NEW** Robust to F-shadowing: uses torch.nn.functional as Fnn and warns if a global 'F' shadows it
# - Finite-difference directional derivatives along training direction
# - Repeats across several random mini-batches and aggregates results
# - Prints a single "CONCLUSION" line with a reasoned verdict, plus a phase-sign recommendation

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn  # <- safe alias (no F-shadowing)
from torch.distributions.utils import probs_to_logits
from dataclasses import dataclass
from typing import Tuple, List, Dict

# --------------------- core setup ---------------------
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RTYPE   = torch.double
CDTYPE  = torch.cdouble
INV_EPS = 1e-6
SEED    = 1234

torch.set_default_dtype(RTYPE)
torch.manual_seed(SEED)
np.random.seed(SEED)

# --------------------- utils ---------------------
def cinner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.conj() * y)

def generate_hilbert_space(size: int, device=DEVICE) -> torch.Tensor:
    if size <= 0:
        return torch.zeros(0, 0, dtype=RTYPE, device=device)
    ar = torch.arange(2**size, device=device, dtype=torch.long)
    shifts = torch.arange(size-1, -1, -1, device=device, dtype=torch.long)
    return ((ar.unsqueeze(1) >> shifts) & 1).to(RTYPE)

def probs_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # KL(p||q) with logits (expects probs normalized)
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
def create_dict(**kwargs):
    s2 = np.sqrt(2.0)
    X = torch.tensor([[1,  1],
                      [1, -1]], dtype=CDTYPE, device=DEVICE) / s2
    Y = torch.tensor([[1,  1j],
                      [1, -1j]], dtype=CDTYPE, device=DEVICE) / s2
    Z = torch.eye(2, dtype=CDTYPE, device=DEVICE)
    d = {"X": X, "Y": Y, "Z": Z}
    for name, m in kwargs.items():
        m = (m.clone().detach() if isinstance(m, torch.Tensor) else torch.tensor(m))
        d[name] = m.to(dtype=CDTYPE, device=DEVICE)
    return d

# --------------------- RBM (real params) ---------------------
class BinaryRBM(nn.Module):
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
                                     / np.sqrt(self.num_visible)), requires_grad=False)
        self.visible_bias = nn.Parameter(torch.zeros(self.num_visible, device=self.device, dtype=RTYPE),
                                         requires_grad=False)
        self.hidden_bias  = nn.Parameter(torch.zeros(self.num_hidden,  device=self.device, dtype=RTYPE),
                                         requires_grad=False)

    def effective_energy(self, v):
        unsq = False
        if v.dim() < 2:
            v = v.unsqueeze(0); unsq = True
        v = v.to(self.weights)
        vb = torch.matmul(v, self.visible_bias)
        hb = Fnn.softplus(Fnn.linear(v, self.weights, self.hidden_bias)).sum(-1)  # <- Fnn
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
        v = (initial_state if overwrite else initial_state.clone()).to(self.weights)
        h = torch.zeros(*v.shape[:-1], self.num_hidden, device=self.device, dtype=RTYPE)
        for _ in range(k):
            self.sample_h_given_v(v, out=h)
            self.sample_v_given_h(h, out=v)
        return v

    def effective_energy_gradient(self, v, reduce=True):
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
    def __init__(self, num_visible, num_hidden=None, unitary_dict=None):
        self.device = DEVICE
        self.rbm_am = BinaryRBM(num_visible, num_hidden, device=self.device)
        self.rbm_ph = BinaryRBM(num_visible, num_hidden, device=self.device)
        self.num_visible = self.rbm_am.num_visible
        self.num_hidden  = self.rbm_am.num_hidden
        self.unitary_dict = unitary_dict if unitary_dict is not None else create_dict()

    # handy
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

    # rotations
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

    def rotate_psi(self, basis, space, psi=None):
        return self.rotate_psi_inner_prod(basis, space, psi=psi)

    # grads (complex combo -> real params)
    def am_grads_complex(self, v):
        return self.rbm_am.effective_energy_gradient(v, reduce=False).to(CDTYPE)

    def ph_grads_complex(self, v):
        return (1j * self.rbm_ph.effective_energy_gradient(v, reduce=False)).to(CDTYPE)

    def rotated_gradient(self, basis, sample, inv_floor: float = INV_EPS):
        Upsi, Upsi_v, v = self.rotate_psi_inner_prod(basis, sample, include_extras=True)
        invU = Upsi.conj() / (Upsi.abs().pow(2).clamp_min(inv_floor))
        rg_am = torch.einsum("ib,ibg->bg", Upsi_v, self.am_grads_complex(v))
        rg_ph = torch.einsum("ib,ibg->bg", Upsi_v, self.ph_grads_complex(v))
        g_am = torch.real(torch.einsum("b,bg->g", invU, rg_am)).to(RTYPE)
        g_ph = torch.real(torch.einsum("b,bg->g", invU, rg_ph)).to(RTYPE)
        return [g_am, g_ph]

    def rotated_phase_variants(self, basis, sample, inv_floor: float = INV_EPS):
        """Return four candidate *real* phase gradients from raw (non-i) phase energy grads.
        We compute S = einsum(invU, einsum(Upsi_v, raw_phase_grads)) and derive:
          A: +i, Real  ->  -Im(S)
          B: -i, Real  ->  +Im(S)
          C: +i, Imag  ->  +Re(S)
          D: -i, Imag  ->  -Re(S)
        Labels mirror your phase-sign probe readout.
        """
        # inner products as in rotated_gradient, but with *raw* (non-i) grads
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
    psi = torch.zeros(N, dtype=CDTYPE, device=DEVICE); psi[idx] = (1/np.sqrt(3))
    train_samples = torch.bernoulli(torch.full((2000,n), 0.5, dtype=RTYPE, device=DEVICE))
    train_bases   = np.random.choice(list("XYZ"), size=(train_samples.shape[0], n))
    bases_eval    = np.array(["X"*n, "Y"*n, "Z"*n], dtype=str)
    return [train_samples, psi, train_bases, bases_eval]

# --------------------- metrics ---------------------
def fidelity(nn_state: ComplexWaveFunction, target: torch.Tensor, space=None, **kwargs) -> float:
    space = nn_state.generate_hilbert_space() if space is None else space
    psi  = nn_state.psi_normalized(space).reshape(-1)
    targ = target.to(nn_state.device, dtype=CDTYPE).reshape(-1)
    return (cinner(targ, psi).abs()**2).item()


def KL(nn_state: ComplexWaveFunction, target: torch.Tensor, space=None, bases=None, **kwargs) -> float:
    space = nn_state.generate_hilbert_space() if space is None else space
    targ  = target.to(nn_state.device, dtype=CDTYPE)
    psi_n = nn_state.psi_normalized(space)
    val = 0.0
    for basis in bases:
        t_r = nn_state.rotate_psi(basis, space, psi=targ)
        p_r = nn_state.rotate_psi(basis, space, psi=psi_n)
        val += probs_kl((t_r.abs()**2), (p_r.abs()**2))
    return float((val / len(bases)).item())

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
    phase_variants = nn_state.rotated_phase_variants(bases_batch[0], pos_batch) if isinstance(bases_batch, np.ndarray) else {}
    # If bases_batch is an array of per-sample bases, choose the first row's basis string
    if isinstance(bases_batch, np.ndarray):
        # evaluate tiny steps per variant
        phase_table: Dict[str, Tuple[float, float]] = {}
        for label, gph in phase_variants.items():
            restore_params([nn_state.rbm_am, nn_state.rbm_ph], base_snapshot)
            apply_step_into_params(nn_state.rbm_ph, gph, step_scale)  # same scale as other variants
            F1, KL1 = eval_metrics()
            phase_table[label] = (F1 - F0, KL1 - KL0)
    else:
        phase_table = {}

    # directional finite-difference along normalized +g
    restore_params([nn_state.rbm_am, nn_state.rbm_ph], base_snapshot)
    am_dir = g_am / (g_am.norm() + 1e-12)
    ph_dir = g_ph / (g_ph.norm() + 1e-12)

    # F(+eps)
    apply_step_into_params(nn_state.rbm_am, am_dir, eps)
    apply_step_into_params(nn_state.rbm_ph, ph_dir, eps)
    Fp, KLp = eval_metrics()

    # F(-eps)
    restore_params([nn_state.rbm_am, nn_state.rbm_ph], base_snapshot)
    apply_step_into_params(nn_state.rbm_am, am_dir, -eps)
    apply_step_into_params(nn_state.rbm_ph, ph_dir, -eps)
    Fm, KLm = eval_metrics()

    dF_dir  = (Fp - Fm) / (2*eps)
    dKL_dir = (KLp - KLm) / (2*eps)

    # restore to base for caller
    restore_params([nn_state.rbm_am, nn_state.rbm_ph], base_snapshot)

    return TrialResult(float(dF_dir), float(dKL_dir), out_table, phase_table)

# --------------------- pretty print ---------------------
def fmt_float(x: float) -> str:
    return f"{x:+.6f}"

def warn_if_F_shadowed():
    # If a global 'F' exists and doesn't look like a module with softplus, warn.
    if 'F' in globals():
        Fobj = globals()['F']
        if not hasattr(Fobj, 'softplus'):
            print("[WARN] Global name 'F' exists and lacks 'softplus' → likely shadowing torch.nn.functional. Using Fnn internally.")

def main():
    print("===== NQT CONCLUSIVE PROBE =====")
    print(f"torch: {torch.__version__} | device: {DEVICE} | RTYPE: {RTYPE} | CDTYPE: {CDTYPE}")

    warn_if_F_shadowed()

    # data
    train_samples, true_psi, train_bases, bases_eval = try_load_or_synth()
    nv = train_samples.shape[-1]; nh = nv
    nn_state = ComplexWaveFunction(nv, nh, create_dict())
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

    # hyperparams (feel free to tweak from CLI via env or edit)
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
    signF = np.mean([1.0 if r.dF_dir < 0 else 0.0 for r in results])  # fraction with dF<0
    signKL = np.mean([1.0 if r.dKL_dir < 0 else 0.0 for r in results])

    # aggregate variants
    keys = list(results[0].variants.keys())
    agg = {k: (np.mean([r.variants[k][0] for r in results]),
               np.mean([r.variants[k][1] for r in results])) for k in keys}

    # aggregate phase grid
    phase_keys = None
    if results[0].phase_table:
        phase_keys = list(results[0].phase_table.keys())
        phase_agg = {k: (np.mean([r.phase_table[k][0] for r in results]),
                         np.mean([r.phase_table[k][1] for r in results])) for k in phase_keys}
        # pick best by ΔF primary, ΔKL secondary (tie-breaker: prefer ΔKL ≤ 0)
        def phase_score(item):
            dF, dKL = item[1]
            return (dF, -abs(max(0.0, dKL)))
        best_phase = max(phase_agg.items(), key=phase_score)
    else:
        phase_agg = {}
        best_phase = None

    print("\n== AGGREGATES OVER TRIALS ==")
    print(f"<dir>  mean dF/dα={mean_dF:+.6e}   mean dKL/dα={mean_dKL:+.6e}   P[dF<0]={signF:.2f}  P[dKL<0]={signKL:.2f}")
    for k in keys:
        dF, dKL = agg[k]
        print(f"{k:16s} :: ⟨ΔF⟩={fmt_float(dF)}   ⟨ΔKL⟩={fmt_float(dKL)}")
    if phase_keys:
        print("-- PHASE SIGN GRID (aggregated) --")
        for k in phase_keys:
            dF, dKL = phase_agg[k]
            print(f"{k:18s} :: ⟨ΔF⟩={fmt_float(dF)}   ⟨ΔKL⟩={fmt_float(dKL)}")

    # verdict logic
    tolF  = float(os.getenv("NQT_TOLF", "1e-4"))
    tolKL = float(os.getenv("NQT_TOLKL", "1e-3"))

    conclusion = []
    if mean_dF < -tolF and mean_dKL < -tolKL and signF >= 0.7 and signKL >= 0.7:
        conclusion.append("Training direction consistently reduces KL but *also* reduces Fidelity → your loss aligns with measurement-KL, not state fidelity.")
    elif mean_dF > tolF and mean_dKL > tolKL and (1.0 - signF) >= 0.7 and (1.0 - signKL) >= 0.7:
        conclusion.append("Training direction consistently worsens both Fidelity and KL → likely gradient sign/mapping bug.")
    elif mean_dF > tolF and mean_dKL < -tolKL and (1.0 - signF) >= 0.7 and signKL >= 0.7:
        conclusion.append("Training direction tends to improve both metrics → ok.")
    else:
        conclusion.append("Mixed signals; inspect amp-only vs phase-only and pos-only vs neg-only aggregates below.")

    # pinpoint which block hurts Fidelity
    amp_dF, _ = agg["amp-only +"]
    phs_dF, _ = agg["phase-only +"]
    pos_dF, _ = agg["pos-only +"]
    neg_dF, _ = agg["neg-only +"]

    detail = []
    if amp_dF > tolF and phs_dF < -tolF:
        detail.append("phase-only step drives ↓F; amplitude-only is benign/positive → check rotated phase gradient mapping.")
    elif amp_dF < -tolF and phs_dF > tolF:
        detail.append("amplitude-only step drives ↓F; phase-only is benign/positive → check amplitude/neg-phase coupling.")
    if pos_dF < -tolF and neg_dF >= -tolF:
        detail.append("positive (data/rotated) term lowers Fidelity; negative (model/CD) term is not the culprit.")
    elif neg_dF < -tolF and pos_dF >= -tolF:
        detail.append("negative (model/CD) term lowers Fidelity; contrastive estimator may be too strong or k too small.")

    print("\nCONCLUSION:")
    print(" - " + " ".join(conclusion))
    for d in detail:
        print(" - " + d)
    if best_phase is not None:
        label, (dF, dKL) = best_phase
        print(f" - Phase-sign recommendation: {label} (⟨ΔF⟩={fmt_float(dF)}, ⟨ΔKL⟩={fmt_float(dKL)})")

    print("===== DONE =====")

if __name__ == "__main__":
    main()
