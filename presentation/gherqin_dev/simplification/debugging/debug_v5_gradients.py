# nqt_stability_exploratory_tests.py
# Self-contained diagnostics for pure-cdouble phase learning stability.
# Focus: denominator conditioning, basis mix, gradient variance, and step sensitivity.

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------
# Config
# -----------------------------------
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RTYPE   = torch.double
CDTYPE  = torch.cdouble
SEED    = int(os.getenv("NQT_SEED", "1234"))
torch.manual_seed(SEED); np.random.seed(SEED)

# Problem size & probe controls (tweak as needed)
NV         = int(os.getenv("NQT_NV", "4"))
BATCH      = int(os.getenv("NQT_BATCH", "256"))
TRIALS     = int(os.getenv("NQT_TRIALS", "12"))
EPS_DIR    = float(os.getenv("NQT_EPS_DIR", "1e-3"))      # finite-diff dir step (params-space)
LINE_STEPS = [float(x) for x in os.getenv("NQT_STEPS", "1e-05,3e-05,1e-04,3e-04,1e-03,3e-03,1e-02").split(",")]
INV_SWEEP  = [float(x) for x in os.getenv("NQT_INV_SWEEP", "1e-12,1e-9,3e-9,1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5").split(",")]
TAU_LIST   = [float(x) for x in os.getenv("NQT_TAU_LIST", "1e-5,3e-5,1e-4,3e-4,1e-3").split(",")]
K_BASIS    = list(range(1, NV+1))  # #non-Z axes in a basis

# -----------------------------------
# Hilbert tools & unitaries (cdouble)
# -----------------------------------
def generate_hilbert_space(size: int, device=DEVICE, dtype=RTYPE) -> torch.Tensor:
    ar = torch.arange(2**size, device=device, dtype=torch.long)
    shifts = torch.arange(size-1, -1, -1, device=device, dtype=torch.long)
    return ((ar.unsqueeze(1) >> shifts) & 1).to(dtype)

def create_unitaries(device=DEVICE):
    s2 = np.sqrt(2.0)
    X = torch.tensor([[1,  1],
                      [1, -1]], dtype=CDTYPE, device=device) / s2
    Y = torch.tensor([[1,  1j],
                      [1, -1j]], dtype=CDTYPE, device=device) / s2
    Z = torch.eye(2, dtype=CDTYPE, device=device)
    return {"X": X, "Y": Y, "Z": Z}

def rotate_basis_state(basis, states, unitaries, device=DEVICE):
    basis_arr = np.array(list(basis))
    sites = np.where(basis_arr != "Z")[0]
    if sites.size == 0:
        v = states.unsqueeze(0)
        Ut = torch.ones(v.shape[:-1], dtype=CDTYPE, device=device)
        return Ut, v
    Uoi = torch.stack([unitaries[b] for b in basis_arr[sites]], dim=0).to(device=device)  # (S,2,2)
    S = len(sites); B = states.shape[0]; C = 2**S
    combos = generate_hilbert_space(S, device=device, dtype=states.dtype)
    v = states.unsqueeze(0).repeat(C, 1, 1)
    v[:, :, sites] = combos.unsqueeze(1)
    # OUT then IN, with rounding before casting to int
    out_idx = states[:, sites].round().long().T
    in_idx  = v[:, :, sites].round().long().permute(0, 2, 1)
    Uoi_exp = Uoi.unsqueeze(0).unsqueeze(2).expand(C, S, B, 2, 2)
    sel_out = torch.gather(Uoi_exp, 3, out_idx.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(C, S, B, 1, 2))
    sel     = torch.gather(sel_out, 4, in_idx.unsqueeze(-1).unsqueeze(-1))
    Ut = sel.squeeze(-1).squeeze(-1).prod(dim=1)  # (C,B)
    return Ut, v

def states_to_index(states):
    powers = (2 ** (torch.arange(states.shape[-1], 0, -1, device=states.device) - 1)).to(states)
    return torch.matmul(states, powers)

# -----------------------------------
# RBM (real) & complex ψ (cdouble)
# -----------------------------------
class RBM(nn.Module):
    def __init__(self, nv, nh=None, device=DEVICE, dtype=RTYPE):
        super().__init__()
        self.nv = int(nv); self.nh = int(nh) if nh else int(nv)
        self.device = device; self.dtype=dtype
        self.weights = nn.Parameter(torch.randn(self.nh, self.nv, device=device, dtype=dtype) / math.sqrt(self.nv),
                                    requires_grad=False)
        self.vb = nn.Parameter(torch.zeros(self.nv, device=device, dtype=dtype), requires_grad=False)
        self.hb = nn.Parameter(torch.zeros(self.nh, device=device, dtype=dtype), requires_grad=False)
        self.num_pars = self.nh*self.nv + self.nv + self.nh

    def effective_energy(self, v):
        unsq = False
        if v.dim() < 2: v = v.unsqueeze(0); unsq = True
        v = v.to(self.weights)
        vb = torch.matmul(v, self.vb)
        hb = F.softplus(F.linear(v, self.weights, self.hb)).sum(-1)
        out = -(vb + hb)
        return out.squeeze(0) if unsq else out

    def prob_h_given_v(self, v):
        unsq = False
        if v.dim() < 2: v = v.unsqueeze(0); unsq = True
        res = torch.matmul(v, self.weights.data.t()).add_(self.hb.data).sigmoid_().clamp_(0, 1)
        return res.squeeze(0) if unsq else res

    def effective_energy_gradient(self, v, reduce=True):
        v = (v.unsqueeze(0) if v.dim() < 2 else v).to(self.weights)
        prob = self.prob_h_given_v(v)
        if reduce:
            Wg  = -torch.matmul(prob.transpose(0,-1), v)
            vbg = -torch.sum(v,0)
            hbg = -torch.sum(prob,0)
            return torch.cat([Wg.reshape(-1), vbg, hbg])
        else:
            Wg  = -torch.einsum("...j,...k->...jk", prob, v)
            vbg = -v
            hbg = -prob
            return torch.cat([Wg.view(*v.shape[:-1], -1), vbg, hbg], dim=-1)

def split_vec(vec, rbm):
    nh, nv = rbm.nh, rbm.nv
    nW = nh*nv
    W  = vec[:nW].view(nh, nv)
    vb = vec[nW:nW+nv]
    hb = vec[nW+nv:]
    return W, vb, hb

def apply_step(rbm, gvec, step):
    Wg, vbg, hbg = split_vec(gvec, rbm)
    with torch.no_grad():
        rbm.weights.add_(-step * Wg)
        rbm.vb.add_(-step * vbg)
        rbm.hb.add_(-step * hbg)

class ComplexWF:
    def __init__(self, nv, nh=None, device=DEVICE):
        self.device = device
        self.rbm_am = RBM(nv, nh, device=device)
        self.rbm_ph = RBM(nv, nh, device=device)
        self.unitaries = create_unitaries(device=device)

    def psi(self, v):
        vv = v.to(self.device, dtype=RTYPE)
        a  = (-self.rbm_am.effective_energy(vv)).exp().sqrt()
        ph = -0.5 * self.rbm_ph.effective_energy(vv)
        return torch.polar(a.to(RTYPE), ph.to(RTYPE)).to(CDTYPE)

    def psi_normalized(self, v):
        vv = v.to(self.rbm_am.weights)
        E  = self.rbm_am.effective_energy(vv)
        a  = torch.exp(-0.5*E - 0.5*torch.logsumexp(-E, dim=0))
        ph = -0.5 * self.rbm_ph.effective_energy(vv)
        return torch.polar(a.to(RTYPE), ph.to(RTYPE)).to(CDTYPE)

    def phase(self, v):
        vv = v.to(self.device, dtype=RTYPE)
        return -0.5 * self.rbm_ph.effective_energy(vv)

    def rotate_psi_inner(self, basis, states, psi=None, include_extras=False):
        Ut, v = rotate_basis_state(basis, states, self.unitaries, device=self.device)
        if psi is None:
            psi_sel = self.psi(v.view(-1, v.shape[-1])).view(v.shape[0], v.shape[1])
        else:
            idx = states_to_index(v).long()
            psi_sel = psi.to(self.device, dtype=CDTYPE)[idx]
        Upsi_v = Ut * psi_sel
        Upsi   = Upsi_v.sum(dim=0)
        return (Upsi, Upsi_v, v) if include_extras else Upsi

    def ph_base_grads(self, v):
        return self.rbm_ph.effective_energy_gradient(v, reduce=False)  # real

# -----------------------------------
# Core gradient mapping (old vs C)
# -----------------------------------
def compute_phase_grad(nn: ComplexWF, basis: str, samples_batch: torch.Tensor, mapping='C', inv_eps: float = 1e-6):
    Upsi, Upsi_v, v = nn.rotate_psi_inner(basis, samples_batch, include_extras=True)
    invU = Upsi.conj() / (Upsi.abs().pow(2).clamp_min(inv_eps))
    base = nn.ph_base_grads(v).to(CDTYPE)                 # (C,B,G)
    S_bg = torch.einsum("ib,ibg->bg", Upsi_v, base)       # (B,G)
    T_g  = torch.einsum("b,bg->g", invU, S_bg)            # (G,)
    if mapping == 'old':
        return (-torch.imag(T_g)).to(RTYPE)
    elif mapping == 'C':
        return ( torch.real(T_g)).to(RTYPE)
    else:
        raise ValueError("mapping must be 'old' or 'C'.")

# -----------------------------------
# Phase error metric (global-phase invariant)
# -----------------------------------
def phase_error(nn: ComplexWF, phi_true: torch.Tensor, space: torch.Tensor) -> float:
    theta_pred = nn.phase(space)
    delta = theta_pred - phi_true
    delta = delta - delta[0]  # remove global shift
    sin_d = torch.sin(delta); cos_d = torch.cos(delta)
    angle = torch.atan2(sin_d, cos_d)
    return float(torch.mean(angle**2).item())

def random_uniform_phase_state(nv: int, device=DEVICE):
    N = 2**nv
    phases = 2 * math.pi * torch.rand(N, device=device, dtype=RTYPE)
    amp = torch.full((N,), 1/math.sqrt(N), device=device, dtype=RTYPE)
    psi = torch.polar(amp.to(RTYPE), phases.to(RTYPE)).to(CDTYPE)
    return psi, phases

def snapshot_params(rbm: RBM):
    return tuple(p.detach().clone() for p in (rbm.weights, rbm.vb, rbm.hb))

def restore_params(rbm: RBM, snap):
    with torch.no_grad():
        rbm.weights.copy_(snap[0]); rbm.vb.copy_(snap[1]); rbm.hb.copy_(snap[2])

def line_scan(nn: ComplexWF, dir_vec: torch.Tensor, steps, phi_true, space):
    snap = snapshot_params(nn.rbm_ph)
    base_err = phase_error(nn, phi_true, space)
    d = dir_vec / (dir_vec.norm() + 1e-12)
    deltas = []
    for s in steps:
        restore_params(nn.rbm_ph, snap)
        apply_step(nn.rbm_ph, d, s)
        deltas.append(phase_error(nn, phi_true, space) - base_err)
    restore_params(nn.rbm_ph, snap)
    return base_err, deltas

def directional_derivative(nn: ComplexWF, dir_vec: torch.Tensor, eps: float, phi_true, space):
    snap = snapshot_params(nn.rbm_ph)
    d = dir_vec / (dir_vec.norm() + 1e-12)
    restore_params(nn.rbm_ph, snap); apply_step(nn.rbm_ph, d, +eps); fp = phase_error(nn, phi_true, space)
    restore_params(nn.rbm_ph, snap); apply_step(nn.rbm_ph, d, -eps); fm = phase_error(nn, phi_true, space)
    restore_params(nn.rbm_ph, snap)
    return float((fp - fm) / (2*eps))

def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.dot(a, b) / (a.norm()*b.norm() + eps))

# -----------------------------------
# Exploratory tests
# -----------------------------------
def prob_sum_sanity(nn, space):
    with torch.no_grad():
        psi_n = nn.psi_normalized(space)
        sums = {}
        for b in ("X"*NV, "Y"*NV, "Z"*NV):
            Ut, v = rotate_basis_state(b, space, nn.unitaries, device=nn.device)
            p = (Ut * psi_n[states_to_index(v).long()]).abs().pow(2).sum(dim=0)
            sums[b] = float(p.sum().item())
    print("[SANITY] Prob sums (should be ~1.0):", {k: f"{v:.12f}" for k,v in sums.items()})

def denom_tail_survey(nn, space, trials=TRIALS, batch=BATCH):
    print("\n== TEST A: Denominator tails |Uψ| distribution ==")
    vals = []
    for t in range(trials):
        basis = random_basis(NV)
        idx = torch.randint(space.shape[0], size=(batch,), device=DEVICE)
        batch_states = space[idx]
        with torch.no_grad():
            Upsi = nn.rotate_psi_inner(basis, batch_states)
            vals.append(Upsi.abs())
    x = torch.cat(vals)
    q = torch.quantile(x, x.new_tensor([0.001,0.01,0.1,0.5,0.9,0.99,0.999]))
    tail = {tau: float((x < tau).float().mean().item()) for tau in TAU_LIST}
    print(f"Quantiles [0.1%,1%,10%,50%,90%,99%,99.9%]: {[f'{v:.3e}' for v in q]}")
    print("Tail masses P(|Uψ| < τ):", {f"{tau:g}": f"{p:.3%}" for tau,p in tail.items()})

def inv_eps_sweep(nn, space, phi_true):
    print("\n== TEST B: Inverse floor sweep (impact on phase step sign & magnitude) ==")
    idx = torch.randint(space.shape[0], size=(BATCH,), device=DEVICE)
    batch_states = space[idx]
    basis = random_basis(NV)
    for inv_eps in INV_SWEEP:
        gC = compute_phase_grad(nn, basis, batch_states, mapping='C', inv_eps=inv_eps)
        ddir = directional_derivative(nn, gC, EPS_DIR, phi_true, space)
        _, deltas = line_scan(nn, gC, LINE_STEPS, phi_true, space)
        best = min(deltas)
        print(f"  inv_eps={inv_eps:>8.1e}  dir_deriv={ddir:+.3e}  bestΔ={best:+.3e}")

def grad_variance(nn, space, phi_true, trials=TRIALS, batch=BATCH):
    print("\n== TEST C: Gradient variance & geometry over random batches/bases ==")
    norms = []; cos_oldC = []; improves = 0
    for t in range(trials):
        idx = torch.randint(space.shape[0], size=(batch,), device=DEVICE)
        basis = random_basis(NV)
        batch_states = space[idx]
        g_old = compute_phase_grad(nn, basis, batch_states, mapping='old')
        g_C   = compute_phase_grad(nn, basis, batch_states, mapping='C')
        norms.append(g_C.norm().item())
        cos_oldC.append(cosine(g_old, g_C))
        ddir = directional_derivative(nn, g_C, EPS_DIR, phi_true, space)
        if ddir < 0: improves += 1
    norms = np.array(norms); cos_oldC = np.array(cos_oldC)
    m, s = norms.mean(), norms.std()
    krt = (np.mean((norms - m)**4) / (np.var(norms)**2 + 1e-12))
    print(f"  ||g_C||: mean={m:.3e}  std={s:.3e}  kurtosis≈{krt:.2f}  (heavier tails > 3)")
    print(f"  cos(g_old, g_C): mean={cos_oldC.mean():+.3f}  std={cos_oldC.std():.3f}")
    print(f"  P[dir_deriv<0] for g_C: {improves/trials:.2f}")

def step_sensitivity(nn, space, phi_true):
    print("\n== TEST D: Step-size sensitivity (line scans) ==")
    idx = torch.randint(space.shape[0], size=(BATCH,), device=DEVICE)
    basis = random_basis(NV)
    batch_states = space[idx]
    g_C = compute_phase_grad(nn, basis, batch_states, mapping='C')
    base, deltas = line_scan(nn, g_C, LINE_STEPS, phi_true, space)
    mono = all(deltas[i] <= deltas[i+1] for i in range(len(deltas)-1))  # monotone worsening?
    print(f"  basis={basis} base_err={base:.3e}")
    print("  steps vs Δ(err):", ", ".join([f"{s:g}:{d:+.2e}" for s,d in zip(LINE_STEPS, deltas)]))
    print(f"  monotone? {mono}")

def basis_composition_sweep(nn, space, phi_true, trials_per_k=4):
    print("\n== TEST E: Basis composition (#non-Z axes) vs stability ==")
    for k in K_BASIS:
        signs = []; norms = []
        for _ in range(trials_per_k):
            b = basis_with_k_nonZ(NV, k)
            idx = torch.randint(space.shape[0], size=(BATCH,), device=DEVICE)
            batch_states = space[idx]
            gC = compute_phase_grad(nn, b, batch_states, mapping='C')
            ddir = directional_derivative(nn, gC, EPS_DIR, phi_true, space)
            signs.append(ddir < 0); norms.append(gC.norm().item())
        print(f"  k={k}: P[improve]={np.mean(signs):.2f}  ||g_C|| mean={np.mean(norms):.2e} std={np.std(norms):.2e}")

def finite_difference_check(nn, space, phi_true):
    print("\n== TEST F: Analytic vs finite-diff directional derivative (sanity) ==")
    idx = torch.randint(space.shape[0], size=(BATCH,), device=DEVICE)
    b = random_basis(NV)
    gC = compute_phase_grad(nn, b, space[idx], mapping='C')
    # Compare d/dα err(θ + α d) at α=0 using symmetric difference against ⟨∇,d⟩ proxy
    # Here "proxy" is *computed via the same routine*, so we check internal consistency only.
    ddir_fd = directional_derivative(nn, gC, EPS_DIR, phi_true, space)
    # Projected change using a tiny α on the line-scan first step (diagnostic)
    _, deltas = line_scan(nn, gC, [EPS_DIR], phi_true, space)
    print(f"  basis={b}  dir_deriv_fd={ddir_fd:+.3e}   small-step Δ≈{deltas[0]:+.3e}")

# -----------------------------------
# Helpers for random bases
# -----------------------------------
def random_basis(n):
    letters = np.array(list("XYZ"))
    basis = "".join(np.random.choice(letters, size=n))
    if set(basis) == {"Z"}:
        i = np.random.randint(n); basis = list(basis); basis[i] = np.random.choice(list("XY")); basis = "".join(basis)
    return basis

def basis_with_k_nonZ(n, k):
    assert 0 <= k <= n
    choice = np.array(["Z"]*n)
    if k > 0:
        idx = np.random.choice(np.arange(n), size=k, replace=False)
        choice[idx] = np.random.choice(list("XY"), size=k, replace=True)
    return "".join(choice.tolist())

# -----------------------------------
# Main
# -----------------------------------
if __name__ == "__main__":
    print("===== NQT EXPLORATORY STABILITY TESTS =====")
    print(f"torch: {torch.__version__} | device: {DEVICE} | RTYPE: {RTYPE} | CDTYPE: {CDTYPE}")
    nv = NV; nh = nv
    nn = ComplexWF(nv, nh, device=DEVICE)
    space = generate_hilbert_space(nv, device=DEVICE)

    # target: uniform amplitude + random phases → isolates *phase* learning
    psi_true, phi_true = random_uniform_phase_state(nv, device=DEVICE)

    # 0) quick rotation prob sum sanity with normalized ψ
    with torch.no_grad():
        psi_n = nn.psi_normalized(space)
        sums = {
            "Z"*nv: float((psi_n.abs()**2).sum().item()),
            "X"*nv: float((nn.rotate_psi_inner("X"*nv, space, psi=psi_n).abs()**2).sum().item()),
            "Y"*nv: float((nn.rotate_psi_inner("Y"*nv, space, psi=psi_n).abs()**2).sum().item()),
        }
    print("[BOOT] prob sums (norm ψ):", {k: f"{v:.12f}" for k,v in sums.items()})

    # 1) normalized prob sums via kernel again (redundant, but shows consistency)
    prob_sum_sanity(nn, space)

    # 2) denominator tail survey (|Uψ|)
    denom_tail_survey(nn, space, trials=TRIALS, batch=BATCH)

    # 3) inverse floor sweep: does PHASE-C step actually reduce error as floor varies?
    inv_eps_sweep(nn, space, phi_true)

    # 4) gradient variance and geometry across batches/bases
    grad_variance(nn, space, phi_true, trials=TRIALS, batch=BATCH)

    # 5) step-size sensitivity on a representative batch
    step_sensitivity(nn, space, phi_true)

    # 6) basis composition sweep (#non-Z axes)
    basis_composition_sweep(nn, space, phi_true, trials_per_k=4)

    # 7) finite-diff agreement check (sanity)
    finite_difference_check(nn, space, phi_true)

    print("===== DONE =====")
