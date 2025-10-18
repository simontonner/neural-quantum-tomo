# rbm_rotkernel_debug.py
# Focus: detect and localize mismatches between two rotate-basis kernels.
# Drops downstream KL/grad noise and the earlier "Y-phase" unitary dispute
# (you've already fixed that). Still includes a minimal unitary sanity gate
# so you don't chase kernel bugs when the unitary set regresses.
#
# Usage (from project folder with your data files):
#   python rbm_rotkernel_debug.py \
#       --train w_state_meas.txt \
#       --train-bases w_state_basis.txt \
#       --psi w_state_aug.txt \
#       --bases w_state_bases.txt \
#       --max-bases 32

from math import sqrt, prod
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------------------------
# Device & dtype
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double  # real params in float64

# -------------------------------
# IO: Tomography data
# -------------------------------
class TomographyData:
    def __init__(self, train_path, psi_path, train_bases_path, bases_path, device=DEVICE):
        self.device = device
        self.train_samples = torch.tensor(np.loadtxt(train_path, dtype="float32"), dtype=DTYPE, device=device)
        psi_np = np.loadtxt(psi_path, dtype="float64")  # [Re, Im]
        self.target_state = torch.tensor(psi_np[:, 0] + 1j * psi_np[:, 1], dtype=torch.cdouble, device=device)
        self.train_bases = np.loadtxt(train_bases_path, dtype=str)
        self.bases = np.loadtxt(bases_path, dtype=str, ndmin=1)

# -------------------------------
# Unitaries: legacy 2-row -> complex (canonical gate sanity optional)
# -------------------------------
def create_dict_2row_legacy():
    inv_sqrt2 = 1.0 / sqrt(2.0)
    return {
        "X": torch.tensor(
            [[[1.0, 1.0], [1.0, -1.0]], [[0.0, 0.0], [0.0, 0.0]]],
            dtype=DTYPE, device=DEVICE,
        ) * inv_sqrt2,
        # Legacy-exact Y -> [[1, -i],[1, i]] / sqrt(2)
        "Y": torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0]], [[0.0, -1.0], [0.0, 1.0]]],
            dtype=DTYPE, device=DEVICE,
        ) * inv_sqrt2,
        "Z": torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
            dtype=DTYPE, device=DEVICE,
        ),
    }

def legacy_2row_to_complex(unitary_2row_dict, device=DEVICE):
    U = {}
    for k, v in unitary_2row_dict.items():
        U[k] = (v[0].to(dtype=torch.cdouble, device=device) +
                1j * v[1].to(dtype=torch.cdouble, device=device)).contiguous()
    return U

def check_unitary_gate_set(Udict, name="U"):
    """Lightweight sanity only (unitarity & det mag)."""
    I = torch.eye(2, dtype=torch.cdouble, device=DEVICE)
    print(f"\n[CHECK] {name} unitarity/det:")
    for g in ["X", "Y", "Z"]:
        U = Udict[g]
        err = torch.linalg.norm(U.conj().T @ U - I).item()
        det_mag = abs(torch.det(U).abs().item() - 1.0)
        print(f"  {g}: ||U†U - I||_F = {err:.3e}   | |det|-1 | = {det_mag:.3e}")

def report_matrix_diff(A, B, name, atol=1e-12, top=4):
    diff = (A - B).abs()
    maxd = float(diff.max().item())
    nz = (diff > atol).nonzero(as_tuple=False)
    print(f"[DIFF] {name}: max|Δ| = {maxd:.3e} (atol={atol})  nonzero_count={nz.shape[0]}")
    for i in range(min(top, nz.shape[0])):
        idx = tuple(int(x) for x in nz[i])
        print(f"   idx {idx}: A={A[idx].item()}  B={B[idx].item()}  Δ={(A[idx]-B[idx]).item()}")

# -------------------------------
# Hilbert utilities
# -------------------------------
def generate_hilbert_space(num_visible, device=DEVICE):
    n = 1 << num_visible
    ar = torch.arange(n, device=device, dtype=torch.long)
    shifts = torch.arange(num_visible - 1, -1, -1, device=device, dtype=torch.long)
    return ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)

# -------------------------------
# Minimal RBM + psi (only to build ψ for rotate_psi)
# -------------------------------
class BinaryRBM(nn.Module):
    def __init__(self, num_visible, num_hidden=None, device: torch.device = DEVICE):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden) if num_hidden else self.num_visible
        self.num_pars = (self.num_visible * self.num_hidden) + self.num_visible + self.num_hidden
        self.device = device
        self.weights = nn.Parameter(torch.randn(self.num_hidden, self.num_visible, device=device, dtype=DTYPE) / np.sqrt(self.num_visible), requires_grad=False)
        self.visible_bias = nn.Parameter(torch.zeros(self.num_visible, device=device, dtype=DTYPE), requires_grad=False)
        self.hidden_bias = nn.Parameter(torch.zeros(self.num_hidden, device=device, dtype=DTYPE), requires_grad=False)

    def effective_energy(self, v):
        unsq = False
        if v.dim() < 2: v = v.unsqueeze(0); unsq = True
        v = v.to(self.weights)
        vb = torch.matmul(v, self.visible_bias)
        hb = F.softplus(F.linear(v, self.weights, self.hidden_bias)).sum(-1)
        out = -(vb + hb)
        return out.squeeze(0) if unsq else out

class ComplexWaveFunction(nn.Module):
    def __init__(self, num_visible, num_hidden=None, U=None, device: torch.device = DEVICE):
        super().__init__()
        self.device = device
        self.rbm_am = BinaryRBM(num_visible, num_hidden, device=device)
        self.rbm_ph = BinaryRBM(num_visible, num_hidden, device=device)
        self.num_visible = self.rbm_am.num_visible
        self.U = {k: v.to(device=self.device, dtype=torch.cdouble).contiguous() for k, v in U.items()}

    def psi_cd(self, v):
        v = v.to(self.device, dtype=DTYPE)
        amp = (-self.rbm_am.effective_energy(v)).exp().sqrt()
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        return amp.to(torch.cdouble) * torch.exp(1j * ph.to(torch.cdouble))

# -------------------------------
# Old vs New rotate-basis kernels
# -------------------------------
def _rotate_basis_state_gather(Udict, device, num_visible, basis, states):
    # "Old" gather-based kernel
    basis_seq = list(basis)
    if len(basis_seq) != num_visible: raise ValueError("basis length mismatch")
    if states.shape[-1] != num_visible: raise ValueError("states width mismatch")
    sites = [i for i, b in enumerate(basis_seq) if b != "Z"]
    if len(sites) == 0:
        v = states.unsqueeze(0)
        Ut = torch.ones(v.shape[:-1], dtype=torch.cdouble, device=device)
        return Ut, v
    Uc = torch.stack([Udict[basis_seq[i]].to(device=device, dtype=torch.cdouble) for i in sites], dim=0)
    S = len(sites); B = states.shape[0]; C = 2 ** S
    combos = generate_hilbert_space(S, device=device)
    v = states.unsqueeze(0).repeat(C, 1, 1)
    v[:, :, sites] = combos.unsqueeze(1)
    v = v.contiguous()
    inp  = states[:, sites].round().long().T
    outp = v[:, :, sites].round().long().permute(0, 2, 1)
    Uio_exp = Uc.unsqueeze(0).unsqueeze(2).expand(C, S, B, 2, 2)
    inp_idx = inp.unsqueeze(0).expand(C, S, B).unsqueeze(-1).unsqueeze(-1)
    sel_in  = torch.gather(Uio_exp, dim=3, index=inp_idx.expand(C, S, B, 1, 2))
    out_idx = outp.unsqueeze(-1).unsqueeze(-1)
    sel_out = torch.gather(sel_in, dim=4, index=out_idx)
    Ut = sel_out.squeeze(-1).squeeze(-1).permute(0, 2, 1).prod(dim=-1)
    return Ut.to(torch.cdouble), v

def _rotate_basis_state_index(Udict, device, num_visible, basis, states):
    # "New" index-based kernel
    basis_seq = list(basis)
    if len(basis_seq) != num_visible: raise ValueError("basis length mismatch")
    if states.shape[-1] != num_visible: raise ValueError("states width mismatch")
    sites = [i for i, b in enumerate(basis_seq) if b != "Z"]
    if len(sites) == 0:
        v = states.unsqueeze(0)
        Ut = torch.ones(v.shape[:-1], dtype=torch.cdouble, device=device)
        return Ut, v
    Uc = torch.stack([Udict[basis_seq[i]].to(device=device, dtype=torch.cdouble) for i in sites], dim=0)  # (S,2,2)
    S = len(sites); B = states.shape[0]; C = 2 ** S
    combos = generate_hilbert_space(S, device=device)
    v = states.unsqueeze(0).repeat(C, 1, 1)
    v[:, :, sites] = combos.unsqueeze(1)
    v = v.contiguous()
    inp_sb   = states[:, sites].round().long().T               # (S,B)
    outp_csb = v[:, :, sites].round().long().permute(0, 2, 1)  # (C,S,B)
    inp_csb  = inp_sb.unsqueeze(0).expand(C, -1, -1)           # (C,S,B)
    s_idx = torch.arange(S, device=device).view(1, S, 1).expand(C, S, B)  # (C,S,B)
    sel = Uc[s_idx, outp_csb, inp_csb]  # (C,S,B)
    Ut = sel.prod(dim=1)  # (C,B)
    return Ut.to(torch.cdouble), v

# -------------------------------
# Diff helpers
# -------------------------------
def sample_bases(bases_np, max_bases=None, seed=1234):
    rng = np.random.default_rng(seed)
    if max_bases is None or max_bases >= len(bases_np):
        return [tuple(row) for row in bases_np]
    idxs = rng.choice(len(bases_np), size=max_bases, replace=False)
    return [tuple(bases_np[i]) for i in idxs]

def report_scalar(name, val):
    print(f"{name}: {val:.3e}")

# -------------------------------
# Main
# -------------------------------
def main(args):
    torch.set_default_dtype(DTYPE)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {DEVICE}  DTYPE: {DTYPE}")
    data = TomographyData(args.train, args.psi, args.train_bases, args.bases, device=DEVICE)
    nv = int(data.train_samples.shape[1])

    # Use ONE unitary set for both kernels to isolate the kernel bug.
    U_compat = legacy_2row_to_complex(create_dict_2row_legacy(), device=DEVICE)
    check_unitary_gate_set(U_compat, "U_compat (legacy-converted)")

    # Optional regression guard: if you pass an alternate U and want to ensure equality, enable this.
    if args.compare_u is not None:
        # Simple canonical set for comparison (should equal U_compat if your fix is in place)
        inv_sqrt2 = 1.0 / sqrt(2.0)
        U_canonical = {
            "X": inv_sqrt2 * torch.tensor([[1.0+0j, 1.0+0j],[1.0+0j,-1.0+0j]], dtype=torch.cdouble, device=DEVICE),
            "Y": inv_sqrt2 * torch.tensor([[1.0+0j, 0.0-1j],[1.0+0j, 0.0+1j]], dtype=torch.cdouble, device=DEVICE),
            "Z": torch.tensor([[1.0+0j, 0.0+0j],[0.0+0j, 1.0+0j]], dtype=torch.cdouble, device=DEVICE),
        }
        print("\n=== Optional unitary equality gate (should be all zeros if your Y fix is kept) ===")
        for g in ["X","Y","Z"]:
            report_matrix_diff(U_compat[g], U_canonical[g], f"U[{g}]_compat_vs_canonical")

    # Build ψ once (identical for both kernels)
    model = ComplexWaveFunction(nv, nv, U=U_compat, device=DEVICE).to(DEVICE)
    space = generate_hilbert_space(nv, device=DEVICE)
    psi_space = model.psi_cd(space)  # not used in the kernel diff itself, but handy if you want to extend

    # === Core test: kernel A/B comparison on Ut and v ===
    print("\n=== Rotation kernel A/B test (same U, different kernels) ===")
    bases_list = sample_bases(data.bases, max_bases=args.max_bases, seed=args.seed)
    worst = (0.0, None, 0.0, 0.0)  # (score, basis, max|ΔUt|, max|Δv|)
    for b in bases_list:
        UtA, vA = _rotate_basis_state_gather(model.U, DEVICE, nv, b, space)
        UtB, vB = _rotate_basis_state_index(model.U, DEVICE, nv, b, space)
        dU = (UtA - UtB).abs().max().item()
        dv = (vA - vB).abs().max().item()
        score = dU + dv
        if score > worst[0]:
            worst = (score, b, dU, dv)

    score, worst_basis, dU, dv = worst
    report_scalar("Max |ΔUt|", dU)
    report_scalar("Max |Δv|", dv)
    print(f"Worst basis: {worst_basis}")

    if score <= args.atol:
        print("\n==> Kernels MATCH within tolerance. If you still see training drift, look elsewhere (dtype/casts, RNG, batching).")
        return

    # === Forensic dump for the worst basis ===
    print("\n=== Forensic dump (worst basis) ===")
    UtA, vA = _rotate_basis_state_gather(model.U, DEVICE, nv, worst_basis, space)
    UtB, vB = _rotate_basis_state_index(model.U, DEVICE, nv, worst_basis, space)

    # Show first few branch amplitudes (Ut) and any discrete v mismatches
    C = UtA.size(0)
    B = UtA.size(1)
    print(f"Branches C={C}, batch B={B}")
    # pick a stable column (first) for readability
    col = 0
    top = min(8, C)
    for c in range(top):
        a = UtA[c, col].item()
        b = UtB[c, col].item()
        print(f"  branch {c:02d}: Ut_old={a}  Ut_new={b}  Δ={(a-b)}")

    # Check if enumerated states disagree (should be 0/1 and equal)
    dv_mask = (vA != vB).any(dim=-1).any(dim=-1)
    if dv_mask.any():
        idxs = torch.nonzero(dv_mask, as_tuple=False).flatten().tolist()
        print("\nState enumeration mismatch at branches:", idxs[:8], ("..." if len(idxs) > 8 else ""))
        # show one offending row
        c0 = idxs[0]
        print(f"v_old[{c0}, 0, :]={vA[c0,0,:].tolist()}")
        print(f"v_new[{c0}, 0, :]={vB[c0,0,:].tolist()}")
    else:
        print("\nState enumeration (v) agrees across kernels.")

    print("\n=== Summary ===")
    print("- These diffs isolate the rotate-basis kernel bug. Fix whichever kernel yields the nonzero ΔUt/Δv above.")
    print("- Your unitary set is held constant (legacy-converted complex). If ΔUt=Δv=0, the kernel refactor is fine.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rotation-kernel mismatch debug: old(gather) vs new(index) on same U.")
    ap.add_argument("--train", default="w_state_meas.txt")
    ap.add_argument("--train-bases", dest="train_bases", default="w_state_basis.txt")
    ap.add_argument("--psi", default="w_state_aug.txt")
    ap.add_argument("--bases", default="w_state_bases.txt")
    ap.add_argument("--max-bases", type=int, default=16, help="number of bases to sample")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--compare-u", dest="compare_u", default=None,
                    help="Set any non-empty value to run a quick U_compat vs canonical sanity diff.")
    args = ap.parse_args()
    main(args)