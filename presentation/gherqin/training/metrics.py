import torch
from ..config import DTYPE
from ..neural_states.measurement import rotate_psi

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
