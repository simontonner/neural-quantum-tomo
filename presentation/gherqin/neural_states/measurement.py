from typing import Iterable, Optional, List
import torch

from ..utils.linalg import _kron_mult
from ..neural_states.pauli import as_complex_unitary

def rotate_psi(nn_state, basis: Iterable[str], space: torch.Tensor,
               unitaries: Optional[dict] = None, psi: Optional[torch.Tensor] = None):
    """
    Rotate psi into `basis`.
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

    if psi is None:
        x = nn_state.psi_complex(space)
    else:
        if not torch.is_complex(psi):
            raise TypeError("rotate_psi: psi must be complex (cdouble).")
        x = psi.to(device=nn_state.device, dtype=torch.cdouble)

    return _kron_mult(us, x)


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

    Ulist: List[torch.Tensor] = []
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
