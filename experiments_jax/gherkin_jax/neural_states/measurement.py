from typing import Iterable, Optional, Dict, Tuple
import jax
import jax.numpy as jnp

from ..config import DTYPE, CDTYPE
from ..utils.linalg import kron_mult
from .pauli import as_complex_unitary

def rotate_psi(nn_state, basis: Iterable[str], space: jnp.ndarray,
               unitaries: Optional[Dict[str, jnp.ndarray]] = None,
               psi: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Apply ⊗ U_basis to psi over `space` without forming the full Kronecker."""
    n_vis = nn_state.num_visible
    basis = list(basis)
    if len(basis) != n_vis:
        raise ValueError(f"rotate_psi: basis length {len(basis)} != num_visible {n_vis}")
    if unitaries is None:
        us = [nn_state.U[b] for b in basis]
    else:
        Udict = {k: as_complex_unitary(v) for k, v in unitaries.items()}
        us = [Udict[b] for b in basis]
    x = nn_state.psi_complex(space) if psi is None else psi
    if x.dtype != CDTYPE:
        raise TypeError("rotate_psi: psi must be complex128.")
    return kron_mult(us, x)

def _generate_hilbert_space(size: int) -> jnp.ndarray:
    n = 1 << int(size)
    ar = jnp.arange(n, dtype=jnp.int64)
    shifts = jnp.arange(size - 1, -1, -1, dtype=jnp.int64)
    return ((ar[:, None] >> shifts[None, :]) & 1).astype(DTYPE)

def _basis_meta(Udict: Dict[str, jnp.ndarray], basis_tuple: Tuple[str, ...]):
    sites = [i for i, b in enumerate(basis_tuple) if b != "Z"]
    S = len(sites)
    if S == 0:
        Uc_flat = jnp.zeros((0, 4), dtype=CDTYPE)
        combos = _generate_hilbert_space(0)
        return Uc_flat, jnp.asarray([], dtype=jnp.int32), combos
    Ulist = [Udict[basis_tuple[i]].reshape(2, 2) for i in sites]
    Uc = jnp.stack(Ulist, axis=0)
    Uc_flat = Uc.reshape(S, 4)
    combos = _generate_hilbert_space(S)
    return Uc_flat, jnp.asarray(sites, dtype=jnp.int32), combos

@jax.jit
def stable_log_overlap_amp2_with_meta(params_am, params_ph, samples,
                                      Uc_flat, sites, combos):
    """
    Compute log |⟨ outcome | ψ ⟩|^2 per sample, robustly, using basis precomputations.
    """
    C = combos.shape[0]
    v = jnp.tile(samples[None, :, :], (C, 1, 1))
    v = v.at[:, :, sites].set(combos[:, None, :])

    from ..models.rbm import effective_energy
    F_am = effective_energy(params_am, v)
    F_ph = effective_energy(params_ph, v)

    if sites.size == 0:
        Ut = jnp.ones((C, samples.shape[0]), dtype=CDTYPE)
    else:
        inp_sb = samples[:, sites].astype(jnp.int64).T
        outp_csb = v[:, :, sites].astype(jnp.int64).transpose(0, 2, 1)
        inp_csb = jnp.broadcast_to(inp_sb[None, :, :], outp_csb.shape)
        index_scb = (inp_csb * 2 + outp_csb).transpose(1, 0, 2)
        gathered = Uc_flat[jnp.arange(sites.size)[:, None, None], index_scb]  # (S,C,B)
        Ut = jnp.prod(gathered.transpose(1, 2, 0), axis=-1).astype(CDTYPE)    # (C,B)

    eps_u = 1e-300
    eps_s = 1e-12
    logmag_total = (-0.5 * F_am) + jnp.log(jnp.maximum(jnp.abs(Ut).astype(DTYPE), eps_u))
    phase_total  = (-0.5 * F_ph).astype(DTYPE) + jnp.angle(Ut).astype(DTYPE)

    M = jnp.max(logmag_total, axis=0, keepdims=True)
    scaled_mag = jnp.exp((logmag_total - M))
    contrib = scaled_mag.astype(CDTYPE) * jnp.exp(1j * phase_total.astype(CDTYPE))
    S_prime = jnp.sum(contrib, axis=0)
    S_abs2  = jnp.maximum((jnp.conj(S_prime) * S_prime).real.astype(DTYPE), eps_s)
    return (2.0 * M.squeeze(0)).astype(DTYPE) + jnp.log(S_abs2)

# Public accessor so ComplexWaveFunction can cache meta per basis
basis_meta = _basis_meta
