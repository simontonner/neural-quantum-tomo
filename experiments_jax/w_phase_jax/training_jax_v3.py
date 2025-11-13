# =============================================================================
# gherqin_jax/config.py  — Mixed-basis dataloader + padded-rotation JIT (no retracing)
# =============================================================================
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt, prod
from typing import Iterable, List, Tuple, Optional, Dict, Any
from functools import partial
from pathlib import Path
import re
from jax.scipy.special import logsumexp
from jax import tree_util as jtu
import optax

# convenient alias (cross-version safe)
tmap = jtu.tree_map

DTYPE = jnp.float64
CDTYPE = jnp.complex128


# =============================================================================
# gherqin_jax/neural_states/pauli.py
# =============================================================================
def create_dict(**overrides):
    """Return {X,Y,Z} single-qubit unitaries as complex128."""
    inv_sqrt2 = 1.0 / sqrt(2.0)
    X = inv_sqrt2 * jnp.array([[1+0j, 1+0j],[1+0j, -1+0j]], dtype=CDTYPE)
    Y = inv_sqrt2 * jnp.array([[1+0j, 0-1j],[1+0j, 0+1j]], dtype=CDTYPE)
    Z = jnp.array([[1+0j, 0+0j],[0+0j, 1+0j]], dtype=CDTYPE)
    U = {"X": X, "Y": Y, "Z": Z}
    for name, mat in overrides.items():
        U[name] = as_complex_unitary(mat)
    return U

def as_complex_unitary(U, device: Any = None):
    """Ensure a (2,2) complex matrix."""
    U_t = jnp.asarray(U)
    if U_t.ndim != 2 or U_t.shape != (2, 2):
        raise ValueError(f"as_complex_unitary expects (2,2), got {tuple(U_t.shape)}")
    return U_t.astype(CDTYPE)


# =============================================================================
# gherqin_jax/utils/linalg.py
# =============================================================================
def _kron_mult(matrices: List[jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    """Apply (⊗_s U_s)·x without materializing the Kronecker."""
    if x.dtype != CDTYPE:
        raise TypeError("x must be complex128")
    if any(m.dtype != CDTYPE for m in matrices):
        raise TypeError("unitaries must be complex128")
    x_cd = x
    L = x_cd.shape[0]
    batch = int(x_cd.size // L)
    y = x_cd.reshape(L, batch)
    n = [m.shape[-1] for m in matrices]
    if prod(n) != L:
        raise ValueError(f"Incompatible sizes: expected leading dim {prod(n)}, got {L}")
    left = L
    for U in reversed(matrices):
        ns = U.shape[-1]
        left //= ns
        y = y.reshape(left, ns, -1)
        y = jnp.einsum('ij,ljm->lim', U, y).reshape(left * ns, -1)
    return y.reshape(*x_cd.shape)


# =============================================================================
# gherqin_jax/neural_states/measurement.py
# =============================================================================
def rotate_psi(nn_state, basis: Iterable[str], space: jnp.ndarray,
               unitaries: Optional[dict] = None, psi: Optional[jnp.ndarray] = None):
    """Rotate psi into a product basis given as tuple/list of 'X','Y','Z'."""
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
    return _kron_mult(us, x)


# =============================================================================
# gherqin_jax/helpers.py
# =============================================================================
def _generate_hilbert_space(size: int) -> jnp.ndarray:
    """(2^size, size) bit-matrix in {0,1}, MSB-first."""
    n = 1 << int(size)
    ar = jnp.arange(n, dtype=jnp.int64)
    shifts = jnp.arange(size - 1, -1, -1, dtype=jnp.int64)
    return ((ar[:, None] >> shifts[None, :]) & 1).astype(DTYPE)

def _basis_meta(Udict: Dict[str, jnp.ndarray], basis_tuple: Tuple[str, ...]):
    """Precompute tensors for rotated overlap under a fixed basis."""
    sites = [i for i, b in enumerate(basis_tuple) if b != "Z"]
    S = len(sites)
    if S == 0:
        Uc_flat = jnp.zeros((0, 4), dtype=CDTYPE)
        combos = _generate_hilbert_space(0)  # (1,0)
        return Uc_flat, jnp.asarray([], dtype=jnp.int32), combos
    Ulist = [Udict[basis_tuple[i]].reshape(2, 2) for i in sites]
    Uc = jnp.stack(Ulist, axis=0)              # (S,2,2)
    Uc_flat = Uc.reshape(S, 4)                 # (S,4)
    combos = _generate_hilbert_space(S)        # (C,S)
    return Uc_flat, jnp.asarray(sites, dtype=jnp.int32), combos


# =============================================================================
# gherqin_jax/models/rbm.py
# =============================================================================
def _init_rbm_params(num_visible: int, num_hidden: Optional[int] = None,
                     zero_weights: bool = False, key: Optional[jax.Array] = None):
    nv = int(num_visible)
    nh = int(num_hidden) if num_hidden is not None else nv
    scale = 1.0 / np.sqrt(nv)
    if key is None:
        key = jax.random.PRNGKey(0)
    kW, = jax.random.split(key, 1)
    W = jnp.zeros((nh, nv), dtype=DTYPE) if zero_weights else scale * jax.random.normal(kW, (nh, nv), dtype=DTYPE)
    b = jnp.zeros((nv,), dtype=DTYPE)
    c = jnp.zeros((nh,), dtype=DTYPE)
    return {"W": W, "b": b, "c": c}

@jax.jit
def _rbm_effective_energy(params: Dict[str, jnp.ndarray], v: jnp.ndarray) -> jnp.ndarray:
    v = v.astype(DTYPE)
    W, b, c = params["W"], params["b"], params["c"]
    visible_bias_term = jnp.dot(v, b)
    hid_lin = jnp.dot(v, W.T) + c
    hid_term = jnp.sum(jax.nn.softplus(hid_lin), axis=-1)
    return -(visible_bias_term + hid_term)

@partial(jax.jit, static_argnames=("k",))
def _rbm_gibbs_steps(params: Dict[str, jnp.ndarray], k: int, initial_state: jnp.ndarray,
                     key: jax.Array, eps_p: float = 1e-6):
    """k-step block Gibbs from initial_state in {0,1}."""
    W, b, c = params["W"], params["b"], params["c"]
    v0 = initial_state.astype(DTYPE)
    k_curr = key

    def body_fun(_, carry):
        v_curr, k_prev = carry
        k_prev, kh, kv = jax.random.split(k_prev, 3)
        h_prob = jax.nn.sigmoid(jnp.dot(v_curr, W.T) + c)
        h_prob = jnp.nan_to_num(h_prob, nan=0.5, posinf=1.0, neginf=0.0)
        h_prob = jnp.clip(h_prob, eps_p, 1.0 - eps_p)
        h = jax.random.bernoulli(kh, p=h_prob).astype(DTYPE)

        v_prob = jax.nn.sigmoid(jnp.dot(h, W) + b)
        v_prob = jnp.nan_to_num(v_prob, nan=0.5, posinf=1.0, neginf=0.0)
        v_prob = jnp.clip(v_prob, eps_p, 1.0 - eps_p)
        v_next = jax.random.bernoulli(kv, p=v_prob).astype(DTYPE)
        return v_next, k_prev

    vT, k_curr = jax.lax.fori_loop(0, k, body_fun, (v0, k_curr))
    return vT, k_curr


# =============================================================================
# gherqin_jax/neural_states/overlap (Padded, batched, no retrace)
# =============================================================================
def _identity_flat():
    """Flattened 2x2 identity in row-major [1,0,0,1]."""
    return jnp.array([1+0j, 0+0j, 0+0j, 1+0j], dtype=CDTYPE)

def _pad_meta(Udict: Dict[str, jnp.ndarray], basis_tuple: Tuple[str, ...], S_max: int):
    """Return padded (Uc_flat, sites, combos, s_mask, c_mask) for a given basis."""
    # actual non-Z sites
    sites_list = [i for i, b in enumerate(basis_tuple) if b != "Z"]
    S = int(len(sites_list))
    C = 1 << S
    C_max = 1 << int(S_max)

    # Uc_flat_S (S,4)
    if S > 0:
        Ulist = [Udict[basis_tuple[i]].reshape(2, 2) for i in sites_list]
        Uc = jnp.stack(Ulist, axis=0).reshape(S, 4).astype(CDTYPE)
    else:
        Uc = jnp.zeros((0, 4), dtype=CDTYPE)

    # pad to (S_max,4) with identity
    if S < S_max:
        pad = jnp.tile(_identity_flat()[None, :], (S_max - S, 1))
        Uc_flat = jnp.concatenate([Uc, pad], axis=0)
    else:
        Uc_flat = Uc

    # sites (S_max,), pad with zeros (ignored by mask)
    sites = jnp.zeros((S_max,), dtype=jnp.int32)
    if S > 0:
        sites = sites.at[:S].set(jnp.asarray(sites_list, dtype=jnp.int32))

    # combos padded (C_max, S_max)
    combos = jnp.zeros((C_max, S_max), dtype=DTYPE)
    if S > 0:
        combos_S = _generate_hilbert_space(S)  # (C,S)
        combos = combos.at[:C, :S].set(combos_S)

    # masks
    s_mask = jnp.zeros((S_max,), dtype=bool).at[:S].set(True)
    c_mask = jnp.zeros((C_max,), dtype=bool).at[:C].set(True)

    return Uc_flat, sites, combos, s_mask, c_mask

def _stack_meta_batch(meta_cache, bases_batch: List[Tuple[str, ...]]):
    """Stack per-sample meta from cache into (B, ...) arrays."""
    Uc_list, si_list, co_list, sm_list, cm_list = [], [], [], [], []
    for b in bases_batch:
        Uc, si, co, sm, cm = meta_cache[b]
        Uc_list.append(Uc); si_list.append(si); co_list.append(co); sm_list.append(sm); cm_list.append(cm)
    Uc_flat_b = jnp.stack(Uc_list, axis=0)
    sites_b   = jnp.stack(si_list, axis=0)
    combos_b  = jnp.stack(co_list, axis=0)
    s_mask_b  = jnp.stack(sm_list, axis=0)
    c_mask_b  = jnp.stack(cm_list, axis=0)
    return Uc_flat_b, sites_b, combos_b, s_mask_b, c_mask_b

def _log_amp2_one(params_am, params_ph,
                  sample,              # (n,)
                  Uc_flat,             # (S_max, 4) complex
                  sites,               # (S_max,)
                  combos,              # (C_max, S_max)
                  s_mask,              # (S_max,)
                  c_mask):             # (C_max,)
    """Stable log |<basis outcome | psi>|^2 for ONE sample with padded meta."""
    C_max = combos.shape[0]
    S_max = sites.shape[0]

    # Build branches v: (C_max, n) by overwriting columns at active sites
    v = jnp.tile(sample[None, :], (C_max, 1))
    def body(s, vv):
        i = sites[s]
        return jax.lax.cond(
            s_mask[s],
            lambda v_in: v_in.at[:, i].set(combos[:, s]),
            lambda v_in: v_in,
            vv
        )
    v = jax.lax.fori_loop(0, S_max, body, v)

    F_am = _rbm_effective_energy(params_am, v)  # (C_max,)
    F_ph = _rbm_effective_energy(params_ph, v)  # (C_max,)

    # Ut via gather from Uc_flat with indices (inp*2 + out), applying s_mask row-wise
    inp = jnp.where(s_mask, sample[sites].astype(jnp.int64), 0)          # (S_max,)
    out = combos.astype(jnp.int64)                                       # (C_max, S_max)
    idx = inp[None, :] * 2 + out                                         # (C_max, S_max)

    Uc_exp = Uc_flat[:, None, :]                                         # (S_max, 1, 4)
    idx_t  = jnp.transpose(idx, (1, 0))[:, :, None]                      # (S_max, C_max, 1)
    gathered = jnp.take_along_axis(Uc_exp, idx_t, axis=-1).squeeze(-1)   # (S_max, C_max)
    gathered = jnp.where(s_mask[:, None], gathered, jnp.ones_like(gathered))
    Ut = jnp.prod(jnp.transpose(gathered, (1, 0)), axis=-1).astype(CDTYPE)  # (C_max,)

    # Log-sum-exp in magnitude with combo mask
    eps_u = 1e-300
    eps_s = 1e-12
    logmag_total = (-0.5 * F_am).astype(DTYPE) + jnp.log(jnp.maximum(jnp.abs(Ut).astype(DTYPE), eps_u))
    phase_total  = (-0.5 * F_ph).astype(DTYPE) + jnp.angle(Ut).astype(DTYPE)

    masked_logmag = jnp.where(c_mask, logmag_total, -jnp.inf)
    M = jnp.max(masked_logmag, axis=0, keepdims=True)                    # (1,)
    scaled_mag = jnp.where(c_mask, jnp.exp(logmag_total - M), 0.0).astype(DTYPE)
    contrib = scaled_mag.astype(CDTYPE) * jnp.exp(1j * phase_total.astype(CDTYPE))
    S_prime = jnp.sum(contrib, axis=0)
    S_abs2  = jnp.maximum((jnp.conj(S_prime) * S_prime).real.astype(DTYPE), eps_s)
    return (2.0 * M.squeeze(0)).astype(DTYPE) + jnp.log(S_abs2)

@jax.jit
def _stable_log_overlap_amp2_with_meta_padded_batched(params_am, params_ph,
                                                      samples,   # (B,n)
                                                      Uc_flat_b, sites_b, combos_b, s_mask_b, c_mask_b):
    """Vectorized over batch; all padded to static (S_max, C_max)."""
    vmapped = jax.vmap(_log_amp2_one, in_axes=(None, None, 0, 0, 0, 0, 0, 0))
    return vmapped(params_am, params_ph, samples, Uc_flat_b, sites_b, combos_b, s_mask_b, c_mask_b)  # (B,)


# =============================================================================
# gherqin_jax/neural_states/complex_rbm.py
# =============================================================================
class ComplexWaveFunction:
    """psi(sigma) = exp(-F_lambda/2) * exp(-i F_mu/2)."""
    def __init__(self, num_visible, num_hidden=None, unitary_dict=None, rng_seed: int = 0):
        self.rng = jax.random.PRNGKey(rng_seed)
        self.num_visible = int(num_visible)
        self.num_hidden  = int(num_hidden) if num_hidden else self.num_visible
        raw = unitary_dict if unitary_dict is not None else create_dict()
        self.U = {k: as_complex_unitary(v) for k, v in raw.items()}
        self.params = {
            "am": _init_rbm_params(self.num_visible, self.num_hidden, key=jax.random.PRNGKey(rng_seed+1)),
            "ph": _init_rbm_params(self.num_visible, self.num_hidden, key=jax.random.PRNGKey(rng_seed+2)),
        }
        self._basis_cache: Dict[Tuple[str, ...], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = {}
        self._padded_cache: Dict[Tuple[Tuple[str, ...], int], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]] = {}

    def psi_complex(self, v):
        E_lam = _rbm_effective_energy(self.params["am"], v)
        E_mu  = _rbm_effective_energy(self.params["ph"], v)
        amp = jnp.exp(jnp.clip(-0.5 * E_lam, -100.0, 100.0)).astype(DTYPE)
        ph  = jnp.clip(-0.5 * E_mu, -1e6, 1e6).astype(DTYPE)
        return amp.astype(CDTYPE) * jnp.exp(1j * ph.astype(CDTYPE))

    def psi_complex_normalized(self, v):
        E = _rbm_effective_energy(self.params["am"], v)
        ph = (-0.5 * _rbm_effective_energy(self.params["ph"], v)).astype(DTYPE)
        logZ = logsumexp(-E, axis=0)
        return jnp.exp(((-0.5 * E) - 0.5 * logZ).astype(CDTYPE) + 1j * ph.astype(CDTYPE))

    def generate_hilbert_space(self, size=None):
        size = self.num_visible if size is None else int(size)
        return _generate_hilbert_space(size)

    def get_basis_meta(self, basis_tuple: Tuple[str, ...]):
        if basis_tuple not in self._basis_cache:
            self._basis_cache[basis_tuple] = _basis_meta(self.U, basis_tuple)
        return self._basis_cache[basis_tuple]

    def get_basis_meta_padded(self, basis_tuple: Tuple[str, ...], S_max: int):
        key = (basis_tuple, int(S_max))
        if key not in self._padded_cache:
            self._padded_cache[key] = _pad_meta(self.U, basis_tuple, int(S_max))
        return self._padded_cache[key]


# =============================================================================
# gherqin_jax/training/metrics.py
# =============================================================================
def fidelity(nn_state, target, space=None, **kwargs):
    if target.dtype != CDTYPE:
        raise TypeError("fidelity: target must be complex128")
    space = nn_state.generate_hilbert_space() if space is None else space
    psi = nn_state.psi_complex_normalized(space).reshape(-1)
    tgt = target.reshape(-1).astype(CDTYPE)
    npsi = jnp.linalg.norm(psi); nt = jnp.linalg.norm(tgt)
    if (npsi == 0) or (nt == 0): return 0.0
    psi_n = psi / npsi; tgt_n = tgt / nt
    inner = jnp.vdot(tgt_n, psi_n)
    return float(jnp.abs(inner) ** 2)

def KL(nn_state, target, space=None, bases=None, **kwargs):
    if bases is None: raise ValueError("KL needs bases")
    if target.dtype != CDTYPE: raise TypeError("KL: target must be complex128")
    space = nn_state.generate_hilbert_space() if space is None else space
    tgt = target.reshape(-1).astype(CDTYPE)
    nt = jnp.linalg.norm(tgt)
    if nt == 0: return 0.0
    tgt_norm = tgt / nt
    psi_norm_cd = nn_state.psi_complex_normalized(space).reshape(-1)
    KL_val = 0.0; eps = 1e-12
    for basis in bases:
        tgt_psi_r = rotate_psi(nn_state, basis, space, psi=tgt_norm)
        psi_r     = rotate_psi(nn_state, basis, space, psi=psi_norm_cd)
        nn_probs_r  = (jnp.abs(psi_r).astype(DTYPE)) ** 2
        tgt_probs_r = (jnp.abs(tgt_psi_r).astype(DTYPE)) ** 2
        p_sum = jnp.maximum(jnp.sum(tgt_probs_r), eps)
        q_sum = jnp.maximum(jnp.sum(nn_probs_r), eps)
        p = jnp.maximum(tgt_probs_r / p_sum, eps)
        q = jnp.maximum(nn_probs_r / q_sum, eps)
        KL_val += float(jnp.sum(p * (jnp.log(p) - jnp.log(q))))
    return KL_val / len(bases)


# =============================================================================
# gherqin_jax/data/tomography.py  (Per-basis directory + MIXED loader)
# =============================================================================
class PerBasisTomographyDataset:
    """
    Load from:
      measurements/
        w_phase_state.txt                    -> target psi: two columns (Re, Im)
        w_phase_<BASIS>_<shots>.txt          -> outcomes encoded per line by case
    Encoding:
      line "ZzZX" => [0,1,0,0] for basis "ZZZX"
    Exposes:
      - train_samples : (N, n) float64 (jnp)
      - train_bases   : list[tuple[str,...]] length N (Python)
      - target_state  : (2^n,) complex128 (jnp)
      - _unique_bases : list[tuple[str,...]]
      - _z_indices    : jnp[int]
    """
    def __init__(self,
                 directory: str = "measurements",
                 state_filename: str = "w_phase_state.txt",
                 file_prefix: str = "w_phase_"):
        d = Path(directory)
        if not d.is_dir():
            raise FileNotFoundError(f"Directory not found: {d}")

        # target psi
        psi_path = d / state_filename
        if not psi_path.exists():
            raise FileNotFoundError(f"Missing state file: {psi_path}")
        psi_np = np.loadtxt(str(psi_path), dtype="float64")
        if psi_np.ndim != 2 or psi_np.shape[1] != 2:
            raise ValueError("State file must have two columns: Re Im.")
        self.target_state = jnp.asarray(psi_np[:, 0] + 1j * psi_np[:, 1], dtype=CDTYPE)

        # per-basis files
        per_basis_files = [p for p in d.glob("*.txt") if p.name != state_filename and p.name.startswith(file_prefix)]
        if not per_basis_files:
            raise FileNotFoundError(f"No per-basis files matching '{file_prefix}*' in {d}")

        samples: List[np.ndarray] = []
        bases_rows: List[Tuple[str, ...]] = []
        unique_bases: Dict[Tuple[str, ...], None] = {}

        for p in sorted(per_basis_files):
            bcode = self._parse_basis_code_from_filename(p.name, prefix=file_prefix)
            basis = tuple(list(bcode))  # uppercase letters 'X','Y','Z' (optionally 'I')

            with open(p, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]

            if not lines:
                continue
            n = len(basis)
            for ln in lines:
                if len(ln) != n:
                    raise ValueError(f"Line length != basis width in {p}: '{ln}'")
                bits = []
                for ch in ln:
                    if ch.isalpha():
                        bits.append(0 if ch.isupper() else 1)
                    else:
                        raise ValueError(f"Illegal char '{ch}' in {p}")
                samples.append(np.asarray(bits, dtype=np.float64))
                bases_rows.append(basis)

            unique_bases[basis] = None

        if not samples:
            raise ValueError("No samples read from per-basis files.")

        self.train_samples = jnp.asarray(np.stack(samples, axis=0), dtype=DTYPE)
        self.train_bases: List[Tuple[str, ...]] = bases_rows
        self._unique_bases = list(unique_bases.keys())

        # guardrails
        widths = {len(row) for row in self.train_bases}
        if len(widths) != 1:
            raise ValueError("Inconsistent basis widths across files.")
        n = next(iter(widths))
        if self.train_samples.shape[1] != n:
            raise ValueError("Sample width != basis width.")
        expected_dim = 1 << n
        if int(self.target_state.size) != expected_dim:
            raise ValueError(f"State length {int(self.target_state.size)} != 2^{n} ({expected_dim}).")

        # z-only indices
        z_mask = np.array([all(ch == "Z" for ch in row) for row in self.train_bases], dtype=bool)
        self._z_indices = jnp.asarray(np.nonzero(z_mask)[0], dtype=jnp.int32)
        if int(self._z_indices.size) == 0:
            raise ValueError("No Z-only rows found; needed for negative sampling.")

        # max non-Z across all rows (for padding)
        self._S_max = max(sum(1 for ch in row if ch != "Z") for row in self.train_bases)

    @staticmethod
    def _parse_basis_code_from_filename(name: str, prefix: str) -> str:
        """
        Accept 'w_phase_XXZZ_5000.txt' -> 'XXZZ'.
        """
        stem = Path(name).stem  # e.g. 'w_phase_XXZZ_5000'
        if not stem.startswith(prefix):
            raise ValueError(f"Bad file name (prefix): {name}")
        tail = stem[len(prefix):]  # 'XXZZ_5000'
        if "_" not in tail:
            raise ValueError(f"Bad file name (missing shots part): {name}")
        code = tail.rsplit("_", 1)[0]
        if not re.fullmatch(r"[XYZI]+", code):
            raise ValueError(f"Basis code must be uppercase letters [XYZI]+, got '{code}' from {name}")
        return code

    # API
    def __len__(self): return int(self.train_samples.shape[0])
    def num_visible(self) -> int: return int(self.train_samples.shape[1])
    def z_indices(self) -> jnp.ndarray: return self._z_indices.copy()
    def eval_bases(self) -> List[Tuple[str, ...]]: return list(self._unique_bases)
    def target(self) -> jnp.ndarray: return self.target_state
    def S_max(self) -> int: return int(self._S_max)
    def train_bases_as_tuples(self) -> List[Tuple[str, ...]]: return list(self.train_bases)


class RBMTomographyLoaderMixed:
    """Yield (pos_batch, neg_batch, bases_batch) minibatches for training (mixed bases)."""
    def __init__(self, dataset: PerBasisTomographyDataset, pos_batch_size: int = 100, neg_batch_size: Optional[int] = None,
                 seed: Optional[int] = None):
        self.ds = dataset
        self.pos_bs = int(pos_batch_size)
        self.neg_bs = int(neg_batch_size or pos_batch_size)
        self._key = jax.random.PRNGKey(0 if seed is None else int(seed))

        n = self.ds.num_visible()
        bases_seq = self.ds.train_bases_as_tuples()
        if any(len(row) != n for row in bases_seq):
            raise ValueError("RBMTomographyLoaderMixed: inconsistent basis widths in dataset")
        if self.ds.z_indices().size == 0:
            raise ValueError("RBMTomographyLoaderMixed: Z-only pool is empty (need negatives)")

    def __len__(self):
        return ceil(len(self.ds) / self.pos_bs)

    def S_max(self) -> int:
        return self.ds.S_max()

    def iter_epoch(self):
        """Iterate once over the dataset, yielding mixed-basis batches."""
        N = len(self.ds)
        n_batches = ceil(N / self.pos_bs)

        self._key, kperm, kneg = jax.random.split(self._key, 3)
        perm = np.asarray(jax.random.permutation(kperm, jnp.arange(N)).tolist(), dtype=int)

        samples = self.ds.train_samples[perm].astype(DTYPE)
        bases_list = self.ds.train_bases_as_tuples()
        bases_perm = [bases_list[i] for i in perm.tolist()]

        z_pool = self.ds.z_indices()
        pool_len = int(z_pool.size)
        neg_choices = jax.random.randint(kneg, shape=(n_batches * self.neg_bs,), minval=0, maxval=pool_len)
        neg_rows = z_pool[neg_choices]
        neg_samples_all = self.ds.train_samples[neg_rows].astype(DTYPE)

        for b in range(n_batches):
            start = b * self.pos_bs
            end = min(start + self.pos_bs, N)
            pos_batch = samples[start:end]
            bases_batch = bases_perm[start:end]

            nb_start = b * self.neg_bs
            nb_end = nb_start + self.neg_bs
            neg_batch = neg_samples_all[nb_start:nb_end]

            yield pos_batch, neg_batch, bases_batch


# =============================================================================
# gherqin_jax/training/losses_and_grads (mixed-basis, padded)
# =============================================================================
@partial(jax.jit, static_argnames=("k",))
def _grads_only_mixed(params, pos_batch, neg_batch, rng,
                      Uc_flat_b, sites_b, combos_b, s_mask_b, c_mask_b,
                      k: int):
    """Return (loss, grads, rng) for a mixed-basis batch with padded rotation meta."""
    vk, rng = _rbm_gibbs_steps(params["am"], k, neg_batch, rng)
    B_pos = pos_batch.shape[0]
    B_neg = neg_batch.shape[0]

    def loss_fn(p):
        # positive phase: Z rows via energy, rotated rows via padded overlap
        Epos = _rbm_effective_energy(p["am"], pos_batch)                  # (B,)
        log_amp2 = _stable_log_overlap_amp2_with_meta_padded_batched(     # (B,)
            p["am"], p["ph"], pos_batch, Uc_flat_b, sites_b, combos_b, s_mask_b, c_mask_b
        )
        # detect Z rows: no active rotations
        is_z = jnp.logical_not(jnp.any(s_mask_b, axis=1))                 # (B,)
        L_i = jnp.where(is_z, Epos, -log_amp2).astype(DTYPE)
        L_pos = jnp.sum(L_i)

        # negative phase (CD-k) for amplitude RBM
        L_neg = jnp.sum(_rbm_effective_energy(p["am"], vk).astype(DTYPE))
        return (L_pos / B_pos) - (L_neg / B_neg)

    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    return loss_val, grads, rng


# =============================================================================
# gherqin_jax/training/trainer.py
# =============================================================================
class Trainer:
    def __init__(self, nn_state: ComplexWaveFunction, base_lr=1e-1,
                 phase_lr_scale=0.25, momentum=0.9, accum_steps=8):
        self.nn = nn_state
        # label tree for multi_transform
        labels = {
            "am": tmap(lambda _: "am", self.nn.params["am"]),
            "ph": tmap(lambda _: "ph", self.nn.params["ph"]),
        }
        transforms = {
            "am": optax.sgd(learning_rate=base_lr, momentum=momentum, nesterov=False),
            "ph": optax.sgd(learning_rate=base_lr * phase_lr_scale, momentum=momentum, nesterov=False),
        }
        self.opt = optax.chain(optax.clip_by_global_norm(10.0), optax.multi_transform(transforms, labels))
        self.opt_state = self.opt.init(self.nn.params)
        self.accum_steps = int(accum_steps)

    def _zero_like_params(self):
        return tmap(jnp.zeros_like, self.nn.params)

    def fit(self, loader: RBMTomographyLoaderMixed, epochs=70, k=10, log_every=5,
            target=None, bases=None, space=None,
            print_metrics=True,
            metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}"):
        history: Dict[str, list] = {"epoch": [], "Fidelity": [], "KL": []}

        # Precompute padded meta cache for *all* unique bases
        S_max = loader.S_max()
        meta_cache = {b: self.nn.get_basis_meta_padded(b, S_max) for b in loader.ds.eval_bases()}

        for ep in range(1, epochs + 1):
            grads_accum = self._zero_like_params()
            micro = 0

            for pos_batch, neg_batch, bases_batch in loader.iter_epoch():
                # Stack per-sample padded meta from cache
                Uc_flat_b, sites_b, combos_b, s_mask_b, c_mask_b = _stack_meta_batch(meta_cache, bases_batch)

                # Mixed-basis JIT step
                loss_val, grads, self.nn.rng = _grads_only_mixed(
                    self.nn.params, pos_batch, neg_batch, self.nn.rng,
                    Uc_flat_b, sites_b, combos_b, s_mask_b, c_mask_b, k=k
                )

                grads_accum = tmap(lambda a, b: a + b, grads_accum, grads)
                micro += 1

                if micro == self.accum_steps:
                    grads_avg = tmap(lambda g: g / self.accum_steps, grads_accum)
                    updates, self.opt_state = self.opt.update(grads_avg, self.opt_state, self.nn.params)
                    self.nn.params = optax.apply_updates(self.nn.params, updates)
                    grads_accum = self._zero_like_params()
                    micro = 0

            # flush remainder
            if micro > 0:
                grads_avg = tmap(lambda g: g / micro, grads_accum)
                updates, self.opt_state = self.opt.update(grads_avg, self.opt_state, self.nn.params)
                self.nn.params = optax.apply_updates(self.nn.params, updates)

            # sync point (force compute)
            jnp.sum(self.nn.params["am"]["W"]).block_until_ready()

            if (target is not None) and (bases is not None) and (ep % log_every == 0):
                fid_val = fidelity(self.nn, target, space=space, bases=bases)
                kl_val  = KL(self.nn, target, space=space, bases=bases)
                history["epoch"].append(ep)
                history["Fidelity"].append(fid_val)
                history["KL"].append(kl_val)
                if print_metrics:
                    print(metric_fmt.format(ep=ep, fid=fid_val, kl=kl_val))
        return history


# =============================================================================
# __main__ — training script (mixed-basis like Torch)
# =============================================================================
if __name__ == "__main__":
    # Paths and data (per-basis directory)
    measurements_dir = "measurements"
    state_filename   = "w_phase_state.txt"

    base_seed = 1234
    np.random.seed(base_seed)

    data = PerBasisTomographyDataset(directory=measurements_dir,
                                     state_filename=state_filename,
                                     file_prefix="w_phase_")
    U = create_dict()

    nv = int(data.num_visible()); nh = nv
    nn_state = ComplexWaveFunction(num_visible=nv, num_hidden=nh, unitary_dict=U, rng_seed=base_seed)

    # config — tuned to mirror Torch defaults
    epochs = 70
    pbs = 100
    nbs = 100
    base_lr = 1e-1
    phase_lr_scale = 0.25
    momentum = 0.9
    k_cd = 10
    log_every = 5
    accum_steps = 8

    loader = RBMTomographyLoaderMixed(data, pos_batch_size=pbs, neg_batch_size=nbs, seed=base_seed)
    space = nn_state.generate_hilbert_space()

    print("===== Run configuration =====")
    print(f"num_visible={nv} | num_hidden={nh} | pos_bs={pbs} | neg_bs={nbs} | k={k_cd}")
    print(f"lr_am={base_lr} | lr_ph={base_lr*phase_lr_scale} | momentum={momentum} | accum_steps={accum_steps}")
    print(f"samples={len(data)} | batches/epoch={len(loader)} | epochs={epochs} | log_every={log_every}")
    print(f"S_max={data.S_max()} | C_max={1<<data.S_max()}")
    print("================================\n")

    trainer = Trainer(nn_state, base_lr=base_lr, phase_lr_scale=phase_lr_scale, momentum=momentum, accum_steps=accum_steps)
    history = trainer.fit(loader,
                          epochs=epochs,
                          k=k_cd,
                          log_every=log_every,
                          target=data.target(),
                          bases=data.eval_bases(),
                          space=space,
                          print_metrics=True,
                          metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}")

    # ---------- Phase comparison ----------
    space = nn_state.generate_hilbert_space()
    psi_m = nn_state.psi_complex_normalized(space).reshape(-1)
    psi_t = data.target().reshape(-1).astype(CDTYPE)
    nm = jnp.linalg.norm(psi_m); nt = jnp.linalg.norm(psi_t)
    if float(nm) > 0.0: psi_m = psi_m / nm
    if float(nt) > 0.0: psi_t = psi_t / nt
    ip = jnp.vdot(psi_t, psi_m)
    if float(jnp.abs(ip)) > 1e-12:
        theta = jnp.angle(ip)
    else:
        j = int(jnp.argmax(jnp.abs(psi_t)))
        theta = jnp.angle(psi_m[j]) - jnp.angle(psi_t[j])
    psi_m_al = psi_m * jnp.exp(-1j * theta)

    phi_t = np.array(jnp.angle(psi_t))
    phi_m = np.array(jnp.angle(psi_m_al))
    dphi = np.remainder((phi_m - phi_t) + np.pi, 2.0 * np.pi) - np.pi
    probs = np.array(jnp.abs(psi_t) ** 2)
    order = np.argsort(-probs)
    cum = np.cumsum(probs[order])
    k_sel = int(min(np.searchsorted(cum, 0.99) + 1, 512, len(order)))
    sel = order[:k_sel]

    fig_p, axp = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    axp.plot(range(k_sel), phi_t[sel], marker=".", linestyle="", label="target phase")
    axp.plot(range(k_sel), phi_m[sel], marker="x", linestyle="", label="model phase (aligned)")
    axp.set_xlabel("basis states (sorted by target mass)")
    axp.set_ylabel("phase [rad]")
    axp.set_title("Phase comparison – top 99% mass")
    axp.grid(True, alpha=0.3); axp.legend(); fig_p.tight_layout()

    fig_e, axe = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    axe.plot(range(k_sel), dphi[sel], marker=".", linestyle="", label="Δphase (wrapped)")
    axe.axhline(0.0, linewidth=1.0)
    axe.set_xlabel("basis states (sorted by target mass)")
    axe.set_ylabel("Δphase [rad] in [-π, π]")
    axe.set_title("Phase error (global phase aligned)")
    axe.grid(True, alpha=0.3); axe.legend(); fig_e.tight_layout()

    # ---------- Metrics plot ----------
    ep_hist = history.get("epoch", [])
    if ep_hist and ("Fidelity" in history) and ("KL" in history):
        fig, ax1 = plt.subplots(figsize=(6.0, 4.0), dpi=140)
        ax2 = ax1.twinx()
        ax1.plot(ep_hist, history["Fidelity"], marker="o", label="Fidelity")
        ax2.plot(ep_hist, history["KL"], marker="s", linestyle="--", label="KL")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel(r"$|\langle \psi_t \mid \psi \rangle|^2$")
        ax2.set_ylabel(r"KL$(p\,\|\,q)$")
        ax1.set_title("RBM Tomography – training metrics (JAX CD-k, momentum+accum; mixed bases)")
        ax1.grid(True, alpha=0.3)
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="best")
        fig.tight_layout()
    else:
        print("No metrics to plot (did you pass target & bases and set log_every?).")

    plt.show()
