# =============================================================================
# JAX RBM Tomography — Torch-faithful, single-compile across mixed rotations
# =============================================================================
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt, prod
from typing import Iterable, List, Tuple, Optional, Dict, Any
from pathlib import Path
import re
from jax.scipy.special import logsumexp
from jax import tree_util as jtu
import optax
from functools import partial

# ---------- dtypes / helpers ----------
tmap = jtu.tree_map
DTYPE = jnp.float64
CDTYPE = jnp.complex128

# ---------- unitaries ----------
def create_dict(**overrides):
    inv_sqrt2 = 1.0 / sqrt(2.0)
    X = inv_sqrt2 * jnp.array([[1+0j, 1+0j],[1+0j, -1+0j]], dtype=CDTYPE)
    Y = inv_sqrt2 * jnp.array([[1+0j, 0-1j],[1+0j, 0+1j]], dtype=CDTYPE)
    Z = jnp.array([[1+0j, 0+0j],[0+0j, 1+0j]], dtype=CDTYPE)
    I = jnp.array([[1+0j, 0+0j],[0+0j, 1+0j]], dtype=CDTYPE)  # identity
    U = {"X": X, "Y": Y, "Z": Z, "I": I}
    for k, v in overrides.items():
        U[k] = as_complex_unitary(v)
    return U

def as_complex_unitary(U, device: Any = None):
    arr = jnp.asarray(U)
    if arr.ndim != 2 or arr.shape != (2, 2):
        raise ValueError(f"as_complex_unitary expects (2,2), got {tuple(arr.shape)}")
    return arr.astype(CDTYPE)

# ---------- hilbert helpers ----------
def _generate_hilbert_space(size: int) -> jnp.ndarray:
    """(2^size, size) bit-matrix in {0,1}, MSB-first. (FIX: descending arange step)"""
    size = int(size)
    n = 1 << size
    ar = jnp.arange(n, dtype=jnp.int64)
    # BUG WAS HERE: missing -1 step -> produced empty axis
    shifts = jnp.arange(size - 1, -1, -1, dtype=jnp.int64)
    return ((ar[:, None] >> shifts[None, :]) & 1).astype(DTYPE)

# ---------- RBM core ----------
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

@jax.jit
def _rbm_energy_sum(params_am, samples):
    return jnp.sum(_rbm_effective_energy(params_am, samples).astype(DTYPE))

@jax.jit
def _rbm_gibbs_steps(params: Dict[str, jnp.ndarray], k: int, initial_state: jnp.ndarray,
                     key: jax.Array, eps_p: float = 1e-6):
    """CD-k with fixed k (compile once)."""
    def body_fun(_, carry):
        v_curr, k_prev = carry
        k_prev, kh, kv = jax.random.split(k_prev, 3)
        h_prob = jax.nn.sigmoid(jnp.dot(v_curr, params["W"].T) + params["c"])
        h_prob = jnp.nan_to_num(h_prob, nan=0.5, posinf=1.0, neginf=0.0)
        h_prob = jnp.clip(h_prob, eps_p, 1.0 - eps_p)
        h = jax.random.bernoulli(kh, p=h_prob).astype(DTYPE)

        v_prob = jax.nn.sigmoid(jnp.dot(h, params["W"]) + params["b"])
        v_prob = jnp.nan_to_num(v_prob, nan=0.5, posinf=1.0, neginf=0.0)
        v_prob = jnp.clip(v_prob, eps_p, 1.0 - eps_p)
        v_next = jax.random.bernoulli(kv, p=v_prob).astype(DTYPE)
        return v_next, k_prev
    return jax.lax.fori_loop(0, k, body_fun, (initial_state.astype(DTYPE), key))

# ---------- complex wavefunction ----------
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

# ---------- rotation overlap with fixed shapes ----------
def _rot_overlap_logamp2_masked(params_am, params_ph, samples,
                                Uc_flat, sites, combos, mask_s, mask_c):
    """
    Fixed-shape kernel via padding/masks.
    Uc_flat : (Smax, 4) complex
    sites   : (Smax,) int32, -1 where padded
    combos  : (Cmax, Smax) float (0/1)
    mask_s  : (Smax,) {0,1}
    mask_c  : (Cmax,) {0,1}
    """
    B = samples.shape[0]
    Cmax, Smax = combos.shape[0], combos.shape[1]

    # Expand samples to branch states v: (Cmax, B, n)
    v = jnp.tile(samples[None, :, :], (Cmax, 1, 1))

    # Write the Smax rotated bits with a static loop; skip when mask_s=0
    def set_bit(s, vv):
        site = sites[s]
        use  = (mask_s[s] == 1)
        def updater(vv_):
            col = combos[:, s]  # (Cmax,)
            return vv_.at[:, :, site].set(col[:, None])
        return jax.lax.cond(use, updater, lambda x: x, vv)
    v = jax.lax.fori_loop(0, Smax, set_bit, v)

    # Energies on branches (Cmax,B)
    F_am = _rbm_effective_energy(params_am, v)
    F_ph = _rbm_effective_energy(params_ph, v)

    # Build Ut by chaining 2x2 lookups
    Ut = jnp.ones((Cmax, B), dtype=CDTYPE)

    def mult_site(s, Ut_curr):
        use = (mask_s[s] == 1)
        def do_mult(Ut_):
            site = sites[s]
            inp  = samples[:, site].astype(jnp.int32)            # (B,)
            outp = v[:, :, site].astype(jnp.int32)               # (Cmax,B)
            idx  = (inp[None, :] * 2 + outp).astype(jnp.int32)   # (Cmax,B)
            sel  = jnp.take(Uc_flat[s], idx, mode='clip')        # (Cmax,B)
            return Ut_ * sel
        return jax.lax.cond(use, do_mult, lambda x: x, Ut_curr)

    Ut = jax.lax.fori_loop(0, Smax, mult_site, Ut)
    Ut = Ut * mask_c[:, None]  # mask padded combos

    # Stable complex log-sum-exp of amplitudes:
    eps_u = 1e-300
    eps_s = 1e-12
    logmag_total = (-0.5 * F_am) + jnp.log(jnp.maximum(jnp.abs(Ut).astype(DTYPE), eps_u))
    phase_total  = (-0.5 * F_ph).astype(DTYPE) + jnp.angle(Ut).astype(DTYPE)

    # mask combos in the sum
    M = jnp.max(jnp.where(mask_c[:, None] == 1, logmag_total, -jnp.inf), axis=0, keepdims=True)
    scaled_mag = jnp.exp((logmag_total - M)) * mask_c[:, None]
    contrib = scaled_mag.astype(CDTYPE) * jnp.exp(1j * phase_total.astype(CDTYPE))
    S_prime = jnp.sum(contrib, axis=0)
    S_abs2  = jnp.maximum((jnp.conj(S_prime) * S_prime).real.astype(DTYPE), eps_s)
    return (2.0 * M.squeeze(0)).astype(DTYPE) + jnp.log(S_abs2)

@jax.jit
def _pos_loss_rot_masked(params_am, params_ph, samples,
                         Uc_flat, sites, combos, mask_s, mask_c):
    log_amp2 = _rot_overlap_logamp2_masked(params_am, params_ph, samples,
                                           Uc_flat, sites, combos, mask_s, mask_c)
    return -jnp.sum(log_amp2.astype(DTYPE))

@partial(jax.jit, static_argnames=("k",))
def _grads_step(params, pos_batch, neg_batch, rng,
                Uc_flat, sites, combos, mask_s, mask_c,
                is_z: jnp.bool_, k: int):
    vk, rng_out = _rbm_gibbs_steps(params["am"], k, neg_batch, rng)
    B_pos = pos_batch.shape[0]; B_neg = neg_batch.shape[0]

    def loss_fn(p):
        L_pos_z   = _rbm_energy_sum(p["am"], pos_batch)
        L_pos_rot = _pos_loss_rot_masked(p["am"], p["ph"], pos_batch, Uc_flat, sites, combos, mask_s, mask_c)
        L_pos = jax.lax.cond(is_z, lambda _: L_pos_z, lambda _: L_pos_rot, operand=None)
        L_neg = jnp.sum(_rbm_effective_energy(p["am"], vk).astype(DTYPE))
        return (L_pos / B_pos) - (L_neg / B_neg)

    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    return loss_val, grads, rng_out

# ---------- metrics ----------
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
    tgt = target.reshape(-1).astype(CDTYPE); nt = jnp.linalg.norm(tgt)
    if nt == 0: return 0.0
    tgt_norm = tgt / nt
    psi_norm_cd = nn_state.psi_complex_normalized(space).reshape(-1)
    KL_val = 0.0; eps = 1e-12

    def rotate_psi(basis, psi):
        us = [nn_state.U[b] for b in basis]
        x  = psi
        L  = x.shape[0]; y = x.reshape(L, 1)
        left = L
        for U in reversed(us):
            ns = U.shape[-1]
            left //= ns
            y = y.reshape(left, ns, -1)
            y = jnp.einsum('ij,ljm->lim', U, y).reshape(left * ns, -1)
        return y.reshape(*x.shape)

    for basis in bases:
        tgt_psi_r = rotate_psi(basis, tgt_norm)
        psi_r     = rotate_psi(basis, psi_norm_cd)
        nn_probs_r  = (jnp.abs(psi_r).astype(DTYPE)) ** 2
        tgt_probs_r = (jnp.abs(tgt_psi_r).astype(DTYPE)) ** 2
        p_sum = jnp.maximum(jnp.sum(tgt_probs_r), eps)
        q_sum = jnp.maximum(jnp.sum(nn_probs_r), eps)
        p = jnp.maximum(tgt_probs_r / p_sum, eps)
        q = jnp.maximum(nn_probs_r / q_sum, eps)
        KL_val += float(jnp.sum(p * (jnp.log(p) - jnp.log(q))))
    return KL_val / len(bases)

# ---------- dataset + homogeneous loader (constant shapes) ----------
class PerBasisTomographyDataset:
    """
    measurements/
      w_phase_state.txt              -> target psi: two columns (Re, Im)
      w_phase_<BASIS>_<shots>.txt    -> outcomes encoded by case (ZzZX -> [0,1,0,0] for ZZZX)
    """
    def __init__(self, directory="measurements", state_filename="w_phase_state.txt", file_prefix="w_phase_"):
        d = Path(directory)
        if not d.is_dir(): raise FileNotFoundError(f"Directory not found: {d}")
        psi_path = d / state_filename
        if not psi_path.exists(): raise FileNotFoundError(f"Missing state file: {psi_path}")
        psi_np = np.loadtxt(str(psi_path), dtype="float64")
        if psi_np.ndim != 2 or psi_np.shape[1] != 2:
            raise ValueError("State file must have two columns: Re Im.")
        self.target_state = jnp.asarray(psi_np[:, 0] + 1j * psi_np[:, 1], dtype=CDTYPE)

        per_basis_files = [p for p in d.glob("*.txt") if p.name != state_filename and p.name.startswith(file_prefix)]
        if not per_basis_files:
            raise FileNotFoundError(f"No per-basis files matching '{file_prefix}*' in {d}")

        samples: List[np.ndarray] = []
        bases_rows: List[Tuple[str, ...]] = []
        unique_bases: Dict[Tuple[str, ...], None] = {}

        for p in sorted(per_basis_files):
            bcode = self._parse_basis_code_from_filename(p.name, file_prefix)
            basis = tuple(list(bcode))
            with open(p, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if not lines: continue
            n = len(basis)
            for ln in lines:
                if len(ln) != n: raise ValueError(f"Line length != basis width in {p}: '{ln}'")
                bits = []
                for ch in ln:
                    if ch.isalpha(): bits.append(0 if ch.isupper() else 1)
                    else: raise ValueError(f"Illegal char '{ch}' in {p}")
                samples.append(np.asarray(bits, dtype=np.float64))
                bases_rows.append(basis)
            unique_bases[basis] = None

        if not samples: raise ValueError("No samples read from per-basis files.")
        self.train_samples = jnp.asarray(np.stack(samples, axis=0), dtype=DTYPE)
        self.train_bases: List[Tuple[str, ...]] = bases_rows
        self._unique_bases = list(unique_bases.keys())

        widths = {len(row) for row in self.train_bases}
        if len(widths) != 1: raise ValueError("Inconsistent basis widths across files.")
        n = next(iter(widths))
        if self.train_samples.shape[1] != n: raise ValueError("Sample width != basis width.")
        expected_dim = 1 << n
        if int(self.target_state.size) != expected_dim:
            raise ValueError(f"State length {int(self.target_state.size)} != 2^{n} ({expected_dim}).")

        z_mask = np.array([all(ch == "Z" for ch in row) for row in self.train_bases], dtype=bool)
        self._z_indices = jnp.asarray(np.nonzero(z_mask)[0], dtype=jnp.int32)
        if int(self._z_indices.size) == 0:
            raise ValueError("No Z-only rows found; needed for negative sampling.")

        self._groups: Dict[Tuple[str, ...], np.ndarray] = {}
        for i, row in enumerate(self.train_bases):
            self._groups.setdefault(tuple(row), []).append(i)
        for k in list(self._groups.keys()):
            self._groups[k] = np.asarray(self._groups[k], dtype=np.int32)

    @staticmethod
    def _parse_basis_code_from_filename(name: str, prefix: str) -> str:
        stem = Path(name).stem
        if not stem.startswith(prefix): raise ValueError(f"Bad file name (prefix): {name}")
        tail = stem[len(prefix):]
        if "_" not in tail: raise ValueError(f"Bad file name (missing shots part): {name}")
        code = tail.rsplit("_", 1)[0]
        if not re.fullmatch(r"[XYZI]+", code):
            raise ValueError(f"Basis code must be [XYZI]+, got '{code}' from {name}")
        return code

    # API
    def __len__(self): return int(self.train_samples.shape[0])
    def num_visible(self) -> int: return int(self.train_samples.shape[1])
    def z_indices(self) -> jnp.ndarray: return self._z_indices.copy()
    def eval_bases(self) -> List[Tuple[str, ...]]: return list(self._unique_bases)
    def target(self) -> jnp.ndarray: return self.target_state
    def groups(self): return self._groups

class RBMTomographyLoaderGrouped:
    """Homogeneous-basis batches with constant shapes (pads the last batch)."""
    def __init__(self, dataset: PerBasisTomographyDataset, pos_batch_size: int = 100,
                 neg_batch_size: Optional[int] = None, seed: Optional[int] = None):
        self.ds = dataset
        self.pos_bs = int(pos_batch_size)
        self.neg_bs = int(neg_batch_size or pos_batch_size)
        self._key = jax.random.PRNGKey(0 if seed is None else int(seed))

        n = self.ds.num_visible()
        for key in self.ds.groups().keys():
            if len(key) != n:
                raise ValueError("RBMTomographyLoaderGrouped: inconsistent basis widths")
        if self.ds.z_indices().size == 0:
            raise ValueError("RBMTomographyLoaderGrouped: Z-only pool is empty")

        self._plan()

    def _plan(self):
        planned = []
        for _, idxs in self.ds.groups().items():
            planned.append(ceil(idxs.shape[0] / self.pos_bs))
        self._planned_batches = int(np.sum(planned))

    def __len__(self): return self._planned_batches

    def iter_epoch(self):
        self._key, korder, kperm_epoch, kneg, kpad = jax.random.split(self._key, 5)
        bases = list(self.ds.groups().keys())
        order = np.asarray(jax.random.permutation(korder, jnp.arange(len(bases))).tolist(), dtype=int)
        bases = [bases[i] for i in order]

        total_batches = self.__len__()
        z_pool = self.ds.z_indices()
        pool_len = int(z_pool.size)
        neg_choices = jax.random.randint(kneg, shape=(total_batches * self.neg_bs,), minval=0, maxval=pool_len)
        neg_rows = z_pool[neg_choices]
        neg_samples_all = self.ds.train_samples[neg_rows].astype(DTYPE)

        nb_cursor = 0
        for basis in bases:
            idxs = self.ds.groups()[tuple(basis)]
            kperm_epoch, kbucket = jax.random.split(kperm_epoch)
            perm = np.asarray(jax.random.permutation(kbucket, jnp.arange(idxs.shape[0])).tolist(), dtype=int)
            idxs = idxs[perm]

            for start in range(0, idxs.shape[0], self.pos_bs):
                end = min(start + self.pos_bs, idxs.shape[0])
                pos_rows = idxs[start:end]
                # pad last chunk to fixed size
                if end - start < self.pos_bs:
                    need = self.pos_bs - (end - start)
                    take = np.asarray(jax.random.randint(kpad, shape=(need,), minval=0, maxval=idxs.shape[0]).tolist(), dtype=int)
                    pos_rows = np.concatenate([pos_rows, idxs[take]], axis=0)
                pos_batch = self.ds.train_samples[pos_rows].astype(DTYPE)

                nb_start = nb_cursor * self.neg_bs
                nb_end = nb_start + self.neg_bs
                neg_batch = neg_samples_all[nb_start:nb_end]
                nb_cursor += 1

                is_z = bool(all(ch in ("Z", "I") for ch in basis))
                yield pos_batch, neg_batch, tuple(basis), jnp.array(is_z, dtype=jnp.bool_)

# ---------- Trainer with padded meta cache ----------
class Trainer:
    def __init__(self, nn_state: ComplexWaveFunction, base_lr=1e-1,
                 phase_lr_scale=0.25, momentum=0.9, accum_steps=8, loader_for_shapes: RBMTomographyLoaderGrouped = None):
        self.nn = nn_state
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

        if loader_for_shapes is None:
            raise ValueError("Pass loader_for_shapes to Trainer to infer Smax/Cmax.")
        bases = loader_for_shapes.ds.eval_bases()
        self.Smax = max(sum(ch not in ("Z", "I") for ch in b) for b in bases)
        self.Cmax = 1 << self.Smax
        self._meta_cache: Dict[Tuple[str, ...], Tuple[jnp.ndarray, ...]] = {}
        self.U = self.nn.U  # shortcut

    def _padded_meta(self, basis: Tuple[str, ...]):
        if basis in self._meta_cache:
            return self._meta_cache[basis]
        # build padded Uc_flat (Smax,4), sites (Smax,), combos (Cmax,Smax), mask_s (Smax,), mask_c (Cmax,)
        sites_list = [i for i, b in enumerate(basis) if b not in ("Z", "I")]
        S = len(sites_list)
        # site-wise unitaries
        if S > 0:
            Ulist = [self.U[basis[i]].reshape(2, 2) for i in sites_list]
            Uc = jnp.stack(Ulist, axis=0)     # (S,2,2)
            Uc_flat = Uc.reshape(S, 4)
            combos_S = _generate_hilbert_space(S)  # (2^S, S)
        else:
            Uc_flat = jnp.zeros((0, 4), dtype=CDTYPE)
            combos_S = _generate_hilbert_space(0)  # (1,0)

        # pad to Smax / Cmax
        U_id_flat = jnp.array([1+0j, 0+0j, 0+0j, 1+0j], dtype=CDTYPE)
        pad_rows = self.Smax - Uc_flat.shape[0]
        if pad_rows > 0:
            Uc_flat = jnp.vstack([Uc_flat, jnp.tile(U_id_flat[None, :], (pad_rows, 1))])
        sites = np.array(sites_list + [-1] * (self.Smax - S), dtype=np.int32)
        mask_s = np.array(([1]*S) + ([0]*(self.Smax - S)), dtype=np.int32)

        C = 1 << S
        combos = jnp.zeros((self.Cmax, self.Smax), dtype=DTYPE)
        if S > 0:
            combos = combos.at[:C, :S].set(combos_S)
        mask_c = np.array(([1]*C) + ([0]*(self.Cmax - C)), dtype=np.int32)

        meta = (Uc_flat, jnp.asarray(sites, dtype=jnp.int32),
                combos, jnp.asarray(mask_s, dtype=jnp.int32),
                jnp.asarray(mask_c, dtype=jnp.int32))
        self._meta_cache[basis] = meta
        return meta

    def _zero_like_params(self):
        return tmap(jnp.zeros_like, self.nn.params)

    def fit(self, loader, epochs=70, k=10, log_every=5,
            target=None, bases=None, space=None,
            print_metrics=True,
            metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}"):
        history: Dict[str, list] = {"epoch": [], "Fidelity": [], "KL": []}

        for ep in range(1, epochs + 1):
            grads_accum = self._zero_like_params()
            micro = 0

            for pos_batch, neg_batch, basis_tuple, is_z in loader.iter_epoch():
                Uc_flat, sites, combos, mask_s, mask_c = self._padded_meta(basis_tuple)
                loss_val, grads, self.nn.rng = _grads_step(self.nn.params, pos_batch, neg_batch, self.nn.rng,
                                                           Uc_flat, sites, combos, mask_s, mask_c,
                                                           is_z=is_z, k=k)
                grads_accum = tmap(lambda a, b: a + b, grads_accum, grads)
                micro += 1

                if micro == self.accum_steps:
                    grads_avg = tmap(lambda g: g / self.accum_steps, grads_accum)
                    updates, self.opt_state = self.opt.update(grads_avg, self.opt_state, self.nn.params)
                    self.nn.params = optax.apply_updates(self.nn.params, updates)
                    grads_accum = self._zero_like_params()
                    micro = 0

            if micro > 0:
                grads_avg = tmap(lambda g: g / micro, grads_accum)
                updates, self.opt_state = self.opt.update(grads_avg, self.opt_state, self.nn.params)
                self.nn.params = optax.apply_updates(self.nn.params, updates)

            # force compute to sync
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

# ---------- main ----------
if __name__ == "__main__":
    # data
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

    # config (Torch-like)
    epochs = 70
    pbs = 20
    nbs = 20
    base_lr = 1e-1
    phase_lr_scale = 0.25
    momentum = 0.9
    k_cd = 10
    log_every = 5
    accum_steps = 6

    loader = RBMTomographyLoaderGrouped(data, pos_batch_size=pbs, neg_batch_size=nbs, seed=base_seed)
    space = nn_state.generate_hilbert_space()

    print("===== Run configuration =====")
    print(f"num_visible={nv} | num_hidden={nh} | pos_bs={pbs} | neg_bs={nbs} | k={k_cd}")
    print(f"lr_am={base_lr} | lr_ph={base_lr*phase_lr_scale} | momentum={momentum} | accum_steps={accum_steps}")
    print(f"samples={len(data)} | batches/epoch={len(loader)} | epochs={epochs} | log_every={log_every}")
    print("================================\n")

    trainer = Trainer(nn_state, base_lr=base_lr, phase_lr_scale=phase_lr_scale,
                      momentum=momentum, accum_steps=accum_steps, loader_for_shapes=loader)

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
        ax1.set_title("RBM Tomography – training metrics (JAX, fixed-shape rotations)")
        ax1.grid(True, alpha=0.3)
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="best")
        fig.tight_layout()
    else:
        print("No metrics to plot (did you pass target & bases and set log_every?).")

    plt.show()
