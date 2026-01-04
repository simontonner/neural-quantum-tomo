# ============================================================
# Simple JAX port with timed 1-epoch training
# - Groups batches by basis to enable stable JIT
# - JITs a single train_step (one compile per S in {0..n_vis})
# - Detailed per-batch timing + end-of-epoch breakdown
# - Uses x64/complex128 and optax SGD + clipping
# ============================================================

from math import ceil, sqrt, prod
from typing import Iterable, List, Tuple, Optional, Dict, Any
from functools import partial
import time
import numpy as np

# enable x64 before importing jax.numpy
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import optax

##### DEVICE AND DTYPES #####
DTYPE = jnp.float64
CDTYPE = jnp.complex128

##### SINGLE-QUBIT UNITARIES (as complex128) #####
def create_dict(**overrides):
    """Return {X,Y,Z} single-qubit unitaries as complex128."""
    inv_sqrt2 = 1.0 / sqrt(2.0)

    X = inv_sqrt2 * jnp.array([[1.0 + 0.0j, 1.0 + 0.0j],
                               [1.0 + 0.0j, -1.0 + 0.0j]], dtype=CDTYPE)

    Y = inv_sqrt2 * jnp.array([[1.0 + 0.0j, 0.0 - 1.0j],
                               [1.0 + 0.0j, 0.0 + 1.0j]], dtype=CDTYPE)

    Z = jnp.array([[1.0 + 0.0j, 0.0 + 0.0j],
                   [0.0 + 0.0j, 1.0 + 0.0j]], dtype=CDTYPE)

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

##### KRON-APPLY WITHOUT EXPLICIT TENSOR PRODUCT #####
def _kron_mult(matrices: List[jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    """Apply tensor product over sites of matrices to state/batch x without building the big matrix."""
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

def rotate_psi(nn_state, basis: Iterable[str], space: jnp.ndarray,
               unitaries: Optional[dict] = None, psi: Optional[jnp.ndarray] = None):
    """Rotate psi into a product basis specified by a tuple of 'X','Y','Z'."""
    n_vis = nn_state.num_visible
    basis = list(basis)
    if len(basis) != n_vis:
        raise ValueError(f"rotate_psi: basis length {len(basis)} != num_visible {n_vis}")

    if unitaries is None:
        us = [nn_state.U[b] for b in basis]
    else:
        Udict = {k: as_complex_unitary(v) for k, v in unitaries.items()}
        us = [Udict[b] for b in basis]

    if psi is None:
        x = nn_state.psi_complex(space)
    else:
        if psi.dtype != CDTYPE:
            raise TypeError("rotate_psi: psi must be complex128.")
        x = psi
    return _kron_mult(us, x)

##### BASIS HELPERS #####
def _generate_hilbert_space(size: int) -> jnp.ndarray:
    """(2^size, size) bit-matrix in {0,1}, MSB-first."""
    n = 1 << int(size)
    ar = jnp.arange(n, dtype=jnp.int64)
    shifts = jnp.arange(size - 1, -1, -1, dtype=jnp.int64)
    return ((ar[:, None] >> shifts[None, :]) & 1).astype(DTYPE)

def _basis_meta(Udict: Dict[str, jnp.ndarray], basis_tuple: Tuple[str, ...]):
    """Precompute tensors used by rotated overlap for a fixed basis."""
    sites = [i for i, b in enumerate(basis_tuple) if b != "Z"]
    S = len(sites)
    if S == 0:
        Uc_flat = jnp.zeros((0, 4), dtype=CDTYPE)
        combos = _generate_hilbert_space(0)  # (1,0)
        return Uc_flat, jnp.asarray([], dtype=jnp.int32), combos

    Ulist = [Udict[basis_tuple[i]].reshape(2, 2) for i in sites]
    Uc = jnp.stack(Ulist, axis=0)              # (S,2,2)
    Uc_flat = Uc.reshape(S, 4)                 # (S,4)
    combos = _generate_hilbert_space(S)        # (C,S), C=2^S
    return Uc_flat, jnp.asarray(sites, dtype=jnp.int32), combos

##### RBM PARAMS AND ENERGY #####
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
    """Free energy F(v). Accepts (..., n). Returns (...,)."""
    v = v.astype(DTYPE)
    W, b, c = params["W"], params["b"], params["c"]
    visible_bias_term = jnp.dot(v, b)
    hid_lin = jnp.dot(v, W.T) + c
    hid_term = jnp.sum(jax.nn.softplus(hid_lin), axis=-1)
    return -(visible_bias_term + hid_term)

@partial(jax.jit, static_argnames=("k",))
def _rbm_gibbs_steps(params: Dict[str, jnp.ndarray], k: int, initial_state: jnp.ndarray,
                     key: jax.Array) -> Tuple[jnp.ndarray, jax.Array]:
    """k-step block Gibbs from initial_state in {0,1}. 'k' is static for JIT."""
    W, b, c = params["W"], params["b"], params["c"]
    v0 = initial_state.astype(DTYPE)
    k_curr = key

    def body_fun(_, carry):
        v_curr, k_prev = carry
        k_prev, kh, kv = jax.random.split(k_prev, 3)
        h_prob = jax.nn.sigmoid(jnp.dot(v_curr, W.T) + c)
        h = jax.random.bernoulli(kh, p=jnp.clip(h_prob, 0.0, 1.0)).astype(DTYPE)
        v_prob = jax.nn.sigmoid(jnp.dot(h, W) + b)
        v_next = jax.random.bernoulli(kv, p=jnp.clip(v_prob, 0.0, 1.0)).astype(DTYPE)
        return v_next, k_prev

    vT, k_curr = jax.lax.fori_loop(0, k, body_fun, (v0, k_curr))
    return vT, k_curr

##### ROTATED OVERLAP (CACHED META) #####
@jax.jit
def _stable_log_overlap_amp2_with_meta(params_am: Dict[str, jnp.ndarray],
                                       params_ph: Dict[str, jnp.ndarray],
                                       samples: jnp.ndarray,
                                       Uc_flat: jnp.ndarray,  # (S,4)
                                       sites: jnp.ndarray,    # (S,)
                                       combos: jnp.ndarray    # (C,S)
                                       ) -> jnp.ndarray:
    """Return log |<basis outcome | psi>|^2 for a single fixed basis over a batch."""
    C = combos.shape[0]
    v = jnp.tile(samples[None, :, :], (C, 1, 1))
    v = v.at[:, :, sites].set(combos[:, None, :])

    F_am = _rbm_effective_energy(params_am, v)  # (C,B)
    F_ph = _rbm_effective_energy(params_ph, v)  # (C,B)

    if sites.size == 0:
        Ut = jnp.ones((C, samples.shape[0]), dtype=CDTYPE)  # C==1
    else:
        inp_sb = samples[:, sites].astype(jnp.int64).T
        outp_csb = v[:, :, sites].astype(jnp.int64).transpose(0, 2, 1)
        inp_csb = jnp.broadcast_to(inp_sb[None, :, :], outp_csb.shape)
        index_scb = (inp_csb * 2 + outp_csb).transpose(1, 0, 2)
        gathered = Uc_flat[jnp.arange(sites.size)[:, None, None], index_scb]  # (S,C,B)
        Ut = jnp.prod(gathered.transpose(1, 2, 0), axis=-1).astype(CDTYPE)    # (C,B)

    logmag_total = (-0.5 * F_am) + jnp.log(jnp.maximum(jnp.abs(Ut).astype(DTYPE), 1e-300))
    phase_total  = (-0.5 * F_ph).astype(CDTYPE) + jnp.angle(Ut).astype(CDTYPE)

    M = jnp.max(logmag_total, axis=0, keepdims=True)
    scaled_mag = jnp.exp((logmag_total - M))
    contrib = scaled_mag.astype(CDTYPE) * jnp.exp(1j * phase_total)
    S_prime = jnp.sum(contrib, axis=0)
    S_abs2  = (jnp.conj(S_prime) * S_prime).real.astype(DTYPE)
    return (2.0 * M.squeeze(0)).astype(DTYPE) + jnp.log(S_abs2 + 1e-6)

##### COMPLEX WAVE FUNCTION (ampl + phase RBM combined into ansatz) #####
class ComplexWaveFunction:
    """psi(sigma) = exp(-F_lambda(sigma)/2) * exp(-i F_mu(sigma)/2)."""

    def __init__(self, num_visible, num_hidden=None, unitary_dict=None, rng_seed: int = 0):
        self.rng = jax.random.PRNGKey(rng_seed)
        self.num_visible = int(num_visible)
        self.num_hidden  = int(num_hidden) if num_hidden else self.num_visible

        raw = unitary_dict if unitary_dict is not None else create_dict()
        self.U = {k: as_complex_unitary(v) for k, v in raw.items()}

        self.params = {
            "am": _init_rbm_params(self.num_visible, self.num_hidden, zero_weights=False, key=jax.random.PRNGKey(rng_seed+1)),
            "ph": _init_rbm_params(self.num_visible, self.num_hidden, zero_weights=False, key=jax.random.PRNGKey(rng_seed+2)),
        }

        self._stop_training = False
        self._max_size = 20
        self._basis_cache: Dict[Tuple[str, ...], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = {}
        self._seen_S = set()

    def reinitialize_parameters(self, seed: int = 0):
        self.params["am"] = _init_rbm_params(self.num_visible, self.num_hidden, key=jax.random.PRNGKey(seed+1))
        self.params["ph"] = _init_rbm_params(self.num_visible, self.num_hidden, key=jax.random.PRNGKey(seed+2))

    def psi_complex(self, v):
        E_lam = _rbm_effective_energy(self.params["am"], v)
        E_mu  = _rbm_effective_energy(self.params["ph"], v)
        amp = jnp.exp(-0.5 * E_lam).astype(DTYPE)
        ph  = (-0.5 * E_mu).astype(DTYPE)
        return amp.astype(CDTYPE) * jnp.exp(1j * ph.astype(CDTYPE))

    def psi_complex_normalized(self, v):
        E = _rbm_effective_energy(self.params["am"], v)
        ph = (-0.5 * _rbm_effective_energy(self.params["ph"], v)).astype(DTYPE)
        logZ = logsumexp(-E, axis=0)
        return jnp.exp(((-0.5 * E) - 0.5 * logZ).astype(CDTYPE) + 1j * ph.astype(CDTYPE))

    def generate_hilbert_space(self, size=None):
        size = self.num_visible if size is None else int(size)
        if size > self._max_size:
            raise ValueError(f"Hilbert space too large (n={size} > max={self._max_size}).")
        return _generate_hilbert_space(size)

    def get_basis_meta(self, basis_tuple: Tuple[str, ...]):
        if basis_tuple not in self._basis_cache:
            self._basis_cache[basis_tuple] = _basis_meta(self.U, basis_tuple)
        return self._basis_cache[basis_tuple]

##### LOSS (JIT-FRIENDLY) #####
@jax.jit
def _pos_loss_z(params_am: Dict[str, jnp.ndarray], samples: jnp.ndarray) -> jnp.ndarray:
    """Sum of Epos over Z-only batch."""
    return jnp.sum(_rbm_effective_energy(params_am, samples).astype(DTYPE))

@jax.jit
def _pos_loss_rot(params_am: Dict[str, jnp.ndarray], params_ph: Dict[str, jnp.ndarray],
                  samples: jnp.ndarray,
                  Uc_flat: jnp.ndarray, sites: jnp.ndarray, combos: jnp.ndarray) -> jnp.ndarray:
    """-sum(log_amp2) for a fixed non-Z basis."""
    log_amp2 = _stable_log_overlap_amp2_with_meta(params_am, params_ph, samples, Uc_flat, sites, combos)
    return -jnp.sum(log_amp2.astype(DTYPE))

@partial(jax.jit, static_argnames=("k", "is_z"))
def _train_step(params: Dict[str, Dict[str, jnp.ndarray]],
                opt_state,
                pos_batch: jnp.ndarray,
                neg_batch: jnp.ndarray,
                rng: jax.Array,
                Uc_flat: jnp.ndarray, sites: jnp.ndarray, combos: jnp.ndarray,
                k: int,
                is_z: bool):
    """One SGD step with CD-k. is_z chooses Z-only vs rotated positive phase."""
    vk, rng = _rbm_gibbs_steps(params["am"], k, neg_batch, rng)
    B_neg = neg_batch.shape[0]
    B_pos = pos_batch.shape[0]

    def loss_fn(p):
        if is_z:
            L_pos = _pos_loss_z(p["am"], pos_batch)
        else:
            L_pos = _pos_loss_rot(p["am"], p["ph"], pos_batch, Uc_flat, sites, combos)
        L_neg = jnp.sum(_rbm_effective_energy(p["am"], vk).astype(DTYPE))
        return (L_pos / B_pos) - (L_neg / B_neg)

    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    return loss_val, grads, rng

##### METRICS #####
def fidelity(nn_state, target, space=None, bases=None, **kwargs):
    """Return squared magnitude of overlap(target, psi) with both normalized (diagnostic only)."""
    if target.dtype != CDTYPE:
        raise TypeError("fidelity: target must be complex128")
    space = nn_state.generate_hilbert_space() if space is None else space
    psi = nn_state.psi_complex_normalized(space).reshape(-1)
    tgt = target.reshape(-1).astype(CDTYPE)
    npsi = jnp.linalg.norm(psi); nt = jnp.linalg.norm(tgt)
    if (npsi == 0) or (nt == 0):
        return 0.0
    psi_n = psi / npsi; tgt_n = tgt / nt
    inner = jnp.vdot(tgt_n, psi_n)
    return float(jnp.abs(inner) ** 2)

def KL(nn_state, target, space=None, bases=None, **kwargs):
    """Average KL divergence across bases (diagnostic only)."""
    if bases is None:
        raise ValueError("KL needs bases")
    if target.dtype != CDTYPE:
        raise TypeError("KL: target must be complex128")
    space = nn_state.generate_hilbert_space() if space is None else space
    tgt = target.reshape(-1).astype(CDTYPE)
    nt = jnp.linalg.norm(tgt);
    if nt == 0:
        return 0.0
    tgt_norm = tgt / nt
    psi_norm_cd = nn_state.psi_complex_normalized(space).reshape(-1)

    KL_val = 0.0
    eps = 1e-12
    for basis in bases:
        tgt_psi_r = rotate_psi(nn_state, basis, space, psi=tgt_norm)
        psi_r = rotate_psi(nn_state, basis, space, psi=psi_norm_cd)
        nn_probs_r = (jnp.abs(psi_r).astype(DTYPE)) ** 2
        tgt_probs_r = (jnp.abs(tgt_psi_r).astype(DTYPE)) ** 2
        p_sum = jnp.maximum(jnp.sum(tgt_probs_r), eps)
        q_sum = jnp.maximum(jnp.sum(nn_probs_r), eps)
        p = jnp.maximum(tgt_probs_r / p_sum, eps)
        q = jnp.maximum(nn_probs_r / q_sum, eps)
        KL_val += float(jnp.sum(p * (jnp.log(p) - jnp.log(q))))
    return KL_val / len(bases)

##### DATASET AND LOADER (GROUPED BY BASIS) #####
class TomographyDataset:
    """Container for measurement samples, bases, and target psi."""
    def __init__(self, train_path, psi_path, train_bases_path, bases_path):
        self.train_samples = jnp.asarray(np.loadtxt(train_path, dtype="float32"), dtype=DTYPE)
        psi_np = np.loadtxt(psi_path, dtype="float64")
        self.target_state = jnp.asarray(psi_np[:, 0] + 1j * psi_np[:, 1], dtype=CDTYPE)
        self.train_bases = np.loadtxt(train_bases_path, dtype=str)
        self.bases = np.loadtxt(bases_path, dtype=str, ndmin=1)

        tb = np.asarray(self.train_bases, dtype=object)
        z_mask_np = (tb == "Z").all(axis=1)
        self._z_mask = jnp.asarray(z_mask_np, dtype=bool)
        self._z_indices = jnp.where(self._z_mask)[0].astype(jnp.int32)

        if self.train_samples.shape[0] != len(self.train_bases):
            raise ValueError("TomographyDataset: sample count != basis row count")
        if self._z_indices.size == 0:
            raise ValueError("TomographyDataset: no Z-only rows for negative sampling")

        widths = {len(row) for row in self.train_bases}
        if len(widths) != 1:
            raise ValueError("TomographyDataset: inconsistent basis widths")
        n = next(iter(widths))
        if n != self.train_samples.shape[1]:
            raise ValueError("TomographyDataset: basis width != sample width")

        # pre-group indices by basis for JIT-friendly batches
        self._groups: Dict[Tuple[str, ...], np.ndarray] = {}
        for i, row in enumerate(np.asarray(self.train_bases, dtype=object)):
            key = tuple(row.tolist())
            if key not in self._groups:
                self._groups[key] = []
            self._groups[key].append(i)
        for k in list(self._groups.keys()):
            self._groups[k] = np.asarray(self._groups[k], dtype=np.int32)

    def __len__(self):
        return int(self.train_samples.shape[0])

    def num_visible(self) -> int:
        return int(self.train_samples.shape[1])

    def z_indices(self) -> jnp.ndarray:
        return self._z_indices.copy()

    def eval_bases(self) -> List[Tuple[str, ...]]:
        return [tuple(row) for row in np.asarray(self.bases, dtype=object)]

    def target(self) -> jnp.ndarray:
        return self.target_state

    def groups(self):
        return self._groups

class RBMTomographyLoaderGrouped:
    """Yield homogeneous-basis batches: (pos_batch, neg_batch, basis_tuple, is_z)."""
    def __init__(self, dataset: TomographyDataset, pos_batch_size: int = 100, neg_batch_size: Optional[int] = None,
                 seed: Optional[int] = None):
        self.ds = dataset
        self.pos_bs = int(pos_batch_size)
        self.neg_bs = int(neg_batch_size or pos_batch_size)
        self._key = jax.random.PRNGKey(0 if seed is None else int(seed))

        n = self.ds.num_visible()
        for key in self.ds.groups().keys():
            if len(key) != n:
                raise ValueError("RBMTomographyLoaderGrouped: inconsistent basis widths in dataset")
        if self.ds.z_indices().size == 0:
            raise ValueError("RBMTomographyLoaderGrouped: Z-only pool is empty (need negatives)")

        self._planned_batches = None
        self._plan_batches()

    def _plan_batches(self):
        planned = []
        for _, idxs in self.ds.groups().items():
            planned.append(ceil(idxs.shape[0] / self.pos_bs))
        self._planned_batches = int(np.sum(planned))

    def __len__(self):
        return self._planned_batches

    def iter_epoch(self):
        """Iterate once, yielding homogeneous-basis batches in random basis order."""
        self._key, korder, kneg = jax.random.split(self._key, 3)

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
            # shuffle indices inside the bucket with fresh key
            self._key, kperm = jax.random.split(self._key)
            perm = np.asarray(jax.random.permutation(kperm, jnp.arange(idxs.shape[0])).tolist(), dtype=int)
            idxs = idxs[perm]

            for start in range(0, idxs.shape[0], self.pos_bs):
                end = min(start + self.pos_bs, idxs.shape[0])
                pos_rows = idxs[start:end]
                pos_batch = self.ds.train_samples[pos_rows].astype(DTYPE)

                nb_start = nb_cursor * self.neg_bs
                nb_end = nb_start + self.neg_bs
                neg_batch = neg_samples_all[nb_start:nb_end]
                nb_cursor += 1

                is_z = all(ch == "Z" for ch in basis)
                yield pos_batch, neg_batch, tuple(basis), bool(is_z)

##### TRAIN LOOP (ONE EPOCH, JIT STEP) #####
class Trainer:
    def __init__(self, nn_state: ComplexWaveFunction, lr=1e-1):
        self.nn = nn_state
        self.opt = optax.chain(optax.clip_by_global_norm(10.0), optax.sgd(lr))
        self.opt_state = self.opt.init(self.nn.params)

    def step(self, pos_batch, neg_batch, basis_tuple, is_z, k):
        Uc_flat, sites, combos = self.nn.get_basis_meta(basis_tuple)
        loss_val, grads, self.nn.rng = _train_step(self.nn.params, self.opt_state,
                                                   pos_batch, neg_batch, self.nn.rng,
                                                   Uc_flat, sites, combos,
                                                   k=k, is_z=is_z)
        updates, self.opt_state = self.opt.update(grads, self.opt_state, self.nn.params)
        self.nn.params = optax.apply_updates(self.nn.params, updates)
        return loss_val

    def fit_one_epoch(self, loader, k=10, target=None, bases=None, space=None,
                      log_batches=True, metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}"):
        epoch_t0 = time.perf_counter()
        sum_t_batch = 0.0
        n_batches = 0

        for b_idx, (pos_batch, neg_batch, basis_tuple, is_z) in enumerate(loader.iter_epoch(), start=1):
            batch_t0 = time.perf_counter()
            S = sum(1 for b in basis_tuple if b != "Z")
            compile_tag = ""
            if S not in self.nn._seen_S:
                compile_tag = f" (includes JIT compile for S={S})"
                self.nn._seen_S.add(S)

            loss_val = self.step(pos_batch, neg_batch, basis_tuple, is_z, k)
            jnp.asarray(loss_val).block_until_ready()  # sync for honest timing
            t_end = time.perf_counter()

            n_batches += 1
            total = t_end - batch_t0
            sum_t_batch += total

            if log_batches:
                print(f"[Batch {b_idx}] step={total:.4f}s{compile_tag}")

        epoch_t1 = time.perf_counter()
        metrics_time = 0.0
        if target is not None and bases is not None:
            m0 = time.perf_counter()
            fid_val = fidelity(self.nn, target, space=space, bases=bases)
            m1 = time.perf_counter()
            kl_val = KL(self.nn, target, space=space, bases=bases)
            m2 = time.perf_counter()
            metrics_time = (m2 - m0)
            print(metric_fmt.format(ep=1, fid=fid_val, kl=kl_val) + f" | metrics_fid={m1-m0:.4f}s | metrics_kl={m2-m1:.4f}s")

        avg_batch = sum_t_batch / max(1, n_batches)
        print("\n===== Timing breakdown (epoch 1) =====")
        print(f"batches: {n_batches}")
        print(f"step total: total={sum_t_batch:.4f}s | avg={avg_batch:.4f}s")
        print(f"metrics total: {metrics_time:.4f}s")
        print(f"epoch wall time (incl metrics): {epoch_t1 - epoch_t0 + metrics_time:.4f}s")
        print("=====================================\n")

##### STANDALONE TRAINING SCRIPT (EXAMPLE, NO PLOTTING) #####
if __name__ == "__main__":
    # file paths
    train_path = "w_state_meas.txt"
    train_bases_path = "w_state_basis.txt"
    psi_path = "w_state_aug.txt"
    bases_path = "w_state_bases.txt"

    # data
    data = TomographyDataset(train_path, psi_path, train_bases_path, bases_path)
    U = create_dict()

    nv = int(data.train_samples.shape[1])
    nh = nv
    nn_state = ComplexWaveFunction(num_visible=nv, num_hidden=nh, unitary_dict=U, rng_seed=1234)

    # config
    epochs = 1
    pbs = 100
    nbs = 100
    lr = 1e-1
    k_cd = 10

    loader = RBMTomographyLoaderGrouped(data, pos_batch_size=pbs, neg_batch_size=nbs, seed=1234)
    space = nn_state.generate_hilbert_space()

    print("===== Run configuration =====")
    print(f"num_visible={nv} | num_hidden={nh} | pos_bs={pbs} | neg_bs={nbs} | k={k_cd} | lr={lr}")
    print(f"samples={len(data)} | batches={len(loader)}")
    print("=============================\n")

    trainer = Trainer(nn_state, lr=lr)
    trainer.fit_one_epoch(loader, k=k_cd,
                          target=data.target(), bases=data.eval_bases(), space=space,
                          log_batches=True,
                          metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}")
