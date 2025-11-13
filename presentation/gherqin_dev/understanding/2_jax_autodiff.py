# ============================================================
# Simple JAX port of the Torch tomography experiment (single file)
# - Keeps the same structure/names as much as possible
# - Uses x64 / complex128 to avoid dtype warnings
# - Uses optax SGD + global-norm clipping
# - Prints epoch time + Fidelity + KL at log_every
# ============================================================

from math import ceil, sqrt, prod
from typing import Iterable, List, Tuple, Optional, Dict, Any

import time
import numpy as np

# ---- enable x64 before importing jax.numpy ----
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import optax


##### DEVICE AND DTYPES #####
DTYPE = jnp.float64          # RBM energies in float64 for stability
CDTYPE = jnp.complex128      # complex dtype


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


def inverse(z: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Safe complex inverse: conj(z) / max(|z|^2, eps)."""
    zz = z.astype(CDTYPE)
    return jnp.conj(zz) / jnp.maximum(jnp.abs(zz) ** 2, eps)


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
        # einsum identical to Torch 'ij,ljm->lim'
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


##### BASIS-BRANCH ENUMERATION FOR LOCAL PRODUCT MEASUREMENTS #####
def _generate_hilbert_space(size: int) -> jnp.ndarray:
    """(2^size, size) bit-matrix in {0,1}, MSB-first."""
    n = 1 << int(size)
    ar = jnp.arange(n, dtype=jnp.int64)
    shifts = jnp.arange(size - 1, -1, -1, dtype=jnp.int64)
    return ((ar[:, None] >> shifts[None, :]) & 1).astype(DTYPE)


def _rotate_basis_state(nn_state, basis, states, unitaries=None):
    """Enumerate coherent branches for a measured batch of states under the given basis."""
    n_vis = nn_state.num_visible
    basis_seq = list(basis)

    if len(basis_seq) != n_vis:
        raise ValueError(f"_rotate_basis_state: basis length {len(basis_seq)} != num_visible {n_vis}")
    if states.shape[-1] != n_vis:
        raise ValueError(f"_rotate_basis_state: states width {states.shape[-1]} != num_visible {n_vis}")

    sites = [i for i, b in enumerate(basis_seq) if b != "Z"]
    if len(sites) == 0:
        v = states[None, ...]
        Ut = jnp.ones(v.shape[:-1], dtype=CDTYPE)
        return Ut, v

    src = nn_state.U if unitaries is None else {k: as_complex_unitary(v) for k, v in unitaries.items()}
    Ulist = [src[basis_seq[i]].reshape(2, 2) for i in sites]
    Uc = jnp.stack(Ulist, axis=0)     # (S,2,2)
    Uc_flat = Uc.reshape(Uc.shape[0], 4)  # (S,4)

    S = len(sites)
    B = states.shape[0]
    C = 1 << S

    combos = _generate_hilbert_space(S)   # (C, S)

    v = jnp.tile(states[None, :, :], (C, 1, 1))  # (C,B,n)
    v = v.at[:, :, jnp.array(sites)].set(combos[:, None, :])

    inp_sb = states[:, jnp.array(sites)].astype(jnp.int64).T             # (S,B)
    outp_csb = v[:, :, jnp.array(sites)].astype(jnp.int64).transpose(0, 2, 1)  # (C,S,B)
    inp_csb = jnp.broadcast_to(inp_sb[None, :, :], (C, S, B))            # (C,S,B)

    index_scb = (inp_csb * 2 + outp_csb).transpose(1, 0, 2)              # (S,C,B)
    gathered = Uc_flat[jnp.arange(S)[:, None, None], index_scb]          # (S,C,B)
    gathered = gathered.transpose(1, 2, 0)                                # (C,B,S)
    Ut = jnp.prod(gathered, axis=-1).astype(CDTYPE)                       # (C,B)

    return Ut, v


def _convert_basis_element_to_index(states):
    """Map rows in {0,1}^n to flat indices [0, 2^n - 1] (MSB-first)."""
    s = states.round().astype(jnp.int64)
    n = s.shape[-1]
    shifts = jnp.arange(n - 1, -1, -1, dtype=jnp.int64)
    return jnp.sum((s << shifts), axis=-1)


def rotate_psi_inner_prod(nn_state, basis, states, unitaries=None, psi=None, include_extras=False):
    """Compute overlap for measured outcomes in the given basis: sum_c Ut[c] * psi(branch_state[c])."""
    Ut, v = _rotate_basis_state(nn_state, basis, states, unitaries=unitaries)

    if psi is None:
        psi_sel = nn_state.psi_complex(v)
    else:
        if psi.dtype != CDTYPE:
            raise TypeError("rotate_psi_inner_prod: psi must be complex.")
        idx = _convert_basis_element_to_index(v).astype(jnp.int64)
        psi_sel = psi[idx]

    Upsi_v_c = Ut * psi_sel
    Upsi_c = jnp.sum(Upsi_v_c, axis=0)

    if include_extras:
        return Upsi_c, Upsi_v_c, v
    return Upsi_c


##### BINARY RBM (functional params) #####
def _init_rbm_params(num_visible: int, num_hidden: Optional[int] = None,
                     zero_weights: bool = False, key: Optional[jax.Array] = None):
    nv = int(num_visible)
    nh = int(num_hidden) if num_hidden is not None else nv
    scale = 1.0 / np.sqrt(nv)

    if key is None:
        key = jax.random.PRNGKey(0)
    kW, = jax.random.split(key, 1)

    if zero_weights:
        W = jnp.zeros((nh, nv), dtype=DTYPE)
    else:
        W = scale * jax.random.normal(kW, (nh, nv), dtype=DTYPE)

    b = jnp.zeros((nv,), dtype=DTYPE)
    c = jnp.zeros((nh,), dtype=DTYPE)
    return {"W": W, "b": b, "c": c}


def _rbm_effective_energy(params: Dict[str, jnp.ndarray], v: jnp.ndarray) -> jnp.ndarray:
    """Free energy F(v). Accepts (..., n). Returns (...,)."""
    v = v.astype(DTYPE)
    W, b, c = params["W"], params["b"], params["c"]
    visible_bias_term = jnp.dot(v, b)  # (...,)
    hid_lin = jnp.dot(v, W.T) + c      # (..., h)
    hid_term = jnp.sum(jax.nn.softplus(hid_lin), axis=-1)  # (...,)
    out = -(visible_bias_term + hid_term)
    return out


def _rbm_gibbs_steps(params: Dict[str, jnp.ndarray], k: int, initial_state: jnp.ndarray,
                     key: jax.Array) -> Tuple[jnp.ndarray, jax.Array]:
    """k-step block Gibbs from initial_state in {0,1}."""
    W, b, c = params["W"], params["b"], params["c"]
    v = initial_state.astype(DTYPE)
    k_curr = key
    for _ in range(int(k)):
        k_curr, kh, kv = jax.random.split(k_curr, 3)
        h_prob = jax.nn.sigmoid(jnp.dot(v, W.T) + c)
        h = jax.random.bernoulli(kh, p=jnp.clip(h_prob, 0.0, 1.0)).astype(DTYPE)
        v_prob = jax.nn.sigmoid(jnp.dot(h, W) + b)
        v = jax.random.bernoulli(kv, p=jnp.clip(v_prob, 0.0, 1.0)).astype(DTYPE)
    return v, k_curr


##### COMPLEX WAVE FUNCTION (ampl + phase RBM combined into ansatz) #####
class ComplexWaveFunction:
    """psi(sigma) = exp(-F_lambda(sigma)/2) * exp(-i F_mu(sigma)/2)."""

    def __init__(self, num_visible, num_hidden=None, unitary_dict=None, rng_seed: int = 0):
        self.rng = jax.random.PRNGKey(rng_seed)
        self.num_visible = int(num_visible)
        self.num_hidden  = int(num_hidden) if num_hidden else self.num_visible

        raw = unitary_dict if unitary_dict is not None else create_dict()
        self.U = {k: as_complex_unitary(v) for k, v in raw.items()}

        # params as nested dicts
        self.params = {
            "am": _init_rbm_params(self.num_visible, self.num_hidden, zero_weights=False, key=jax.random.PRNGKey(rng_seed+1)),
            "ph": _init_rbm_params(self.num_visible, self.num_hidden, zero_weights=False, key=jax.random.PRNGKey(rng_seed+2)),
        }

        self._stop_training = False
        self._max_size = 20

    # control
    @property
    def stop_training(self):
        return self._stop_training

    @stop_training.setter
    def stop_training(self, new_val):
        if isinstance(new_val, bool):
            self._stop_training = new_val
        else:
            raise ValueError("stop_training must be bool")

    @property
    def max_size(self):
        return self._max_size

    def reinitialize_parameters(self, seed: int = 0):
        self.params["am"] = _init_rbm_params(self.num_visible, self.num_hidden, key=jax.random.PRNGKey(seed+1))
        self.params["ph"] = _init_rbm_params(self.num_visible, self.num_hidden, key=jax.random.PRNGKey(seed+2))

    # psi accessors
    def amplitude(self, v):
        E = _rbm_effective_energy(self.params["am"], v)
        return jnp.exp(-0.5 * E).astype(DTYPE)

    def phase(self, v):
        E_mu = _rbm_effective_energy(self.params["ph"], v)
        return (-0.5 * E_mu).astype(DTYPE)

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

    # aliases
    def psi(self, v):  return self.psi_complex(v)
    def psi_normalized(self, v):  return self.psi_complex_normalized(v)
    def phase_angle(self, v):  return self.phase(v)

    # utilities
    def generate_hilbert_space(self, size=None):
        size = self.num_visible if size is None else int(size)
        if size > self._max_size:
            raise ValueError(f"Hilbert space too large (n={size} > max={self._max_size}).")
        return _generate_hilbert_space(size)

    # stable overlap for rotated bases
    def _stable_log_overlap_amp2(self, basis: Tuple[str, ...], states: jnp.ndarray, eps_rot: float = 1e-6, unitaries=None):
        """Stable log of squared overlap for rotated-basis outcomes via complex log-sum-exp."""
        Ut, v = _rotate_basis_state(self, basis, states, unitaries=unitaries)
        F_am = _rbm_effective_energy(self.params["am"], v)
        F_ph = _rbm_effective_energy(self.params["ph"], v)

        logmag_total = (-0.5 * F_am) + jnp.log(jnp.maximum(jnp.abs(Ut).astype(DTYPE), 1e-300))
        phase_total  = (-0.5 * F_ph).astype(CDTYPE) + jnp.angle(Ut).astype(CDTYPE)

        M = jnp.max(logmag_total, axis=0, keepdims=True)
        scaled_mag = jnp.exp((logmag_total - M))
        contrib = scaled_mag.astype(CDTYPE) * jnp.exp(1j * phase_total)
        S_prime = jnp.sum(contrib, axis=0)
        S_abs2  = (jnp.conj(S_prime) * S_prime).real.astype(DTYPE)
        log_amp2 = (2.0 * M.squeeze(0)).astype(DTYPE) + jnp.log(S_abs2 + eps_rot)
        return log_amp2

    # loss pieces
    def _positive_phase_loss(self, samples: jnp.ndarray, bases_batch: List[Tuple[str, ...]], eps_rot: float = 1e-6):
        """Data term: Z-basis NLL plus rotated-basis likelihood."""
        # bucket by identical basis rows (Python like Torch)
        buckets: Dict[Tuple[str, ...], List[int]] = {}
        for i, row in enumerate(bases_batch):
            buckets.setdefault(tuple(row), []).append(i)

        loss_rot = jnp.array(0.0, dtype=DTYPE)
        loss_z   = jnp.array(0.0, dtype=DTYPE)

        for basis_t, idxs in buckets.items():
            idxs_t = jnp.array(idxs, dtype=jnp.int32)
            if any(ch != "Z" for ch in basis_t):
                log_amp2 = self._stable_log_overlap_amp2(basis_t, samples[idxs_t], eps_rot=eps_rot)
                loss_rot = loss_rot - jnp.sum(log_amp2.astype(DTYPE))
            else:
                Epos = _rbm_effective_energy(self.params["am"], samples[idxs_t])
                loss_z = loss_z + jnp.sum(Epos.astype(DTYPE))

        return (loss_rot + loss_z).astype(DTYPE)

    def _negative_phase_stats(self, k: int, neg_init: jnp.ndarray, key: jax.Array):
        """Return vk (CD-k) and its energy sum (needs params gradients)."""
        vk, key = _rbm_gibbs_steps(self.params["am"], k, neg_init, key)
        Eneg = _rbm_effective_energy(self.params["am"], vk)
        return jnp.sum(Eneg.astype(DTYPE)), int(vk.shape[0]), vk, key

    # training loop
    def fit(self, loader, epochs=70, k=10, lr=1e-1, log_every=5,
            optimizer: Optional[optax.GradientTransformation] = None, optimizer_args=None,
            target=None, bases=None, space=None, print_metrics=True,
            metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}"):

        if self.stop_training:
            return {"epoch": []}

        if optimizer is None:
            # mimic Torch SGD + grad clipping 10.0
            optimizer = optax.chain(optax.clip_by_global_norm(10.0), optax.sgd(lr))
        opt = optimizer
        opt_state = opt.init(self.params)

        history = {"epoch": []}
        if target is not None:
            if target.dtype != CDTYPE:
                target = target.astype(CDTYPE)
            history["Fidelity"], history["KL"] = [], []

        if space is None:
            space = self.generate_hilbert_space()

        for ep in range(1, epochs + 1):
            t0 = time.time()
            tot_loss = 0.0
            n_batches = 0

            for pos_batch, neg_batch, bases_batch in loader.iter_epoch():
                # negative phase: sample vk without gradient
                neg_sum, B_neg, vk, self.rng = self._negative_phase_stats(k, neg_batch, self.rng)

                # define loss with current vk constant; grads flow through params
                def loss_fn(params):
                    # temporarily swap params
                    saved = self.params
                    self.params = params
                    try:
                        L_pos = self._positive_phase_loss(pos_batch, bases_batch)
                        B_pos = float(pos_batch.shape[0])
                        # Eneg must be recomputed with 'params' for grad (vk is constant)
                        Eneg = _rbm_effective_energy(self.params["am"], vk)
                        L_neg = jnp.sum(Eneg.astype(DTYPE))
                        return (L_pos / B_pos) - (L_neg / B_neg)
                    finally:
                        self.params = saved

                loss_val, grads = jax.value_and_grad(loss_fn)(self.params)
                updates, opt_state = opt.update(grads, opt_state, self.params)
                self.params = optax.apply_updates(self.params, updates)

                tot_loss += float(loss_val)
                n_batches += 1

                if self.stop_training:
                    break

            # epoch metrics
            avg_loss = tot_loss / max(1, n_batches)

            if (target is not None) and (ep % log_every == 0):
                fid_val = fidelity(self, target, space=space, bases=bases)
                kl_val  = KL(self, target, space=space, bases=bases)
                history["epoch"].append(ep)
                history["Fidelity"].append(fid_val)
                history["KL"].append(kl_val)
                if print_metrics:
                    elapsed = time.time() - t0
                    print(metric_fmt.format(ep=ep, fid=fid_val, kl=kl_val) + f" | {elapsed:.2f}s")
            else:
                elapsed = time.time() - t0
                if print_metrics:
                    print(f"Epoch {ep}: Loss = {avg_loss:.6f} | {elapsed:.2f}s")

            if self.stop_training:
                break

        return history


##### METRICS #####
def fidelity(nn_state, target, space=None, **kwargs):
    """Return squared magnitude of overlap(target, psi) with both normalized (diagnostic only)."""
    if target.dtype != CDTYPE:
        raise TypeError("fidelity: target must be complex128")

    space = nn_state.generate_hilbert_space() if space is None else space

    psi = nn_state.psi_complex_normalized(space).reshape(-1)
    tgt = target.reshape(-1).astype(CDTYPE)

    npsi = jnp.linalg.norm(psi)
    nt = jnp.linalg.norm(tgt)
    if (npsi == 0) or (nt == 0):
        return 0.0

    psi_n = psi / npsi
    tgt_n = tgt / nt

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
    nt = jnp.linalg.norm(tgt)
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


##### DATASET AND LOADER #####
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

    def __len__(self):
        return int(self.train_samples.shape[0])

    def num_visible(self) -> int:
        return int(self.train_samples.shape[1])

    def z_indices(self) -> jnp.ndarray:
        return self._z_indices.copy()

    def train_bases_as_tuples(self) -> List[Tuple[str, ...]]:
        return [tuple(row) for row in np.asarray(self.train_bases, dtype=object)]

    def eval_bases(self) -> List[Tuple[str, ...]]:
        return [tuple(row) for row in np.asarray(self.bases, dtype=object)]

    def target(self) -> jnp.ndarray:
        return self.target_state


class RBMTomographyLoader:
    """Yield (pos_batch, neg_batch, bases_batch) minibatches for training."""

    def __init__(self, dataset: TomographyDataset, pos_batch_size: int = 100, neg_batch_size: Optional[int] = None,
                 strict: bool = True, seed: Optional[int] = None):
        self.ds = dataset
        self.pos_bs = int(pos_batch_size)
        self.neg_bs = int(neg_batch_size or pos_batch_size)
        self.strict = strict
        self._key = jax.random.PRNGKey(0 if seed is None else int(seed))

        n = self.ds.num_visible()
        bases_seq = self.ds.train_bases_as_tuples()
        if any(len(row) != n for row in bases_seq):
            raise ValueError("RBMTomographyLoader: inconsistent basis widths in dataset")
        if self.ds.z_indices().size == 0:
            raise ValueError("RBMTomographyLoader: Z-only pool is empty (need negatives)")

    def set_seed(self, seed: Optional[int]):
        self._key = jax.random.PRNGKey(np.random.randint(2**31 - 1)) if seed is None else jax.random.PRNGKey(int(seed))

    def __len__(self):
        return ceil(len(self.ds) / self.pos_bs)

    def iter_epoch(self):
        """Iterate once over the dataset, yielding aligned batches."""
        N = len(self.ds)
        n_batches = ceil(N / self.pos_bs)

        self._key, kperm, kneg = jax.random.split(self._key, 3)

        perm = jax.random.permutation(kperm, jnp.arange(N, dtype=jnp.int32))
        pos_samples = self.ds.train_samples[perm].astype(DTYPE)

        bases_list = self.ds.train_bases_as_tuples()
        perm_idx = np.array(perm, dtype=int).tolist()
        pos_bases_perm = [bases_list[i] for i in perm_idx]

        z_pool = self.ds.z_indices()
        pool_len = int(z_pool.size)
        neg_choices = jax.random.randint(kneg, shape=(n_batches * self.neg_bs,), minval=0, maxval=pool_len)
        neg_rows = z_pool[neg_choices]
        neg_samples_all = self.ds.train_samples[neg_rows].astype(DTYPE)

        for b in range(n_batches):
            start = b * self.pos_bs
            end = min(start + self.pos_bs, N)
            pos_batch = pos_samples[start:end]

            nb_start = b * self.neg_bs
            nb_end = nb_start + self.neg_bs
            neg_batch = neg_samples_all[nb_start:nb_end]

            bases_batch = pos_bases_perm[start:end]

            if self.strict:
                if len(bases_batch) != pos_batch.shape[0]:
                    raise RuntimeError("Loader invariant broken: bases_batch length mismatch")
                if pos_batch.shape[1] != self.ds.num_visible():
                    raise RuntimeError("Loader invariant broken: pos_batch width != num_visible")
                if neg_batch.shape[1] != self.ds.num_visible():
                    raise RuntimeError("Loader invariant broken: neg_batch width != num_visible")

            yield pos_batch, neg_batch, bases_batch


##### STANDALONE TRAINING SCRIPT (EXAMPLE, NO PLOTTING) #####
if __name__ == "__main__":
    # Replace these with your file paths
    train_path = "w_state_meas.txt"
    train_bases_path = "w_state_basis.txt"
    psi_path = "w_state_aug.txt"
    bases_path = "w_state_bases.txt"

    # Data
    data = TomographyDataset(train_path, psi_path, train_bases_path, bases_path)
    U = create_dict()

    nv = data.num_visible()
    nh = nv
    nn_state = ComplexWaveFunction(num_visible=nv, num_hidden=nh, unitary_dict=U, rng_seed=1234)

    epochs = 70
    pbs = 100
    nbs = 100
    lr = 1e-1
    k_cd = 10
    log_every = 5

    loader = RBMTomographyLoader(data, pos_batch_size=pbs, neg_batch_size=nbs, strict=True, seed=1234)
    space = nn_state.generate_hilbert_space()

    history = nn_state.fit(loader, epochs=epochs, k=k_cd, lr=lr, log_every=log_every,
                           optimizer=None, optimizer_args=None,
                           target=data.target(), bases=data.eval_bases(), space=space,
                           print_metrics=True,
                           metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}")
