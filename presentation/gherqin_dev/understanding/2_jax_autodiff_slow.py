# ============================================================
# JAX TOMOGRAPHY TRAINER — Torch-Style, with Fast Warmup & Prints
# ============================================================

from math import ceil, sqrt, prod
from typing import Iterable, List, Tuple, Optional, Dict, Any
import time
import numpy as np

# ------------------------------------------------------------
# ENABLE 64-BIT FLOATS (must happen before importing jax.numpy)
# ------------------------------------------------------------
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import optax


##### DEVICE AND DTYPES #####
DTYPE = jnp.float64
CDTYPE = jnp.complex128


##### SINGLE-QUBIT UNITARIES (as complex128) #####
def create_dict(**overrides) -> Dict[str, jnp.ndarray]:
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


def as_complex_unitary(U: Any) -> jnp.ndarray:
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
    """
    Apply tensor product over sites of matrices to state/batch x without building the big matrix.

    matrices: list of (2,2) complex matrices for each site (length n)
    x: vectorized state with leading dimension 2**n; trailing dims treated as batch
    """
    if any(m.dtype != CDTYPE for m in matrices):
        raise TypeError("unitaries must be complex128")
    if x.dtype != CDTYPE:
        raise TypeError("x must be complex128")

    L = x.shape[0]
    batch = int(x.size // L)
    y = x.reshape(L, batch)

    n = [m.shape[-1] for m in matrices]
    if prod(n) != L:
        raise ValueError(f"Incompatible sizes: expected leading dim {prod(n)}, got {L}")

    left = L
    for U in reversed(matrices):
        ns = U.shape[-1]
        left //= ns
        y = y.reshape(left, ns, -1)
        y = jnp.einsum('ij,ljm->lim', U, y).reshape(left * ns, -1)

    return y.reshape(*x.shape)


def rotate_psi(nn_state: "ComplexWaveFunction", basis: Iterable[str], space: jnp.ndarray,
               unitaries: Optional[dict] = None, psi: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Rotate psi into a product basis specified by a tuple/list of 'X','Y','Z'.
    space: (2**n, n) bit-matrix for computational basis (MSB-first). Used only when psi=None.
    """
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
        raise TypeError("rotate_psi: psi must be complex128")
    return _kron_mult(us, x)


##### BASIS-BRANCH ENUMERATION FOR LOCAL PRODUCT MEASUREMENTS #####
def _generate_bitstrings(num_bits: int) -> jnp.ndarray:
    """Return (2**num_bits, num_bits) with MSB-first binary encodings in {0,1}."""
    n = 1 << num_bits
    ar = jnp.arange(n, dtype=jnp.int32)
    shifts = jnp.arange(num_bits - 1, -1, -1, dtype=jnp.int32)
    return ((ar[:, None] >> shifts[None, :]) & 1).astype(jnp.int32)


def _rotate_basis_state(nn_state: "ComplexWaveFunction",
                        basis: Iterable[str],
                        states: jnp.ndarray,
                        unitaries: Optional[Dict[str, jnp.ndarray]] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Enumerate coherent branches for a measured batch of states under the given basis.

    states: (B, n) in {0,1}
    Returns:
        Ut: (C, B) product of local rotation matrix elements for each branch c
        v:  (C, B, n) branch states consistent with rotated outcomes
    """
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
    Ulist = [src[basis_seq[i]].reshape(2, 2) for i in sites]  # (S, 2, 2)
    Uc = jnp.stack(Ulist, axis=0)                              # (S, 2, 2)
    Uc_flat = Uc.reshape(Uc.shape[0], 4)                       # (S, 4)
    S = len(sites)
    B = states.shape[0]
    C = 1 << S

    combos = _generate_bitstrings(S)                           # (C, S)
    v = jnp.tile(states[None, :, :], (C, 1, 1))                # (C, B, n)
    v = v.at[:, :, jnp.array(sites)].set(combos[:, None, :])   # fill rotated sites

    # gather input/output bits and multiply matrix elements across sites
    inp_sb = states[:, jnp.array(sites)].astype(jnp.int32).T                       # (S, B)
    outp_csb = v[:, :, jnp.array(sites)].astype(jnp.int32).transpose(0, 2, 1)      # (C, S, B)
    inp_csb = jnp.broadcast_to(inp_sb[None, :, :], (C, S, B))                      # (C, S, B)

    index_scb = (inp_csb * 2 + outp_csb).transpose(1, 0, 2)                        # (S, C, B)
    gathered = Uc_flat[jnp.arange(S)[:, None, None], index_scb]                    # (S, C, B)
    gathered = gathered.transpose(1, 2, 0)                                          # (C, B, S)
    Ut = jnp.prod(gathered, axis=-1).astype(CDTYPE)                                 # (C, B)
    return Ut, v


def _convert_basis_element_to_index(states: jnp.ndarray) -> jnp.ndarray:
    """Map rows in {0,1}^n to flat indices [0, 2^n - 1] (MSB-first)."""
    s = states.astype(jnp.int64)
    n = s.shape[-1]
    shifts = jnp.arange(n - 1, -1, -1, dtype=jnp.int64)
    return jnp.sum((s << shifts), axis=-1)


def rotate_psi_inner_prod(nn_state: "ComplexWaveFunction",
                          basis: Iterable[str],
                          states: jnp.ndarray,
                          unitaries: Optional[Dict[str, jnp.ndarray]] = None,
                          psi: Optional[jnp.ndarray] = None,
                          include_extras: bool = False):
    """Compute overlap for measured outcomes in the given basis: sum_c Ut[c] * psi(branch_state[c])."""
    Ut, v = _rotate_basis_state(nn_state, basis, states, unitaries=unitaries)

    if psi is None:
        psi_sel = nn_state.psi_complex(v)  # (C, B)
    else:
        if psi.dtype != CDTYPE:
            raise TypeError("rotate_psi_inner_prod: psi must be complex128")
        idx = _convert_basis_element_to_index(v).astype(jnp.int64)  # (C, B)
        psi_sel = psi[idx]  # (C, B)

    Upsi_v_c = Ut * psi_sel
    Upsi_c = jnp.sum(Upsi_v_c, axis=0)

    return (Upsi_c, Upsi_v_c, v) if include_extras else Upsi_c


##### BINARY RBM #####
def init_rbm_params(num_visible: int,
                    num_hidden: Optional[int] = None,
                    zero_weights: bool = False,
                    key: Optional[jax.random.PRNGKey] = None) -> Dict[str, jnp.ndarray]:
    """Initialize RBM parameters. Optionally all-zero for debugging."""
    num_hidden = int(num_hidden) if num_hidden is not None else int(num_visible)
    scale = 1.0 / np.sqrt(num_visible)

    if key is None:
        key = jax.random.PRNGKey(0)
    key, kW = jax.random.split(key)

    if zero_weights:
        W = jnp.zeros((num_hidden, num_visible), dtype=DTYPE)
    else:
        W = scale * jax.random.normal(kW, (num_hidden, num_visible), dtype=DTYPE)

    b = jnp.zeros((num_visible,), dtype=DTYPE)
    c = jnp.zeros((num_hidden,), dtype=DTYPE)

    return {"W": W, "b": b, "c": c}


def rbm_effective_energy(params: Dict[str, jnp.ndarray], v: jnp.ndarray) -> jnp.ndarray:
    """Free energy F(v). Accepts (..., n). Returns (...,)."""
    v = v.astype(DTYPE)
    W, b, c = params["W"], params["b"], params["c"]
    visible_bias_term = jnp.dot(v, b)                 # (...,)
    hid_lin = jnp.dot(v, W.T) + c                     # (..., h)
    hid_term = jnp.sum(jax.nn.softplus(hid_lin), axis=-1)  # (...,)
    out = -(visible_bias_term + hid_term)
    return out


def rbm_gibbs_steps(params: Dict[str, jnp.ndarray],
                    k: int,
                    initial_state: jnp.ndarray,
                    key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jax.random.PRNGKey]:
    """k-step block Gibbs from initial_state in {0,1}."""
    W, b, c = params["W"], params["b"], params["c"]
    v = initial_state.astype(DTYPE)
    key_loc = key
    for _ in range(int(k)):
        key_loc, kh, kv = jax.random.split(key_loc, 3)
        h_prob = jax.nn.sigmoid(jnp.dot(v, W.T) + c)
        h = jax.random.bernoulli(kh, p=jnp.clip(h_prob, 0.0, 1.0)).astype(DTYPE)
        v_prob = jax.nn.sigmoid(jnp.dot(h, W) + b)
        v = jax.random.bernoulli(kv, p=jnp.clip(v_prob, 0.0, 1.0)).astype(DTYPE)
    return jax.lax.stop_gradient(v), key_loc


##### COMPLEX WAVE FUNCTION (ampl + phase RBM combined into ansatz) #####
class ComplexWaveFunction:
    """psi(sigma) = exp(-F_lambda(sigma)/2) * exp(-i * F_mu(sigma)/2)."""

    def __init__(self,
                 num_visible: int,
                 num_hidden: Optional[int] = None,
                 unitary_dict: Optional[Dict[str, jnp.ndarray]] = None,
                 rng_seed: int = 0):
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden) if num_hidden else self.num_visible

        self.U = {k: as_complex_unitary(v) for k, v in (unitary_dict or create_dict()).items()}

        key = jax.random.PRNGKey(rng_seed)
        key, k_am, k_ph, k_run = jax.random.split(key, 4)
        self.params = {
            "am": init_rbm_params(self.num_visible, self.num_hidden, zero_weights=False, key=k_am),
            "ph": init_rbm_params(self.num_visible, self.num_hidden, zero_weights=False, key=k_ph),
        }

        self._stop_training = False
        self._max_size = 20
        self._rng = k_run  # for Gibbs

    # control
    @property
    def stop_training(self) -> bool:
        return self._stop_training

    @stop_training.setter
    def stop_training(self, new_val: bool):
        if isinstance(new_val, bool):
            self._stop_training = new_val
        else:
            raise ValueError("stop_training must be bool")

    @property
    def max_size(self) -> int:
        return self._max_size

    def reinitialize_parameters(self, seed: int = 0):
        key = jax.random.PRNGKey(seed)
        key, k_am, k_ph, k_run = jax.random.split(key, 4)
        self.params["am"] = init_rbm_params(self.num_visible, self.num_hidden, key=k_am)
        self.params["ph"] = init_rbm_params(self.num_visible, self.num_hidden, key=k_ph)
        self._rng = k_run

    # psi accessors
    def amplitude(self, v: jnp.ndarray) -> jnp.ndarray:
        """abs(psi(v)) = exp(-F_lambda(v)/2)."""
        E = rbm_effective_energy(self.params["am"], v)
        return jnp.exp(-0.5 * E).astype(DTYPE)

    def phase(self, v: jnp.ndarray) -> jnp.ndarray:
        """phase(v) = -F_mu(v)/2 (real)."""
        E_mu = rbm_effective_energy(self.params["ph"], v)
        return -0.5 * E_mu

    def psi_complex(self, v: jnp.ndarray) -> jnp.ndarray:
        """Complex psi(v). v can be (B, n) or (C,B,n)."""
        E_lam = rbm_effective_energy(self.params["am"], v)
        E_mu = rbm_effective_energy(self.params["ph"], v)
        amp = jnp.exp(-0.5 * E_lam).astype(DTYPE)
        ph = -0.5 * E_mu
        return amp.astype(CDTYPE) * jnp.exp(1j * ph.astype(CDTYPE))

    def psi_complex_normalized(self, v: jnp.ndarray) -> jnp.ndarray:
        """Normalized psi(v) using exact log Z_lambda (safe only for small n)."""
        E = rbm_effective_energy(self.params["am"], v)
        ph = -0.5 * rbm_effective_energy(self.params["ph"], v)
        logZ = jax.scipy.special.logsumexp(-E, axis=0)
        return jnp.exp(((-0.5 * E) - 0.5 * logZ).astype(CDTYPE) + 1j * ph.astype(CDTYPE))

    # aliases
    def psi(self, v: jnp.ndarray) -> jnp.ndarray:
        return self.psi_complex(v)

    def psi_normalized(self, v: jnp.ndarray) -> jnp.ndarray:
        return self.psi_complex_normalized(v)

    def phase_angle(self, v: jnp.ndarray) -> jnp.ndarray:
        return self.phase(v)

    # utilities
    def generate_hilbert_space(self, size: Optional[int] = None) -> jnp.ndarray:
        """Enumerate computational basis as (2^size, size) bit-matrix in {0,1}."""
        size = self.num_visible if size is None else int(size)
        if size > self._max_size:
            raise ValueError(f"Hilbert space too large (n={size} > max={self._max_size}).")
        return _generate_bitstrings(size).astype(DTYPE)

    # stable overlap for rotated bases
    def _stable_log_overlap_amp2(self, basis: Tuple[str, ...], states: jnp.ndarray, eps_rot: float = 1e-6) -> jnp.ndarray:
        """Stable log of squared overlap for rotated-basis outcomes via complex log-sum-exp."""
        Ut, v = _rotate_basis_state(self, basis, states)
        F_am = rbm_effective_energy(self.params["am"], v)
        F_ph = rbm_effective_energy(self.params["ph"], v)

        logmag_total = (-0.5 * F_am) + jnp.log(jnp.maximum(jnp.abs(Ut).astype(DTYPE), 1e-300))
        phase_total = (-0.5 * F_ph).astype(CDTYPE) + jnp.angle(Ut).astype(CDTYPE)

        M = jnp.max(logmag_total, axis=0, keepdims=True)
        scaled_mag = jnp.exp((logmag_total - M))
        contrib = scaled_mag.astype(CDTYPE) * jnp.exp(1j * phase_total)
        S_prime = jnp.sum(contrib, axis=0)
        S_abs2 = (jnp.conj(S_prime) * S_prime).real.astype(DTYPE)
        log_amp2 = (2.0 * M.squeeze(0)).astype(DTYPE) + jnp.log(S_abs2 + eps_rot)
        return log_amp2  # (B,)

    # loss pieces
    def _positive_phase_loss(self, samples: jnp.ndarray, bases_batch: List[Tuple[str, ...]], eps_rot: float = 1e-6) -> jnp.ndarray:
        """Data term: Z-basis NLL plus rotated-basis likelihood."""
        buckets: Dict[Tuple[str, ...], List[int]] = {}
        for i, row in enumerate(bases_batch):
            buckets.setdefault(tuple(row), []).append(i)

        loss_rot = jnp.array(0.0, dtype=DTYPE)
        loss_z = jnp.array(0.0, dtype=DTYPE)

        for basis_t, idxs in buckets.items():
            idxs_t = jnp.array(idxs, dtype=jnp.int32)
            if any(ch != "Z" for ch in basis_t):
                log_amp2 = self._stable_log_overlap_amp2(basis_t, samples[idxs_t], eps_rot=eps_rot)
                loss_rot = loss_rot - jnp.sum(log_amp2.astype(DTYPE))
            else:
                Epos = rbm_effective_energy(self.params["am"], samples[idxs_t])
                loss_z = loss_z + jnp.sum(Epos.astype(DTYPE))

        return (loss_rot + loss_z).astype(DTYPE)

    def _negative_phase_loss(self, k: int, neg_init: jnp.ndarray, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, int, jax.random.PRNGKey]:
        """CD-k negative phase for amplitude RBM."""
        vk, key = rbm_gibbs_steps(self.params["am"], k, neg_init, key)
        Eneg = rbm_effective_energy(self.params["am"], vk)
        return jnp.sum(Eneg.astype(DTYPE)), int(vk.shape[0]), key

    # warmup (forces compilation so you see progress immediately later)
    def warmup(self, loader: "RBMTomographyLoader", k: int = 10):
        try:
            pos_batch, neg_batch, bases_batch = next(loader.iter_epoch())
        except StopIteration:
            return
        # Force-compile core pieces
        neg_loss_sum, B_neg, self._rng = self._negative_phase_loss(k, neg_batch, self._rng)
        L_pos = self._positive_phase_loss(pos_batch, bases_batch)
        dummy = (L_pos / float(pos_batch.shape[0])) - (neg_loss_sum / float(B_neg))
        _ = dummy.block_until_ready()

    # training loop
    def fit(self,
            loader: "RBMTomographyLoader",
            epochs: int = 70,
            k: int = 10,
            lr: float = 1e-1,
            log_every: int = 5,
            target: Optional[jnp.ndarray] = None,
            bases: Optional[List[Tuple[str, ...]]] = None,
            space: Optional[jnp.ndarray] = None,
            print_metrics: bool = True,
            metric_fmt: str = "Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}") -> Dict[str, List[float]]:
        """Autodiff CD training with rotated-basis likelihood and CD-k negative phase."""
        opt = optax.sgd(lr)
        opt_state = opt.init(self.params)

        history: Dict[str, List[float]] = {"epoch": []}
        if target is not None:
            history["Fidelity"], history["KL"] = [], []

        if space is None:
            space = self.generate_hilbert_space()

        # -------- Warmup compile (visible) --------
        print("🔧 Compiling JAX kernels (warmup)...")
        t0 = time.time()
        self.warmup(loader, k=k)
        print(f"✅ Warmup done in {time.time() - t0:.2f}s")

        for ep in range(1, epochs + 1):
            tot_loss = 0.0
            n_batches = 0

            for pos_batch, neg_batch, bases_batch in loader.iter_epoch():
                # sample negative phase once per minibatch (no grad through Gibbs)
                neg_loss_sum, B_neg, self._rng = self._negative_phase_loss(k, neg_batch, self._rng)

                def loss_with_params(params):
                    saved = self.params
                    try:
                        self.params = params
                        L_pos = self._positive_phase_loss(pos_batch, bases_batch)
                        B_pos = float(pos_batch.shape[0])
                        return (L_pos / B_pos) - (neg_loss_sum / B_neg)
                    finally:
                        self.params = saved

                loss_val, grads = jax.value_and_grad(loss_with_params)(self.params)
                # make sure we don't hide async compute before printing loss
                loss_val = float(loss_val.block_until_ready())
                tot_loss += loss_val
                n_batches += 1

                updates, opt_state = opt.update(grads, opt_state, self.params)
                self.params = optax.apply_updates(self.params, updates)

                if self.stop_training:
                    break

            avg_loss = tot_loss / max(1, n_batches)
            # Always show something each epoch:
            print(f"Epoch {ep}/{epochs} │ train_loss: {avg_loss:.6e}", flush=True)

            if (target is not None) and (ep % log_every == 0):
                fid_val = fidelity(self, target, space=space, bases=bases)
                kl_val = KL(self, target, space=space, bases=bases)
                history["epoch"].append(ep)
                history["Fidelity"].append(float(fid_val))
                history["KL"].append(float(kl_val))
                if print_metrics:
                    print(metric_fmt.format(ep=ep, fid=fid_val, kl=kl_val))

            if self.stop_training:
                break

        return history


##### METRICS #####
def fidelity(nn_state: ComplexWaveFunction,
             target: jnp.ndarray,
             space: Optional[jnp.ndarray] = None,
             **kwargs) -> float:
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

    inner = jnp.vdot(tgt_n, psi_n)  # conj(tgt_n) * psi_n sum
    return float(jnp.abs(inner) ** 2)


def KL(nn_state: ComplexWaveFunction,
       target: jnp.ndarray,
       space: Optional[jnp.ndarray] = None,
       bases: Optional[List[Tuple[str, ...]]] = None,
       **kwargs) -> float:
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

    def __init__(self,
                 train_path: str,
                 psi_path: str,
                 train_bases_path: str,
                 bases_path: str):

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

    def __init__(self, dataset: TomographyDataset, pos_batch_size: int = 100,
                 neg_batch_size: Optional[int] = None, seed: Optional[int] = None):
        self.ds = dataset
        self.pos_bs = int(pos_batch_size)
        self.neg_bs = int(neg_batch_size or pos_batch_size)
        self._key = jax.random.PRNGKey(0 if seed is None else int(seed))

        n = self.ds.num_visible()
        bases_seq = self.ds.train_bases_as_tuples()
        if any(len(row) != n for row in bases_seq):
            raise ValueError("RBMTomographyLoader: inconsistent basis widths in dataset")
        if self.ds.z_indices().size == 0:
            raise ValueError("RBMTomographyLoader: Z-only pool is empty (need negatives)")

    def set_seed(self, seed: Optional[int]):
        """Optional deterministic shuffling."""
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

            if int(pos_batch.shape[1]) != self.ds.num_visible() or int(neg_batch.shape[1]) != self.ds.num_visible():
                raise RuntimeError("Loader invariant broken: batch width != num_visible")

            yield pos_batch, neg_batch, bases_batch


##### STANDALONE TRAINING SCRIPT (EXAMPLE, NO PLOTTING) #####
if __name__ == "__main__":
    train_path = "w_state_meas.txt"
    train_bases_path = "w_state_basis.txt"
    psi_path = "w_state_aug.txt"
    bases_path = "w_state_bases.txt"

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

    loader = RBMTomographyLoader(data, pos_batch_size=pbs, neg_batch_size=nbs, seed=1234)
    space = nn_state.generate_hilbert_space()

    history = nn_state.fit(loader, epochs=epochs, k=k_cd, lr=lr, log_every=log_every,
                           target=data.target(), bases=data.eval_bases(), space=space,
                           print_metrics=True,
                           metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}")
