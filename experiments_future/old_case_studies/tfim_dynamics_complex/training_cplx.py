# Complex Conditional RBM (VMC) for TFIM in JAX/Flax
# ---------------------------------------------------
# This rewrites your classical conditional RBM (with Gibbs/CD) into a
# *complex-valued* conditional RBM trained as a *variational quantum state*
# via Metropolis sampling and energy minimization.
#
# Key changes vs your original:
# - Parameters (W, b, c) are complex; conditioner outputs complex modulations.
# - Sampling is Metropolis-Hastings on |psi|^2 (no Gibbs/sigmoid/softplus).
# - Loss is the variational energy ⟨H⟩ for TFIM (not free-energy contrast).
# - Local energy uses amplitude ratios for single-spin flips.
# - Works with per-sample conditioning field `m` (your `field`), so you can
#   sweep/learn a single conditional model across many m values.

import site
from pathlib import Path
site.addsitedir(str(Path.cwd().parents[2]))

import csv
from pathlib import Path
from typing import Tuple, Dict, Any, Callable, Optional

import jax
import jax.lax
import jax.numpy as jnp
from jax.random import PRNGKey
import optax
import flax

from flax.training.train_state import TrainState
from flax import linen as nn

import matplotlib.pyplot as plt

from presentation.measurement.data_loading import load_measurements, MixedDataLoader

# ---------------------------------------------------
# Utilities
# ---------------------------------------------------

def bits_to_spins(bits: jnp.ndarray) -> jnp.ndarray:
    """Map {0,1} -> {-1,+1} elementwise.
    Accepts shape (B,N) or (N,). Returns same shape, dtype float32.
    """
    return 2.0 * bits.astype(jnp.float32) - 1.0


def make_lattice_bonds(Lx: int, Ly: int, periodic: bool = False) -> jnp.ndarray:
    """Return nearest-neighbour bond list for a 2D Lx x Ly lattice.
    Each bond is a pair (i,j) with i<j, flattened in row-major order.
    """
    def idx(x, y):
        return x * Ly + y

    bonds = []
    for x in range(Lx):
        for y in range(Ly):
            i = idx(x, y)
            # +x neighbour
            if x + 1 < Lx:
                j = idx(x + 1, y)
                bonds.append((i, j))
            elif periodic:
                j = idx(0, y)
                bonds.append((i, j))
            # +y neighbour
            if y + 1 < Ly:
                j = idx(x, y + 1)
                bonds.append((i, j))
            elif periodic:
                j = idx(x, 0)
                bonds.append((i, j))
    return jnp.array(bonds, dtype=jnp.int32)


# ---------------------------------------------------
# Complex Conditional RBM wavefunction
# ---------------------------------------------------

class ComplexConditionalRBM(nn.Module):
    num_visible: int
    num_hidden: int
    conditioner_width: int = 64
    dtype: Any = jnp.complex64  # complex64 is faster; use complex128 if you like

    def setup(self):
        # Complex parameter initializers: small random complex (JAX-friendly)
        def cplx_init(scale=0.01, dtype=jnp.complex64):
            def _init(key, shape):
                kr, ki = jax.random.split(key)
                real = jax.random.normal(kr, shape, dtype=jnp.float32) * scale
                imag = jax.random.normal(ki, shape, dtype=jnp.float32) * scale
                return (real + 1j * imag).astype(dtype)
            return _init

        self.W = self.param("W", cplx_init(0.01, self.dtype), (self.num_visible, self.num_hidden))
        self.b = self.param("b", cplx_init(0.01, self.dtype), (self.num_visible,))   # visible biases
        self.c = self.param("c", cplx_init(0.01, self.dtype), (self.num_hidden,))   # hidden biases

    @nn.compact
    def conditioner(self, field: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return complex (b_mod, c_mod) with shape (B,N) and (B,H).
        We emit gamma/beta for both b and c, for real & imag parts, then combine.
        """
        # Ensure (B,1)
        x = field[:, None] if field.ndim == 1 else field
        x = nn.Dense(self.conditioner_width)(x)
        x = nn.tanh(x)
        # We need 8 segments totaling 4*(N+H):
        # [γ_b^R(N), β_b^R(N), γ_c^R(H), β_c^R(H), γ_b^I(N), β_b^I(N), γ_c^I(H), β_c^I(H)]
        N, H = self.num_visible, self.num_hidden
        out_dim = 4 * (N + H)
        x = nn.Dense(out_dim)(x)
        # jnp.split returns len(boundaries)+1 parts → use 7 boundaries to get 8 tensors.
        boundaries = [
            N,                 # γ_b^R (N)
            2 * N,             # β_b^R (N)
            2 * N + H,         # γ_c^R (H)
            2 * N + 2 * H,     # β_c^R (H)
            3 * N + 2 * H,     # γ_b^I (N)
            4 * N + 2 * H,     # β_b^I (N)
            4 * N + 3 * H,     # γ_c^I (H)
        ]
        (
            gamma_b_r, beta_b_r, gamma_c_r, beta_c_r,
            gamma_b_i, beta_b_i, gamma_c_i, beta_c_i,
        ) = jnp.split(x, boundaries, axis=-1)

        # Build complex modulations
        gamma_b = gamma_b_r + 1j * gamma_b_i
        beta_b  = beta_b_r  + 1j * beta_b_i
        gamma_c = gamma_c_r + 1j * gamma_c_i
        beta_c  = beta_c_r  + 1j * beta_c_i

        # Apply to base biases (broadcast (B,1) with (N,) → (B,N))
        b_mod = (1.0 + gamma_b) * self.b[None, :] + beta_b
        c_mod = (1.0 + gamma_c) * self.c[None, :] + beta_c
        return b_mod.astype(self.dtype), c_mod.astype(self.dtype)

    # ------------------------ Wavefunction ------------------------

    @nn.nowrap
    def log_psi(self, v: jnp.ndarray, field: jnp.ndarray) -> jnp.ndarray:
        """Complex log amplitude log ψ(v|field).
        v must be spins in {-1,+1}, shape (B,N). field shape (B,).
        Returns shape (B,), complex.
        """
        assert v.ndim == 2
        assert field.ndim in (1, 2)
        if field.ndim == 2:
            assert field.shape[1] == 1
            field = field[:, 0]

        b_mod, c_mod = self.conditioner(field)
        theta = (v.astype(self.dtype) @ self.W) + c_mod  # (B,H)
        # log ψ = v·b + sum_j log(2 cosh(theta_j))
        log2 = jnp.log(jnp.array(2.0, dtype=self.dtype))
        return jnp.sum(v.astype(self.dtype) * b_mod, axis=-1) + jnp.sum(jnp.log(jnp.cosh(theta)) + log2, axis=-1)

    @nn.nowrap
    def delta_logpsi_all_flips(self, v: jnp.ndarray, field: jnp.ndarray) -> jnp.ndarray:
        """Δ logψ for flipping each spin i: logψ(v^i) - logψ(v).
        Returns shape (B,N), complex. Uses a vectorized formula.
        """
        b_mod, c_mod = self.conditioner(field)              # (B,N), (B,H)
        v_c = v.astype(self.dtype)
        theta = (v_c @ self.W) + c_mod                      # (B,H)
        # For each site i: theta' = theta - 2*v_i * W[i,:]
        W_rows = self.W[None, :, :]                         # (1,N,H)
        vW = v_c[:, :, None] * W_rows                       # (B,N,H)
        theta_prime = theta[:, None, :] - 2.0 * vW          # (B,N,H)
        # Sum over hidden: logcosh(theta') - logcosh(theta)
        d_hidden = jnp.sum(jnp.log(jnp.cosh(theta_prime)) - jnp.log(jnp.cosh(theta))[:, None, :], axis=-1)  # (B,N)
        # Visible bias change: -2*v_i*b_i
        d_visible = -2.0 * v_c * b_mod                      # (B,N)
        return (d_visible + d_hidden).astype(self.dtype)    # (B,N)


# ---------------------------------------------------
# Metropolis sampler on |ψ|^2
# ---------------------------------------------------

def log_prob_from_logpsi(logpsi: jnp.ndarray) -> jnp.ndarray:
    """Return log |ψ|^2 = 2 * Re(logψ)."""
    return 2.0 * jnp.real(logpsi)


def metropolis_step(
        key: PRNGKey,
        v: jnp.ndarray,
        field: jnp.ndarray,
        apply_fn: Callable,
        params: flax.core.FrozenDict,
) -> Tuple[jnp.ndarray, PRNGKey]:
    """Single-site flip Metropolis step for a batch.
    Chooses one random site per sample, proposes a flip, and accepts with
    probability min(1, exp(Δ log |ψ|^2)).
    v: (B,N) spins in {-1,+1}
    field: (B,)
    """
    B, N = v.shape
    key, k_site, k_u = jax.random.split(key, 3)
    sites = jax.random.randint(k_site, shape=(B,), minval=0, maxval=N)

    # Compute Δ logψ for all sites; then gather chosen Δ per sample
    dlogpsi_all = apply_fn({'params': params}, v, field, method=ComplexConditionalRBM.delta_logpsi_all_flips)  # (B,N)
    dlogpsi_i = jnp.take_along_axis(dlogpsi_all, sites[:, None], axis=1)[:, 0]  # (B,)

    # Δ log |ψ|^2 = 2 Re(Δ logψ)
    dlogp = 2.0 * jnp.real(dlogpsi_i)
    accept_logp = jnp.minimum(0.0, -dlogp)  # log of min(1, e^{dlogp}) == min(0, dlogp)
    u = jax.random.uniform(k_u, shape=(B,), dtype=jnp.float32)
    accept = (jnp.log(u + 1e-12) < dlogp).astype(v.dtype)  # bool→{0,1}

    # Apply flips where accepted
    flip_mask = (accept[:, None] * (jnp.arange(N)[None, :] == sites[:, None]).astype(v.dtype))
    v_new = v * (1.0 - 2.0 * flip_mask)  # flips selected site when accept==1
    # If rejected, keep v
    v_final = jnp.where(accept[:, None] == 1.0, v_new, v)
    return v_final, key


def run_metropolis(
        key: PRNGKey,
        v0: jnp.ndarray,
        field: jnp.ndarray,
        apply_fn: Callable,
        params: flax.core.FrozenDict,
        n_burn: int = 20,
        n_steps: int = 20,
        thin: int = 1,
) -> Tuple[jnp.ndarray, PRNGKey]:
    """Run a short Metropolis chain starting from v0; return final states.
    Uses JAX-friendly loops (lax.fori_loop) so it works under jit with dynamic integers.
    """
    v = v0

    def mh_body(_, carry):
        v, key = carry
        v, key = metropolis_step(key, v, field, apply_fn, params)
        return v, key

    # Burn-in
    v, key = jax.lax.fori_loop(0, n_burn, lambda i, c: mh_body(i, c), (v, key))

    # Production steps (optionally thin, though thin is unused here)
    v, key = jax.lax.fori_loop(0, n_steps, lambda i, c: mh_body(i, c), (v, key))

    return v, key


# ---------------------------------------------------
# TFIM local energy and observables
# ---------------------------------------------------

def local_energy_TFIM(
        v: jnp.ndarray,               # (B,N) spins {-1,+1}
        field: jnp.ndarray,           # (B,)
        bonds: jnp.ndarray,           # (M,2)
        J: float,
        apply_fn: Callable,
        params: flax.core.FrozenDict,
) -> jnp.ndarray:
    """Per-sample local energy E_loc(v) for TFIM:
        H = -J Σ_{<ij>} σ_i^z σ_j^z - m Σ_i σ_i^x
        E_loc(v) = -J Σ_{<ij>} s_i s_j - m Σ_i [ψ(v^i)/ψ(v)]
    Returns (B,) real values (we take Re at the end).
    """
    s = v  # already ±1
    # Diagonal term
    i_idx = bonds[:, 0]
    j_idx = bonds[:, 1]
    s_i = s[:, i_idx]  # (B,M)
    s_j = s[:, j_idx]  # (B,M)
    e_diag = -J * jnp.sum(s_i * s_j, axis=1)  # (B,)

    # Off-diagonal term via amplitude ratios
    dlogpsi_all = apply_fn({'params': params}, v, field, method=ComplexConditionalRBM.delta_logpsi_all_flips)  # (B,N)
    ratios = jnp.exp(dlogpsi_all)  # (B,N) complex
    # Sum over i, weighted by m (field)
    e_off = -jnp.sum(ratios, axis=1) * field.astype(ratios.dtype)  # (B,) complex

    e_total = e_diag.astype(jnp.complex64) + e_off
    return jnp.real(e_total)


def magnetizations(
        v: jnp.ndarray,
        field: jnp.ndarray,
        apply_fn: Callable,
        params: flax.core.FrozenDict,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (Mz, Mx) per sample.
    Mz = (1/N) Σ_i s_i
    Mx = (1/N) Σ_i Re[ ψ(v^i)/ψ(v) ]
    """
    B, N = v.shape
    Mz = jnp.mean(v, axis=1)  # (B,)
    dlogpsi_all = apply_fn({'params': params}, v, field, method=ComplexConditionalRBM.delta_logpsi_all_flips)  # (B,N)
    ratios = jnp.exp(dlogpsi_all)
    Mx = jnp.mean(jnp.real(ratios), axis=1)
    return Mz, Mx


# ---------------------------------------------------
# Training
# ---------------------------------------------------

@jax.jit
def train_step(
        state: TrainState,
        v_chain: jnp.ndarray,
        field: jnp.ndarray,
        bonds: jnp.ndarray,
        J: float,
        key: PRNGKey,
        n_burn: int,
        n_steps: int,
        thin: int,
) -> Tuple[TrainState, jnp.ndarray, Dict[str, Any], jnp.ndarray, PRNGKey]:
    """One VMC step: run a short Metropolis chain, estimate energy, take a grad step.
    Returns updated (state, loss, metrics, v_chain, key).
    """
    # Sample new configurations with current params (no grad through sampling)
    v_detached = jax.lax.stop_gradient(v_chain)
    v_new, key = run_metropolis(key, v_detached, field, state.apply_fn, state.params, n_burn, n_steps, thin)

    def loss_fn(params):
        e_loc = local_energy_TFIM(v_new, field, bonds, J, state.apply_fn, params)
        return jnp.mean(e_loc), e_loc

    (loss, e_loc_batch), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    # Observables for logging
    Mz, Mx = magnetizations(v_new, field, state.apply_fn, state.params)
    metrics = {
        "E_mean": jnp.mean(e_loc_batch),
        "E_std": jnp.std(e_loc_batch),
        "Mz_mean": jnp.mean(Mz),
        "Mx_mean": jnp.mean(Mx),
        "accept_note": 0.0,  # placeholder if you later record acceptance
    }
    return state, loss, metrics, v_new, key


def train(
        state: TrainState,
        loader: MixedDataLoader,
        num_epochs: int,
        key: PRNGKey,
        J: float,
        Lx: int,
        Ly: int,
        periodic: bool,
        n_burn: int,
        n_steps: int,
        thin: int,
) -> Tuple[TrainState, Dict[int, Any]]:
    metrics: Dict[int, Any] = {}
    bonds = make_lattice_bonds(Lx, Ly, periodic)

    v_chain = None

    for epoch in range(num_epochs):
        tot_loss = 0.0
        last_metrics = None

        for batch_bits, batch_field in loader:
            B, N = batch_bits.shape
            # Initialize chain if needed (random spins)
            if v_chain is None or v_chain.shape[0] != B:
                key, k_init = jax.random.split(key)
                v_chain = jax.random.choice(k_init, jnp.array([-1.0, 1.0], dtype=jnp.float32), shape=(B, N))

            # We DO NOT use measurement bits for VMC; only the field conditions
            state, loss, batch_metrics, v_chain, key = train_step(
                state, v_chain, batch_field, bonds, J, key,
                n_burn=n_burn, n_steps=n_steps, thin=thin,
            )
            tot_loss += float(loss)
            last_metrics = batch_metrics

        avg_loss = tot_loss / max(1, len(loader))
        metrics[epoch] = {
            "loss": avg_loss,
            **{k: float(v) for k, v in (last_metrics or {}).items()},
        }
        print(
            f"Epoch {epoch+1}/{num_epochs} │ "
            f"Loss(E): {avg_loss:+.6f} │ "
            f"E_mean: {metrics[epoch]['E_mean']:+.6f} │ "
            f"Mz: {metrics[epoch]['Mz_mean']:+.4f} │ "
            f"Mx: {metrics[epoch]['Mx_mean']:+.4f}"
        )

    return state, metrics


def get_sigmoid_curve(high, low, steps, falloff, center_step=None):
    if center_step is None:
        center_step = steps / 2
    curve_fn = lambda step: low + (high - low) / (1 + jnp.exp(falloff * (jnp.minimum(step, steps) - center_step)))
    return curve_fn


# ---------------------------------------------------
# Data & Run
# ---------------------------------------------------

data_dir = Path("./measurements")
print(f"Data resides in: {data_dir}")

file_names = [
    "tfim_h1.00_3x3_10000.txt",
    "tfim_h2.00_3x3_10000.txt",
    "tfim_h2.80_3x3_10000.txt",
    "tfim_h3.00_3x3_10000.txt",
    "tfim_h3.30_3x3_10000.txt",
    "tfim_h3.60_3x3_10000.txt",
    "tfim_h4.00_3x3_10000.txt",
    "tfim_h5.00_3x3_10000.txt",
    "tfim_h6.00_3x3_10000.txt",
    "tfim_h7.00_3x3_10000.txt",
]

bits_list = []
fields_list = []
for fn in file_names:
    bits, field = load_measurements(data_dir / fn)
    bits_list.append(bits)    # (N, num_qubits) in {0,1}
    fields_list.append(field) # (N,) the transverse field values per sample

bits   = jnp.concatenate(bits_list,  axis=0)
fields = jnp.concatenate(fields_list, axis=0)

# MixedDataLoader will just serve us batches of field values; we ignore bits for VMC.
batch_size        = 1024
num_visible       = 9
num_hidden        = 12               # a bit larger by default
conditioner_width = 64
num_epochs        = 50
init_lr           = 3e-3
final_lr          = 5e-4

# TFIM / lattice params
J_coupling        = -1.0            # your convention; set +1.0 for ferro
Lx, Ly            = 3, 3            # must satisfy Lx*Ly == num_visible
periodic_bc       = False           # set True if you want PBC

# Sampler params
n_burn            = 20
n_steps           = 20
thin              = 1

# Sanity checks
assert Lx * Ly == num_visible, "Lx*Ly must equal num_visible"

# Model
key = jax.random.PRNGKey(42)
key, k_params = jax.random.split(key)

model = ComplexConditionalRBM(
    num_visible=num_visible,
    num_hidden=num_hidden,
    conditioner_width=conditioner_width,
    dtype=jnp.complex64,
)

# Dummy init
bits_dummy   = jnp.zeros((batch_size, num_visible), dtype=jnp.float32)
fields_dummy = jnp.zeros((batch_size,), dtype=jnp.float32)
variables = model.init({"params": k_params}, bits_dummy, fields_dummy, method=ComplexConditionalRBM.log_psi)

# Optimizer
loader = MixedDataLoader(bits=bits, field=fields, batch_size=batch_size)
schedule_steps = num_epochs * max(1, len(loader))
lr_schedule_fn = get_sigmoid_curve(high=init_lr, low=final_lr, steps=schedule_steps, falloff=5e-4)
optim = optax.adam(lr_schedule_fn)

state = TrainState.create(apply_fn=model.apply, params=variables["params"], tx=optim)

# Train
state, metrics = train(
    state=state,
    loader=loader,
    num_epochs=num_epochs,
    key=key,
    J=J_coupling,
    Lx=Lx,
    Ly=Ly,
    periodic=periodic_bc,
    n_burn=n_burn,
    n_steps=n_steps,
    thin=thin,
)

# ---------------------------------------------------
# Example: evaluate magnetization curve post-hoc (optional)
# ---------------------------------------------------
# You can sweep a set of fields, re-run the sampler for each, and measure (Mz, Mx)
# to produce plots similar to your original figure. Keep batches mono-field for
# clean curves.


# ============================================================================
# OPTION A: Learn FROM SAMPLES (PCD/CD) — classical conditional RBM
# ----------------------------------------------------------------------------
# This is the path you originally had: fit the empirical measurement
# distribution p_data(v|m) using an RBM with per-sample conditioning.
# NOTE: If your measurements are only in Z basis, phases are unidentifiable;
# complex parameters won't help. So we keep this model REAL and fast.
# ============================================================================

class ConditionalRBM_Samples(nn.Module):
    num_visible: int
    num_hidden: int
    conditioner_width: int = 64
    k: int = 1                  # CD-k steps per update
    T: float = 1.0

    def setup(self):
        self.W = self.param("W", nn.initializers.normal(0.01), (self.num_visible, self.num_hidden))
        self.b = self.param("b", nn.initializers.zeros, (self.num_visible,))
        self.c = self.param("c", nn.initializers.zeros, (self.num_hidden,))

    @nn.compact
    def conditioner(self, field: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # field shape: (B,) or (B,1)
        x = field[:, None] if field.ndim == 1 else field
        x = nn.Dense(self.conditioner_width)(x)
        x = nn.tanh(x)
        x = nn.Dense(2 * (self.num_visible + self.num_hidden))(x)
        N, H = self.num_visible, self.num_hidden
        boundaries = [N, 2*N, 2*N + H]
        gamma_b, beta_b, gamma_c, beta_c = jnp.split(x, boundaries, axis=-1)
        b_mod = (1.0 + gamma_b) * self.b[None, :] + beta_b
        c_mod = (1.0 + gamma_c) * self.c[None, :] + beta_c
        return b_mod, c_mod

    @staticmethod
    def free_energy(v, W, b, c) -> jnp.ndarray:
        return -(v @ b) - jnp.sum(jax.nn.softplus(v @ W + c), axis=-1)

    @staticmethod
    def gibbs_step(state, W, b, c, T):
        v, key = state
        key, h_key, v_key = jax.random.split(key, 3)
        h = jax.random.bernoulli(h_key, jax.nn.sigmoid((v @ W + c) / T)).astype(jnp.float32)
        v = jax.random.bernoulli(v_key, jax.nn.sigmoid((h @ W.T + b) / T)).astype(jnp.float32)
        return v, key

    def __call__(self, batch: Tuple[jnp.ndarray, jnp.ndarray], aux_vars: Dict[str, Any]) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        measurement, field = batch  # (B,N), (B,)
        b_mod, c_mod = self.conditioner(field)                # (B,N), (B,H)

        # CD-k with per-batch persistent chain (provided via aux_vars)
        key = aux_vars["key"]
        if "v_chain" in aux_vars and aux_vars["v_chain"] is not None:
            gibbs_chain = aux_vars["v_chain"]
        else:
            key, key_init = jax.random.split(key)
            gibbs_chain = jax.random.bernoulli(key_init, 0.5, shape=measurement.shape).astype(jnp.float32)

        gibbs_step_fn = lambda i, s: self.gibbs_step(s, self.W, b_mod, c_mod, self.T)
        gibbs_chain, key = jax.lax.fori_loop(0, self.k, gibbs_step_fn, (gibbs_chain, key))
        gibbs_chain = jax.lax.stop_gradient(gibbs_chain)

        fe_data  = jax.vmap(self.free_energy, in_axes=(0, None, 0, 0))(measurement, self.W, b_mod, c_mod)
        fe_model = jax.vmap(self.free_energy, in_axes=(0, None, 0, 0))(gibbs_chain,  self.W, b_mod, c_mod)

        free_energy_delta = fe_data - fe_model
        free_energy_mean  = jnp.mean(free_energy_delta)
        free_energy_std   = jnp.std(free_energy_delta)

        l2_strength = aux_vars.get("l2_strength", 0.0)
        l2_regularization = (
                jnp.sum((self.b[None, :] - b_mod) ** 2) +
                jnp.sum((self.c[None, :] - c_mod) ** 2)
        )

        loss = free_energy_mean + l2_strength * l2_regularization

        aux_vars_out = {
            "key": key,
            "free_energy_mean": free_energy_mean,
            "free_energy_std": free_energy_std,
            "v_chain": gibbs_chain,  # persist PCD chain
        }

        return loss, aux_vars_out

    @nn.nowrap
    def generate(self, field: jnp.ndarray, k: int, key: PRNGKey, init_v: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, PRNGKey]:
        b_mod, c_mod = self.conditioner(field)
        B = field.shape[0]
        key, key_init = jax.random.split(key)
        if init_v is None:
            v = jax.random.bernoulli(key_init, 0.5, shape=(B, self.num_visible)).astype(jnp.float32)
        else:
            v = init_v
        step = lambda i, s: self.gibbs_step(s, self.W, b_mod, c_mod, self.T)
        v, key = jax.lax.fori_loop(0, k, step, (v, key))
        return v, key


@jax.jit
def train_step_samples(state: TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray], aux_vars: Dict[str, Any]):
    loss_fn = lambda params: state.apply_fn({'params': params}, batch, aux_vars)
    (loss, aux_out), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, aux_out


def train_from_samples(
        state: TrainState,
        loader: MixedDataLoader,
        num_epochs: int,
        key: PRNGKey,
        l2_strength: float,
        lr_schedule_fn: Callable[[int], float],
) -> Tuple[TrainState, Dict[int, Any]]:
    metrics: Dict[int, Any] = {}
    v_chain = None
    for epoch in range(num_epochs):
        tot_loss = 0.0
        for batch_bits, batch_field in loader:
            key, subkey = jax.random.split(key)
            aux_vars = {"key": subkey, "l2_strength": l2_strength, "v_chain": v_chain}
            batch = (batch_bits, batch_field)
            state, loss, aux_out = train_step_samples(state, batch, aux_vars)
            v_chain = aux_out["v_chain"]
            free_energy_mean = aux_out["free_energy_mean"]
            free_energy_std  = aux_out["free_energy_std"]
            tot_loss += float(loss)
        avg_loss = tot_loss / max(1, len(loader))
        lr = lr_schedule_fn(state.step)
        metrics[epoch] = dict(
            loss=avg_loss,
            free_energy_mean=float(free_energy_mean),
            free_energy_std=float(free_energy_std),
            lr=float(lr),
        )
        print(
            f"[SAMPLES] Epoch {epoch+1}/{num_epochs} │ "
            f"Loss: {avg_loss:+.4f} │ FE mean: {float(free_energy_mean):+.4f} │ FE std: {float(free_energy_std):+.4f} │ LR: {float(lr):.5f}"
        )
    return state, metrics


# Quick driver showing how to switch between modes
if __name__ == "__main__":
    TRAIN_MODE = "samples"  # "samples" or "vmc"

    if TRAIN_MODE == "samples":
        batch_size        = 1024
        num_visible       = 9
        num_hidden        = 9
        conditioner_width = 32
        num_epochs        = 100
        k_steps           = 100
        init_lr           = 1e-2
        final_lr          = init_lr * 0.1
        l2_strength       = 0.0

        key = jax.random.PRNGKey(0)
        key, key_params, key_dummy = jax.random.split(key, 3)

        model_s = ConditionalRBM_Samples(
            num_visible=num_visible,
            num_hidden=num_hidden,
            conditioner_width=conditioner_width,
            k=k_steps,
        )

        # init with dummy batch
        bits_dummy   = jnp.zeros((batch_size, num_visible), dtype=jnp.float32)
        fields_dummy = jnp.zeros((batch_size,), dtype=jnp.float32)
        variables = model_s.init({"params": key_params}, (bits_dummy, fields_dummy), {"key": key_dummy, "l2_strength": l2_strength})

        loader = MixedDataLoader(bits=bits, field=fields, batch_size=batch_size)
        schedule_steps = num_epochs * max(1, len(loader))
        lr_schedule_fn = get_sigmoid_curve(high=init_lr, low=final_lr, steps=schedule_steps, falloff=5e-4)
        optim = optax.adam(lr_schedule_fn)

        state = TrainState.create(apply_fn=model_s.apply, params=variables["params"], tx=optim)
        state, metrics = train_from_samples(state, loader, num_epochs, key, l2_strength, lr_schedule_fn)

    # else: the VMC path above remains available
