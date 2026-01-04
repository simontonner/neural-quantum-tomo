from math import ceil, sqrt, prod
from typing import Iterable, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================
# Device & dtypes
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We run all RBM energies in float64.
# Reason: RBM "free energy" values go straight into exp() to make amplitudes
# exp(-F/2). If F is large-magnitude and you do this in float32,
# you overflow/underflow instantly. float64 buys us stability.
DTYPE = torch.double


# ============================================================
# Single-qubit unitaries (as cdouble)
# ============================================================
def create_dict(**overrides):
    """
    Build the dictionary {X,Y,Z} of the single-qubit basis-change unitaries.

    Convention:
    - 'X' corresponds to projective measurement in the X basis.
      We use 1/sqrt(2) * [[1, 1],[1,-1]], i.e. Hadamard.
    - 'Y' corresponds to Y-basis measurement.
      We choose 1/sqrt(2) * [[1,-i],[1,+i]].
      This fixes the relative phase convention for Y.
      IMPORTANT: this matters, because tomography depends on these global
      per-qubit phase choices when reconstructing off-diagonal coherences.
    - 'Z' is just the computational basis, i.e. identity.

    All returned as torch.cdouble so we can feed them directly into the
    complex overlap calculations.
    """
    inv_sqrt2 = 1.0 / sqrt(2.0)

    X = inv_sqrt2 * torch.tensor([[1.0+0.0j,  1.0+0.0j],
                                  [1.0+0.0j, -1.0+0.0j]],
                                 dtype=torch.cdouble, device=DEVICE)

    Y = inv_sqrt2 * torch.tensor([[1.0+0.0j,  0.0-1.0j],
                                  [1.0+0.0j,  0.0+1.0j]],
                                 dtype=torch.cdouble, device=DEVICE)

    Z = torch.tensor([[1.0+0.0j, 0.0+0.0j],
                      [0.0+0.0j, 1.0+0.0j]],
                     dtype=torch.cdouble, device=DEVICE)

    U = {"X": X.contiguous(), "Y": Y.contiguous(), "Z": Z.contiguous()}

    # Allow overriding these with custom 2x2 unitaries, e.g. if you measured
    # in some other local basis and want to plug that in.
    for name, mat in overrides.items():
        U[name] = as_complex_unitary(mat, DEVICE)

    return U


def as_complex_unitary(U, device: torch.device):
    """Return a (2,2) complex (cdouble) matrix on `device`."""
    if torch.is_tensor(U):
        if U.dim() != 2 or U.shape != (2, 2):
            raise ValueError(f"as_complex_unitary expects (2,2), got {tuple(U.shape)}")
        return U.to(device=device, dtype=torch.cdouble).contiguous()

    U_t = torch.tensor(U, device=device)
    if U_t.dim() != 2 or U_t.shape != (2, 2):
        raise ValueError(f"as_complex_unitary expects (2,2), got {tuple(U_t.shape)}")
    return U_t.to(dtype=torch.cdouble, device=device).contiguous()


def inverse(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Safe complex inverse:
        z^{-1} = z* / max(|z|^2, eps)

    Used in some algebra where dividing by something that might go to ~0
    would otherwise explode gradients.
    """
    zz = z.to(torch.cdouble)
    return zz.conj() / (zz.abs().pow(2).clamp_min(eps))


# ============================================================
# Kronecker-apply without explicitly forming ⊗
# ============================================================
def _kron_mult(matrices: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """
    Apply the tensor product (⊗_s U_s) to a state vector psi without
    ever explicitly building the 2^n x 2^n Kronecker matrix.

    This is done by repeatedly reshaping and einsum'ing single-site 2x2
    operators into the right axes.

    Why:
    - Explicit Kronecker blows up as 2^n x 2^n.
    - Here we only ever need (⊗U_s) |psi> or its action on batches of psi.
    """
    assert all(torch.is_complex(m) for m in matrices), "unitaries must be complex"
    if not torch.is_complex(x):
        raise TypeError("x must be complex (cdouble)")

    x_cd = x.to(torch.cdouble)

    # Assume x has shape (2^n, batch_dim...)
    L = x_cd.shape[0]
    batch = int(x_cd.numel() // L)
    y = x_cd.reshape(L, batch)

    # Check dimension compatibility
    n = [m.size(-1) for m in matrices]
    if prod(n) != L:
        raise ValueError(f"Incompatible sizes: expected leading dim {prod(n)}, got {L}")

    # Sequentially apply each single-qubit U by reshaping and einsum
    left = L
    for U in reversed(matrices):
        ns = U.shape[-1]     # 2
        left //= ns
        y = y.reshape(left, ns, -1)
        # y_out[i, l, m] = sum_j  U[i,j] * y[j,l,m]
        y = torch.einsum('ij,ljm->lim', U, y).reshape(left * ns, -1)

    return y.reshape(*x_cd.shape)


def rotate_psi(nn_state, basis: Iterable[str], space: torch.Tensor,
               unitaries: Optional[dict] = None, psi: Optional[torch.Tensor] = None):
    """
    Rotate psi into an arbitrary local product basis.

    Arguments:
    - basis: sequence like ('X','Y','Z',...) of length num_visible.
    - space: computational basis states, shape (2^n, n), in {0,1}.
    - psi:   optional precomputed state amplitudes in computational basis;
             if None, we'll call nn_state.psi_complex(space).

    This is only used for *evaluation/metrics* (e.g. computing KL in rotated
    bases), not for training. So we can afford the "full vector" picture here.
    """
    n_vis = nn_state.num_visible
    basis = list(basis)
    if len(basis) != n_vis:
        raise ValueError(f"rotate_psi: basis length {len(basis)} != num_visible {n_vis}")

    # Build the per-site unitaries on the fly
    if unitaries is None:
        us = [nn_state.U[b].to(device=nn_state.device, dtype=torch.cdouble) for b in basis]
    else:
        Udict = {k: as_complex_unitary(v, nn_state.device) for k, v in unitaries.items()}
        us = [Udict[b] for b in basis]

    # psi in computational basis (keep grad if this is called in a differentiable
    # context, which it's typically not in metrics)
    if psi is None:
        x = nn_state.psi_complex(space)
    else:
        if not torch.is_complex(psi):
            raise TypeError("rotate_psi: psi must be complex (cdouble).")
        x = psi.to(device=nn_state.device, dtype=torch.cdouble)

    return _kron_mult(us, x)


# ============================================================
# Basis-branch enumeration for local product measurements
# ============================================================
def _rotate_basis_state(nn_state, basis, states, unitaries=None):
    """
    The tomography trick:

    Given:
    - a measured bitstring `states` in some basis b = (b0,b1,...,bn-1),
    - where each b_j ∈ {X,Y,Z},

    we DO NOT have to sum over all 2^n computational states to get
    <σ^[b]|U_b|ψ>. Only qubits measured in X/Y create superpositions.
    If S sites are non-Z, then only 2^S branches contribute.

    This function:
    - finds those S rotated sites,
    - enumerates all 2^S possible assignments for them,
    - builds:
        v  : (C,B,n) candidate computational basis states per data row
        Ut : (C,B)   the product of the appropriate single-site matrix elements
                     for each branch

    We'll later multiply Ut * ψ(v) and sum over C for each row B.
    """
    device = nn_state.device
    n_vis = nn_state.num_visible
    basis_seq = list(basis)

    if len(basis_seq) != n_vis:
        raise ValueError(f"_rotate_basis_state: basis length {len(basis_seq)} != num_visible {n_vis}")
    if states.shape[-1] != n_vis:
        raise ValueError(f"_rotate_basis_state: states width {states.shape[-1]} != num_visible {n_vis}")

    # Indices of sites that are actually rotated (X or Y as opposed to Z)
    sites = [i for i, b in enumerate(basis_seq) if b != "Z"]
    if len(sites) == 0:
        # Pure Z measurement: trivial branch set of size 1
        v = states.unsqueeze(0)  # (1, B, n)
        Ut = torch.ones(v.shape[:-1], dtype=torch.cdouble, device=device)  # (1, B)
        return Ut, v

    # Collect single-qubit unitaries for JUST the rotated sites
    src = nn_state.U if unitaries is None else unitaries
    Ulist = [
        as_complex_unitary(src[basis_seq[i]], device).reshape(2, 2).contiguous()
        for i in sites
    ]
    Uc = torch.stack(Ulist, dim=0)  # (S, 2, 2)

    S = len(sites)           # number of rotated sites
    B = states.shape[0]      # minibatch size
    C = 2 ** S               # number of coherent branches for each data row

    # Enumerate all 2^S spin assignments for those S rotated sites
    combos = nn_state.generate_hilbert_space(size=S, device=device)  # (C, S)

    # Build candidate computational states per branch:
    # For each data row, clone it, then overwrite the rotated sites with the
    # branch assignment.
    v = states.unsqueeze(0).repeat(C, 1, 1)           # (C, B, n)
    v[:, :, sites] = combos.unsqueeze(1)              # broadcast combos to each data row
    v = v.contiguous()

    # Build the per-branch weight Ut = ∏_sites <σ^[b]_site | U_site | σ'_site>
    # (this is just picking the relevant 2x2 element per site and multiplying)
    inp_sb   = states[:, sites].round().long().T               # (S, B)
    outp_csb = v[:, :, sites].round().long().permute(0, 2, 1)  # (C, S, B)
    inp_csb  = inp_sb.unsqueeze(0).expand(C, -1, -1)           # (C, S, B)

    s_idx = torch.arange(S, device=device).view(1, S, 1).expand(C, S, B)  # (C, S, B)
    sel = Uc[s_idx, inp_csb, outp_csb]                                    # (C, S, B) complex
    Ut = sel.prod(dim=1)  # (C, B), complex coefficient per branch

    return Ut.to(torch.cdouble), v


def _convert_basis_element_to_index(states):
    """
    Convert rows of bits in {0,1} to flat indices in [0, 2^n - 1].
    MSB-first ordering, i.e.  [σ0 σ1 ... σ{n-1}] -> integer
    """
    s = states.round().to(torch.long)
    n = s.shape[-1]
    shifts = torch.arange(n - 1, -1, -1, device=s.device, dtype=torch.long)
    return (s << shifts).sum(dim=-1)


def rotate_psi_inner_prod(nn_state, basis, states, unitaries=None, psi=None, include_extras=False):
    """
    Compute <σ^[b] | U_b | ψ> for a batch of measured outcomes in basis `basis`.

    This is the *naive* (direct) complex sum over branches:
        sum_branch Ut * ψ(branch_state)

    It's fine for debugging / inspection.
    BUT:
    - It does the raw complex sum and can overflow/underflow when
      ψ(branch) spans huge dynamic range.
    - The TRAINING LOSS actually uses a stabilized log-sum-exp version now
      (see ComplexWaveFunction._stable_log_overlap_amp2).

    Returns:
      total        : (B,) complex
      (optionally) branches : (C,B) complex, per-branch contribution
      (optionally) v        : (C,B,n) float64 branch configurations
    """
    Ut, v = _rotate_basis_state(nn_state, basis, states, unitaries=unitaries)

    if psi is None:
        # Track gradients through psi (amplitude+phase RBMs).
        psi_sel = nn_state.psi_complex(v)  # (C,B)
    else:
        if not torch.is_complex(psi):
            raise TypeError("rotate_psi_inner_prod: psi must be complex.")
        idx = _convert_basis_element_to_index(v).long()  # (C,B)
        psi_c = psi.to(dtype=torch.cdouble, device=nn_state.device)
        psi_sel = psi_c[idx]

    Upsi_v_c = Ut * psi_sel        # (C,B)
    Upsi_c   = Upsi_v_c.sum(dim=0) # (B,)

    if include_extras:
        return Upsi_c, Upsi_v_c, v
    return Upsi_c


# ============================================================
# RBM definition
# ============================================================
class BinaryRBM(nn.Module):
    """
    Plain Bernoulli/Bernoulli RBM (visible∈{0,1}^n, hidden∈{0,1}^m).

    We'll use this RBM twice:
      - one copy (rbm_am) to learn amplitudes,
      - one copy (rbm_ph) to learn phases.

    IMPORTANT NOTATION POINT:
    In the theory we call the RBM "free energy"

        𝓕(v) = - Σ_j a_j v_j
                - Σ_i log(1 + exp(b_i + (W v)_i))

    In code we call this `effective_energy(v)` but it's actually 𝓕(v).
    So:
        psi(v) ∝ exp(-𝓕_am(v)/2) * exp[-i 𝓕_ph(v)/2].

    That matches the amplitude-phase factorisation in the writeup.
    """

    def __init__(self, num_visible, num_hidden=None, zero_weights=False, device: torch.device = DEVICE):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden) if num_hidden else self.num_visible
        self.device = device
        self.initialize_parameters(zero_weights=zero_weights)

    def __repr__(self):
        return f"BinaryRBM(num_visible={self.num_visible}, num_hidden={self.num_hidden}, device='{self.device}')"

    def initialize_parameters(self, zero_weights=False):
        """
        Either random-init or exact-zero init (for debugging / ablations).
        We scale the random init like ~N(0, 1/sqrt(num_visible)) just to avoid
        ridiculous logits at step 0.
        """
        gen_tensor = torch.zeros if zero_weights else torch.randn
        scale = 1.0 / np.sqrt(self.num_visible)

        self.weights = nn.Parameter(
            gen_tensor(self.num_hidden, self.num_visible, device=self.device, dtype=DTYPE) * scale,
            requires_grad=True,
            )
        self.visible_bias = nn.Parameter(
            torch.zeros(self.num_visible, device=self.device, dtype=DTYPE),
            requires_grad=True
        )
        self.hidden_bias = nn.Parameter(
            torch.zeros(self.num_hidden, device=self.device, dtype=DTYPE),
            requires_grad=True
        )

    def effective_energy(self, v):
        """
        Compute 𝓕(v), the RBM "free energy" for visible state(s) v.

        In formulas:
            𝓕(v)
            = -v·a - Σ_j softplus(b_j + (Wv)_j)

        Shape rules:
        - v can be (batch, n) or (C, B, n) etc; we broadcast over trailing dim.
        - Return has same leading shape as v without the last dim.

        This is exactly the 𝓕_λ(σ) / 𝓕_μ(σ) in the theory.
        """
        unsq = False
        if v.dim() < 2:
            v = v.unsqueeze(0)
            unsq = True

        v = v.to(self.weights)

        visible_bias_term = torch.matmul(v, self.visible_bias)  # (...,)
        hid_bias_term = F.softplus(F.linear(v, self.weights, self.hidden_bias)).sum(-1)  # (...,)

        out = -(visible_bias_term + hid_bias_term)  # (...,)  This is 𝓕(v).
        return out.squeeze(0) if unsq else out

    @torch.no_grad()
    def gibbs_steps(self, k, initial_state, overwrite=False):
        """
        k-step block Gibbs sampling starting at `initial_state`.

        This is Contrastive Divergence (CD-k):
          v0 -> h0 ~ p(h|v0)
             -> v1 ~ p(v|h0)
             ...
             -> vk

        We ONLY use these vk samples to provide the negative phase for the
        amplitude RBM (rbm_am). We explicitly DO NOT propagate gradients
        through this Markov chain (no .backward through sampling),
        because that's not how classical CD works.

        Numerical safety:
        - We clamp sigmoid outputs back into [0,1] and replace NaN/Inf so
          torch.bernoulli() never crashes, even if parameters temporarily
          blow up. That keeps training alive instead of NaN-ing out.
        """
        v = (initial_state if overwrite else initial_state.clone()).to(self.weights)
        h = torch.empty(*v.shape[:-1], self.num_hidden, device=self.device, dtype=DTYPE)

        for _ in range(k):
            # Sample hidden given visible
            h_lin  = F.linear(v, self.weights, self.hidden_bias)
            h_prob = torch.sigmoid(h_lin)
            h_prob = torch.nan_to_num(h_prob, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            torch.bernoulli(h_prob, out=h)

            # Sample visible given hidden
            v_lin  = F.linear(h, self.weights.t(), self.visible_bias)
            v_prob = torch.sigmoid(v_lin)
            v_prob = torch.nan_to_num(v_prob, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            torch.bernoulli(v_prob, out=v)

        return v


# ============================================================
# ComplexWaveFunction: amplitude+phase RBMs glued into ψ
# ============================================================
class ComplexWaveFunction:
    """
    ψ(σ) = exp(-𝓕_λ(σ)/2) * exp( -i 𝓕_μ(σ)/2 )

    where 𝓕_λ, 𝓕_μ are two *separate* RBM free energies (amplitude RBM and
    phase RBM). This matches the amplitude-phase factorisation in the theory.

    Training objective is an autodiff-friendly contrastive divergence version
    of RBM tomography:

    - Positive/data term:
        * For Z-basis rows (direct computational basis measurements):
            add 𝓕_λ(σ)
          which is just RBM negative log-likelihood of the visible state.
        * For rotated-basis rows (X/Y/...):
            add -log( |<σ^[b] | U_b | ψ>|^2 + ε )
          which forces the phase RBM to match the observed interference
          patterns under local basis rotations.

      We compute that second term in a numerically STABLE way via a complex
      log-sum-exp trick (_stable_log_overlap_amp2).

    - Negative/model term:
        * Classical CD-k negative phase on the amplitude RBM only:
            draw v_k ~ Gibbs^k
            add 𝓕_λ(v_k)

        This approximates the ∂/∂λ log Z_λ term.

    Final minibatch loss:
        L = (L_pos / B_pos) - (L_neg / B_neg)

    And we backprop that through BOTH RBMs at once every step
    (i.e. interleaved training instead of staged "amplitude-then-phase").
    """

    def __init__(self, num_visible, num_hidden=None, unitary_dict=None, device: torch.device = DEVICE):
        self.device = device

        # rbm_am learns amplitudes (|ψ|);
        # rbm_ph learns phases (arg ψ).
        self.rbm_am = BinaryRBM(num_visible, num_hidden, device=self.device)
        self.rbm_ph = BinaryRBM(num_visible, num_hidden, device=self.device)

        self.num_visible = self.rbm_am.num_visible
        self.num_hidden = self.rbm_am.num_hidden

        # Local unitaries {X,Y,Z} in our chosen measurement convention.
        raw = unitary_dict if unitary_dict is not None else create_dict()
        self.U = {k: as_complex_unitary(v, self.device) for k, v in raw.items()}

        # Convenience knobs
        self._stop_training = False
        self._max_size = 20  # only brute-force Hilbert space up to n=20 for metrics

    # ------------------ control / hygiene ------------------
    @property
    def stop_training(self):
        return self._stop_training

    @stop_training.setter
    def stop_training(self, new_val):
        if isinstance(new_val, bool):
            self._stop_training = new_val
        else:
            raise ValueError("stop_training must be bool!")

    @property
    def max_size(self):
        return self._max_size

    def reinitialize_parameters(self):
        """
        Re-init both RBMs. Sometimes you want to restart training quickly,
        e.g. different seed or different learning rate schedule.
        """
        self.rbm_am.initialize_parameters()
        self.rbm_ph.initialize_parameters()

    # ------------------ core ψ accessors ------------------
    def amplitude(self, v):
        """
        |ψ(v)| = exp(-𝓕_λ(v)/2), purely real/positive.

        This is literally the Born amplitude magnitude, i.e. sqrt of the
        model's unnormalized probability mass for bitstring v.
        """
        v = v.to(self.device, dtype=DTYPE)
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()

    def phase(self, v):
        """
        Phase angle θ(v) = -𝓕_μ(v)/2.

        The phase RBM contributes ONLY a phase factor, it never touches
        the magnitude. That's why |ψ|^2 depends only on rbm_am.
        """
        v = v.to(self.device, dtype=DTYPE)
        return -0.5 * self.rbm_ph.effective_energy(v)

    def psi_complex(self, v):
        """
        ψ(v) as a complex-valued tensor.

        ψ(v) = exp(-𝓕_λ(v)/2) * exp(i * (-𝓕_μ(v)/2))
             = amplitude(v) * exp(i * phase(v))
        """
        v = v.to(self.device, dtype=DTYPE)
        amp = (-self.rbm_am.effective_energy(v)).exp().sqrt()   # real ≥ 0
        ph  = -0.5 * self.rbm_ph.effective_energy(v)            # real phase angle
        return amp.to(torch.cdouble) * torch.exp(1j * ph.to(torch.cdouble))

    def psi_complex_normalized(self, v):
        """
        Normalized ψ(v) using the EXACT log Z_λ from rbm_am.

        WARNING:
        - This requires summing over the whole Hilbert space (2^n).
        - Only safe for tiny n, and we only use it for metrics/plots.

        We do:
          ψ_norm(v) = exp(-(𝓕_λ(v))/2 - (1/2)log Z_λ) * exp(-i 𝓕_μ(v)/2)
        """
        v = v.to(self.device, dtype=DTYPE)
        E  = self.rbm_am.effective_energy(v)   # 𝓕_λ(v)
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        logZ = torch.logsumexp(-E, dim=0)      # log Z_λ (exact for small n)

        return torch.exp(
            ((-0.5 * E) - 0.5 * logZ).to(torch.cdouble)
            + 1j * ph.to(torch.cdouble)
        )

    # Convenience aliases
    def psi(self, v): return self.psi_complex(v)
    def psi_normalized(self, v): return self.psi_complex_normalized(v)
    def phase_angle(self, v): return self.phase(v)

    # ------------------ utilities ------------------
    def generate_hilbert_space(self, size=None, device=None):
        """
        Enumerate computational basis as a (2^size, size) bit-matrix in {0,1}.

        We only allow brute-force up to self._max_size sites, because for
        tomography diagnostics at the end we sometimes iterate all 2^n.
        """
        device = self.device if device is None else device
        size = self.num_visible if size is None else int(size)

        if size > self._max_size:
            raise ValueError(f"Hilbert space too large (n={size} > max={self._max_size}).")

        n = 1 << size
        ar = torch.arange(n, device=device, dtype=torch.long)
        shifts = torch.arange(size - 1, -1, -1, device=device, dtype=torch.long)
        return ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)

    # ------------------ stable overlap for rotated bases ------------------
    def _stable_log_overlap_amp2(
            self,
            basis: Tuple[str, ...],
            states: torch.Tensor,
            eps_rot: float = 1e-6,
            unitaries=None,
    ):
        """
        Numerically stable version of:

            A = <σ^[b] | U_b | ψ>
        and
            |A|^2

        Instead of directly summing Ut * ψ(branch) (which can under/overflow
        badly because exp(-𝓕_λ/2) can span MANY orders of magnitude),
        we do a complex log-sum-exp trick:

        For each measured outcome state (row in `states`):
          - enumerate only the 2^S branches that actually contribute
            (S = #non-Z qubits in `basis`),
            using _rotate_basis_state(...).
          - Each branch contributes:
                Ut * ψ(v_branch)
            where Ut is the product of single-site rotation matrix elements.

          - Write ψ(v_branch) = exp(-𝓕_λ/2) * exp(i * (-𝓕_μ/2)).
            Let
                logmag_total = (-𝓕_λ/2) + log|Ut|
                phase_total  = (-𝓕_μ/2) + arg(Ut)

          - Let M = max_c logmag_total[c] over branches c.
            Then
                S' = Σ_c exp(logmag_total[c] - M) * exp(i * phase_total[c])
            is safe because all magnitudes are ≤ 1.

          - Then
                log |A|^2
              = 2M + log(|S'|^2 + eps_rot)

        We return that log |A|^2 for each row in the mini-batch.
        Training will then add
            -log|A|^2
        to the loss (i.e. "please assign high probability to what we saw").

        Returns:
            log_amp2 : (B,) float64
                       where B = states.shape[0].
        """
        # Enumerate coherent branches relevant to these basis measurements
        Ut, v = _rotate_basis_state(self, basis, states, unitaries=unitaries)
        # shapes:
        #   Ut : (C,B) complex
        #   v  : (C,B,n) float64 in {0,1}

        # Compute RBM free energies for amplitude and phase nets on each branch.
        # These are exactly 𝓕_λ and 𝓕_μ in the theory.
        F_am = self.rbm_am.effective_energy(v)  # (C,B) float64
        F_ph = self.rbm_ph.effective_energy(v)  # (C,B) float64

        # log | Ut * ψ(v) | = log|Ut| + (-0.5 * F_am)
        logmag_total = (-0.5 * F_am) + torch.log(Ut.abs().to(DTYPE).clamp_min(1e-300))

        # total phase angle per branch:
        # arg( Ut * ψ(v) ) = arg(Ut) + (-0.5 * F_ph)
        phase_total = (-0.5 * F_ph).to(torch.cdouble) + torch.angle(Ut).to(torch.cdouble)

        # Complex log-sum-exp:
        # pick the maximum log magnitude per data row to factor out
        M, _ = torch.max(logmag_total, dim=0, keepdim=True)  # (1,B)

        # scaled magnitudes are now <= 1 so no overflow in exp
        scaled_mag = torch.exp((logmag_total - M).to(DTYPE))  # (C,B), real

        # rebuild each branch's scaled complex contribution
        contrib = scaled_mag.to(torch.cdouble) * torch.exp(1j * phase_total)  # (C,B) complex

        # sum branches coherently
        S_prime = contrib.sum(dim=0)  # (B,) complex

        # |S'|^2
        S_abs2 = (S_prime.conj() * S_prime).real.to(DTYPE)  # (B,), float64

        # log |A|^2 = 2M + log(|S'|^2 + eps_rot)
        log_amp2 = (2.0 * M.squeeze(0)).to(DTYPE) + torch.log(S_abs2 + eps_rot)

        return log_amp2  # (B,)

    # ------------------ loss pieces ------------------
    def _positive_phase_loss(
            self,
            samples: torch.Tensor,
            bases_batch: List[Tuple[str, ...]],
            eps_rot: float = 1e-6,
    ):
        """
        Stable positive/data term L_pos.

        Theory recap:
        - For rows measured in pure Z (computational basis):
              add 𝓕_λ(v_row)
          That's the usual RBM positive phase: push down "energy" of data.

        - For rows with any X/Y in their basis:
            we need the likelihood the model assigns to that actually
            observed rotated outcome.
            That is |<σ^[b]|U_b|ψ>|^2.
            We add -log(|...|^2 + ε), i.e. push that probability UP.

        Numerics:
        - Directly summing <σ^[b]|U_b|ψ> is unstable (can under/overflow).
          So we use _stable_log_overlap_amp2 to get log|...|^2 via
          a complex log-sum-exp trick that factors out the largest branch.
        """
        # Bucket rows that share the same basis string so we can process them in chunks.
        buckets = {}
        for i, row in enumerate(bases_batch):
            buckets.setdefault(tuple(row), []).append(i)

        loss_rot = samples.new_tensor(0.0, dtype=DTYPE)
        loss_z   = samples.new_tensor(0.0, dtype=DTYPE)

        for basis_t, idxs in buckets.items():
            idxs_t = torch.tensor(idxs, device=samples.device)

            if any(ch != "Z" for ch in basis_t):
                # Rotated case: phase learning signal.
                # We compute a *stable* log |<σ^[b]|U_b|ψ>|^2 for each row.
                log_amp2 = self._stable_log_overlap_amp2(
                    basis_t,
                    samples[idxs_t],
                    eps_rot=eps_rot,
                )  # (B_sub,)

                # We want -log(|A|^2) summed over the minibatch.
                term = -log_amp2.sum().to(DTYPE)
                loss_rot = loss_rot + term
            else:
                # Pure Z basis: amplitude RBM NLL.
                # Add 𝓕_λ(data), i.e. the positive phase in classical RBM CD.
                Epos = self.rbm_am.effective_energy(samples[idxs_t])  # (B_sub,)
                loss_z = loss_z + Epos.sum()

        return loss_rot + loss_z

    def _negative_phase_loss(self, k: int, neg_init: torch.Tensor):
        """
        CD-k negative phase for the amplitude RBM.

        We:
          - run k-step block Gibbs starting from some Z-basis samples,
          - evaluate 𝓕_λ(v_k),
          - sum over that batch.

        This acts like the "model expectation" part of standard RBM CD,
        i.e. an approximation to ∂ log Z_λ / ∂λ.

        Note: gradients *do* flow into rbm_am parameters through 𝓕_λ(v_k),
        but the chain itself is @torch.no_grad() so we don't try to
        differentiate through the sampler.
        """
        with torch.no_grad():
            vk = self.rbm_am.gibbs_steps(k, neg_init, overwrite=True)

        Eneg = self.rbm_am.effective_energy(vk)  # (B_neg,), float64
        return Eneg.sum(), vk.shape[0]

    # ------------------ training loop ------------------
    def fit(
            self, loader,
            epochs=70,
            k=10,
            lr=1e-1,
            log_every=5,
            optimizer=torch.optim.SGD,
            optimizer_args=None,
            target=None,
            bases=None,
            space=None,
            print_metrics=True,
            metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}",
    ):
        """
        Autodiff Contrastive Divergence training for quantum state tomography.

        At each minibatch:
          1. L_pos  = stabilized positive/data term
          2. L_neg  = CD-k negative phase term
          3. loss   = (L_pos / B_pos) - (L_neg / B_neg)

        Then we do:
            loss.backward()
            clip_grad_norm_(params, 10.0)
            opt.step()

        Key points:
        - We DO NOT do staged amplitude-then-phase training.
          We update both rbm_am (amplitude) and rbm_ph (phase) simultaneously.
          This "interleaved" trick is reverse-engineered from QuCumber-style
          training loops.
        - The rotated-basis likelihood term transmits gradients to BOTH RBMs.
        - The negative phase term only touches rbm_am.
        """
        if self.stop_training:
            return {"epoch": []}

        optimizer_args = {} if optimizer_args is None else optimizer_args
        params = list(self.rbm_am.parameters()) + list(self.rbm_ph.parameters())
        opt = optimizer(params, lr=lr, **optimizer_args)

        history = {"epoch": []}
        if target is not None:
            history["Fidelity"], history["KL"] = [], []

        # We'll often pass in the explicit Hilbert space `space` for metrics.
        if space is None:
            space = self.generate_hilbert_space()

        for ep in range(1, epochs + 1):
            for pos_batch, neg_batch, bases_batch in loader.iter_epoch():
                # -----------------
                # positive/data phase
                # -----------------
                L_pos = self._positive_phase_loss(pos_batch, bases_batch)  # scalar
                B_pos = float(pos_batch.shape[0])

                # -----------------
                # negative/model phase
                # -----------------
                L_neg, B_neg = self._negative_phase_loss(k, neg_batch)

                # -----------------
                # final contrastive objective
                # -----------------
                loss = (L_pos / B_pos) - (L_neg / B_neg)

                opt.zero_grad()
                loss.backward()

                # Safety against insane gradients:
                #  - rotated-basis terms can spike when model puts ~0 mass
                #    on an actually observed outcome.
                #  - We already mitigate with eps_rot and complex log-sum-exp,
                #    this is just extra padding.
                torch.nn.utils.clip_grad_norm_(params, 10.0)

                opt.step()

                if self.stop_training:
                    break

            # Periodic metrics (fidelity, KL) vs known target wavefunction.
            if (target is not None) and (ep % log_every == 0):
                with torch.no_grad():
                    fid_val = fidelity(self, target, space=space, bases=bases)
                    kl_val = KL(self, target, space=space, bases=bases)
                history["epoch"].append(ep)
                history["Fidelity"].append(fid_val)
                history["KL"].append(kl_val)
                if print_metrics:
                    print(metric_fmt.format(ep=ep, fid=fid_val, kl=kl_val))

            if self.stop_training:
                break

        return history


# ============================================================
# Metrics
# ============================================================
@torch.no_grad()
def fidelity(nn_state, target, space=None, **kwargs):
    """
    Fidelity = | <target | ψ> |^2 with both states normalized.

    This is strictly for diagnostics / plotting, *not* part of training.
    """
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

    For each basis b in `bases`:
      - Rotate target and model into that basis (apply ⊗_j U_bj),
      - Compute Born distributions,
      - Renormalize per basis,
      - Accumulate KL(p_target || p_model).

    Again, diagnostics only. This is how we check if tomography is
    matching the actual measurement statistics in all the bases
    we care about.
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
    tgt_norm = tgt / nt  # normalized target ψ

    psi_norm_cd = nn_state.psi_complex_normalized(space).reshape(-1)

    KL_val = 0.0
    eps = 1e-12

    for basis in bases:
        # Rotate both states into that basis
        tgt_psi_r = rotate_psi(nn_state, basis, space, psi=tgt_norm)
        psi_r     = rotate_psi(nn_state, basis, space, psi=psi_norm_cd)

        nn_probs_r  = (psi_r.abs().to(DTYPE)) ** 2      # model distribution in basis b
        tgt_probs_r = (tgt_psi_r.abs().to(DTYPE)) ** 2  # target distribution in basis b

        # Per-basis renormalization
        p_sum = tgt_probs_r.sum().clamp_min(eps)
        q_sum = nn_probs_r.sum().clamp_min(eps)
        p = (tgt_probs_r / p_sum).clamp_min(eps)
        q = (nn_probs_r / q_sum).clamp_min(eps)

        KL_val += torch.sum(p * (torch.log(p) - torch.log(q)))

    return (KL_val / len(bases)).item()


# ============================================================
# Dataset + loader
# ============================================================
class TomographyDataset:
    """
    Minimal dataset container for tomography:

    - train_samples: (N, n) float64 on `device`, each row is a measured
      bitstring in *its* measurement basis row.
    - train_bases  : (N, n) array of {'X','Y','Z'} on CPU describing that row's basis.
    - target_state : (2^n,) complex (cdouble) on `device` giving the known
      "true" ψ for metrics (synthetic / simulation case).
    - bases        : (M, n) array of bases that we will evaluate KL/fidelity on.

    We also precompute which rows were measured in all-Z, because ONLY those
    rows can seed the CD negative phase safely (they correspond to actual
    computational basis samples, i.e. direct draws from |ψ|^2 in Z).
    """
    def __init__(self, train_path, psi_path, train_bases_path, bases_path, device: torch.device = DEVICE):
        self.device = device

        # (N, n) measured bitstrings, already 0/1 floats
        self.train_samples = torch.tensor(
            np.loadtxt(train_path, dtype="float32"),
            dtype=DTYPE, device=device
        )

        # target wavefunction amplitudes as (Re, Im), one row per computational basis state
        psi_np = np.loadtxt(psi_path, dtype="float64")
        self.target_state = torch.tensor(
            psi_np[:, 0] + 1j * psi_np[:, 1],
            dtype=torch.cdouble, device=device
        )

        # basis metadata stays on CPU (tuples of 'X','Y','Z')
        self.train_bases = np.loadtxt(train_bases_path, dtype=str)
        self.bases = np.loadtxt(bases_path, dtype=str, ndmin=1)

        # Indices of rows that are pure Z measurements.
        # These are used to seed CD-negative-phase because they're actual
        # computational basis samples (no rotation).
        tb = np.asarray(self.train_bases)
        z_mask_np = (tb == "Z").all(axis=1)
        self._z_mask = torch.as_tensor(z_mask_np, dtype=torch.bool)
        self._z_indices = self._z_mask.nonzero(as_tuple=False).view(-1)

        # Sanity checks
        if self.train_samples.shape[0] != len(self.train_bases):
            raise ValueError("TomographyDataset: sample count != basis row count.")
        if self._z_indices.numel() == 0:
            raise ValueError("TomographyDataset: no Z-only rows for negative sampling.")

        widths = {len(row) for row in self.train_bases}
        if len(widths) != 1:
            raise ValueError("TomographyDataset: inconsistent basis widths.")
        n = next(iter(widths))
        if n != self.train_samples.shape[1]:
            raise ValueError("TomographyDataset: basis width != sample width.")

    def __len__(self):
        return int(self.train_samples.shape[0])

    def num_visible(self) -> int:
        return int(self.train_samples.shape[1])

    def z_indices(self) -> torch.Tensor:
        """Indices (CPU long) for Z-only rows. These seed the CD chain."""
        return self._z_indices.clone()

    def train_bases_as_tuples(self) -> List[Tuple[str, ...]]:
        """Return bases row-by-row as tuples of strings ('X','Y','Z',...)."""
        return [tuple(row) for row in np.asarray(self.train_bases, dtype=object)]

    def eval_bases(self) -> List[Tuple[str, ...]]:
        """
        Bases we want to evaluate KL/fidelity on (e.g. all unique bases
        from the experiment, or some canonical set).
        """
        return [tuple(row) for row in np.asarray(self.bases, dtype=object)]

    def target(self) -> torch.Tensor:
        """The known/simulated target ψ, as a cdouble vector on device."""
        return self.target_state


class RBMTomographyLoader:
    """
    Simple minibatch loader that yields triples:
        (pos_batch, neg_batch, bases_batch)

    - pos_batch: shuffled measurement outcomes from *all* bases
    - neg_batch: rows drawn ONLY from Z-basis pool, used to seed CD-k
    - bases_batch: the matching per-row basis labels for pos_batch

    We keep neg_batch separate because only Z-basis samples are valid Gibbs
    seeds for the amplitude RBM's contrastive divergence step.
    """
    def __init__(
            self,
            dataset: TomographyDataset,
            pos_batch_size: int = 100,
            neg_batch_size: Optional[int] = None,
            device: torch.device = DEVICE,
            dtype: torch.dtype = DTYPE,
            strict: bool = True,
    ):
        self.ds = dataset
        self.pos_bs = int(pos_batch_size)
        self.neg_bs = int(neg_batch_size or pos_batch_size)
        self.device = device
        self.dtype = dtype
        self.strict = strict
        self._gen: Optional[torch.Generator] = None  # optional independent RNG for reproducibility

        n = self.ds.num_visible()
        bases_seq = self.ds.train_bases_as_tuples()
        if any(len(row) != n for row in bases_seq):
            raise ValueError("RBMTomographyLoader: inconsistent basis widths in dataset.")
        if self.ds.z_indices().numel() == 0:
            raise ValueError("RBMTomographyLoader: Z-only pool is empty (need negatives).")

    def set_seed(self, seed: Optional[int]):
        """
        Optional deterministic shuffling, if you want reproducibility between runs.
        """
        if seed is None:
            self._gen = None
        else:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            self._gen = g

    def __len__(self):
        return ceil(len(self.ds) / self.pos_bs)

    def iter_epoch(self):
        """
        Yield batches for a single epoch:

            for pos_batch, neg_batch, bases_batch in loader.iter_epoch():
                ...

        We:
        - shuffle all rows to create pos_batch chunks,
        - for each chunk, also sample neg_batch_size many Z-only rows
          (with replacement) to feed the CD-k negative phase.
        """
        N = len(self.ds)
        n_batches = ceil(N / self.pos_bs)

        # One-pass shuffle for positives
        perm = torch.randperm(N, generator=self._gen) if self._gen is not None else torch.randperm(N)
        pos_samples = self.ds.train_samples[perm].to(self.device, dtype=self.dtype)

        bases_list = self.ds.train_bases_as_tuples()
        perm_idx = perm.detach().cpu().tolist()
        pos_bases_perm = [bases_list[i] for i in perm_idx]

        # Draw all negatives from Z-only pool (with replacement)
        z_pool = self.ds.z_indices()
        pool_len = z_pool.numel()
        if self._gen is None:
            neg_choices = torch.randint(pool_len, size=(n_batches * self.neg_bs,))
        else:
            neg_choices = torch.randint(pool_len, size=(n_batches * self.neg_bs,), generator=self._gen)
        neg_rows = z_pool[neg_choices]
        neg_samples_all = self.ds.train_samples[neg_rows].to(self.device, dtype=self.dtype)

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
                    raise RuntimeError("Loader invariant broken: bases_batch length mismatch.")
                if pos_batch.shape[1] != self.ds.num_visible():
                    raise RuntimeError("Loader invariant broken: pos_batch width != num_visible.")
                if neg_batch.shape[1] != self.ds.num_visible():
                    raise RuntimeError("Loader invariant broken: neg_batch width != num_visible.")

            yield pos_batch, neg_batch, bases_batch


# ============================================================
# Standalone training script section
# ============================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # -------------------------------
    # Data file paths (adapt to your data)
    # -------------------------------
    train_path = "w_state_meas.txt"
    train_bases_path = "w_state_basis.txt"
    psi_path = "w_state_aug.txt"
    bases_path = "w_state_bases.txt"

    # -------------------------------
    # Seeds
    # -------------------------------
    torch.manual_seed(1234)
    # np.random.seed(1234)  # optional if you want NumPy RNG sync

    # -------------------------------
    # Dataset / model
    # -------------------------------
    data = TomographyDataset(
        train_path,
        psi_path,
        train_bases_path,
        bases_path,
        device=DEVICE,
    )

    U = create_dict()  # local basis rotations {X,Y,Z} with our chosen Y convention

    nv = data.num_visible()
    nh = nv  # hidden = visible by default, can tune
    nn_state = ComplexWaveFunction(
        num_visible=nv,
        num_hidden=nh,
        unitary_dict=U,
        device=DEVICE,
    )

    # -------------------------------
    # Hyperparams
    # -------------------------------
    epochs = 70
    pbs = 100
    nbs = 100
    lr = 1e-1          # keep SGD ~0.1 to mimic classical CD training
    k_cd = 10          # CD-k steps
    log_every = 5

    loader = RBMTomographyLoader(
        data,
        pos_batch_size=pbs,
        neg_batch_size=nbs,
        device=DEVICE,
        dtype=DTYPE,
    )
    # loader.set_seed(1234)  # uncomment for deterministic batches

    # Hilbert space for metrics (only OK for tiny n)
    space = nn_state.generate_hilbert_space()

    history = nn_state.fit(
        loader,
        epochs=epochs,
        k=k_cd,
        lr=lr,
        log_every=log_every,
        optimizer=torch.optim.SGD,
        optimizer_args=None,
        target=data.target(),
        bases=data.eval_bases(),
        space=space,
        print_metrics=True,
        metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}",
    )

    # -------------------------------
    # Phase comparison diagnostic
    # -------------------------------
    # We align the global phase and then compare the learned phase vs target
    # for the basis states that carry most of the Born probability mass.
    with torch.no_grad():
        space = nn_state.generate_hilbert_space()
        psi_m = nn_state.psi_complex_normalized(space).reshape(-1).contiguous()
        psi_t = data.target().to(device=DEVICE, dtype=torch.cdouble).reshape(-1).contiguous()

        # normalize both
        nm = torch.linalg.vector_norm(psi_m)
        nt = torch.linalg.vector_norm(psi_t)
        if nm > 0:
            psi_m = psi_m / nm
        if nt > 0:
            psi_t = psi_t / nt

        # align one global phase so phase comparisons are meaningful
        ip = torch.sum(psi_t.conj() * psi_m)
        if ip.abs() > 1e-12:
            theta = torch.angle(ip)
        else:
            # fallback: align so that their largest-amplitude basis state matches
            j = int(torch.argmax(psi_t.abs()))
            theta = torch.angle(psi_m[j]) - torch.angle(psi_t[j])
        psi_m_al = psi_m * torch.exp(-1j * theta)

        phi_t = torch.angle(psi_t).cpu().numpy()
        phi_m = torch.angle(psi_m_al).cpu().numpy()

        # wrapped Δphase in [-π, π]
        dphi = np.remainder((phi_m - phi_t) + np.pi, 2.0 * np.pi) - np.pi

        # focus on the high-probability support (top 99% mass, cap at 512 points)
        probs = (psi_t.abs() ** 2).cpu().numpy()
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        mass_cut = 0.99
        k_cap = 512
        k_sel = int(min(np.searchsorted(cum, mass_cut) + 1, k_cap, len(order)))
        sel = order[:k_sel]

        # target vs model phases
        fig_p, axp = plt.subplots(figsize=(7.2, 3.8), dpi=150)
        axp.plot(range(k_sel), phi_t[sel], marker=".", linestyle="", label="target phase")
        axp.plot(range(k_sel), phi_m[sel], marker="x", linestyle="", label="model phase (aligned)")
        axp.set_xlabel("basis states (sorted by target mass)")
        axp.set_ylabel("phase [rad]")
        axp.set_title("Phase comparison – top 99% mass")
        axp.grid(True, alpha=0.3)
        axp.legend()
        fig_p.tight_layout()

        # wrapped phase error
        fig_e, axe = plt.subplots(figsize=(7.2, 3.8), dpi=150)
        axe.plot(range(k_sel), dphi[sel], marker=".", linestyle="", label="Δphase (wrapped)")
        axe.axhline(0.0, linewidth=1.0)
        axe.set_xlabel("basis states (sorted by target mass)")
        axe.set_ylabel("Δphase [rad] in [-π, π]")
        axe.set_title("Phase error (global phase aligned)")
        axe.grid(True, alpha=0.3)
        axe.legend()
        fig_e.tight_layout()

    # -------------------------------
    # Metrics plot (Fidelity & KL over epochs)
    # -------------------------------
    ep_hist = history.get("epoch", [])
    if ep_hist and ("Fidelity" in history) and ("KL" in history):
        fig, ax1 = plt.subplots(figsize=(6.0, 4.0), dpi=140)
        ax2 = ax1.twinx()

        ax1.plot(ep_hist, history["Fidelity"], marker="o", label="Fidelity")
        ax2.plot(ep_hist, history["KL"], marker="s", linestyle="--", label="KL")

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel(r"$|\langle \psi_t \mid \psi \rangle|^2$")
        ax2.set_ylabel(r"KL$(p\,\|\,q)$")
        ax1.set_title("RBM Tomography – training metrics (stabilized autodiff CD)")
        ax1.grid(True, alpha=0.3)

        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="best")

        fig.tight_layout()
    else:
        print("No metrics to plot (did you pass target & bases and set log_every?).")

    plt.show()