#### Autograd Rotated-Overlap Stability | RBM Tomography

# FUTURE IMPROVEMENTS PLAYBOOK: STABLE ROTATED LOG-OVERLAP

#### Goal

You currently compute (for a basis row `basis` and measured samples `states`):

$$
\log \left|\left\langle\sigma^{[b]} \mid \psi_{\lambda, \mu}\right\rangle\right|^2=\log \left|\sum_\sigma U_b\left(\sigma^{[b]}, \sigma\right) e^{-\frac{1}{2} F_\lambda(\sigma)} e^{-\frac{i}{2} F_\mu(\sigma)}\right|^2-\log Z_\lambda .
$$

In code, `_stable_log_overlap_amp2()` approximates the *log amplitude-squared* in rotated bases via branch enumeration and stabilization.

This document lists improvements to try **one-by-one**, with minimal changes per step, so you can measure impact (fidelity, stability, gradient stats) and stop when it’s “good enough”.

---

#### Baseline

Current implementation:

* enumerates branches via `_rotate_basis_state` → returns `Ut` and `v` (branches)
* computes `F_am, F_ph`
* uses polar split of `Ut`:

  * `log|Ut|` adds to log-magnitude
  * `angle(Ut)` adds to phase
* uses max-shift `M = max(logmag_total)`
* sums complex contributions, then returns:
  $$
  \log \text{amp}^2 = 2M + \log(|S'|^2 + \epsilon).
  $$

Keep this baseline unchanged for comparison.

**Record for each run**

* final fidelity
* training curve smoothness (spikes / NaNs)
* typical `S_abs2` distribution (min/median)
* max grad norm (before clipping)

---

#### Step 1: Remove `angle/abs` by using direct complex multiplication

**Reason**

* `angle(Ut)` introduces phase wrap and branch-cut discontinuities in the forward path.
* `log(abs(Ut))` + `angle(Ut)` is unnecessary because `Ut` is already complex.
* Direct complex multiplication typically gives smoother optimization and fewer “cancellation spikes”.

**Change**
Replace the polar split with a single complex coefficient:

$$
S = \sum_c U_t(c), \exp\left(-\tfrac12 F_\lambda(v_c) - \tfrac{i}{2}F_\mu(v_c)\right).
$$

**Implementation sketch**

* build complex branch amplitude:
  $$
  A_c = \exp\left((-\tfrac12 F_{am}) - \tfrac{i}{2} F_{ph}\right)
  $$
* multiply by `Ut` directly
* sum over branches

You can still keep your stabilization (Step 2 refines it).

**Expected outcome**

* fewer weird “phase jumps”
* more consistent gradients early in training

---

#### Step 2: Stabilize by shifting only the real amplitude part

**Reason**

* the overflow/underflow danger comes from $\exp(-\tfrac12 F_{am})$
* `Ut` is bounded (unitaries) and doesn’t need to be in the shift
* shifting only the real part is simpler and more robust

**Target form**
Let $a_c = -\tfrac12 F_{am}(v_c)$ and $\phi_c = -\tfrac12 F_{ph}(v_c)$.

Choose:
$$
M = \max_c a_c
$$

Then:
$$
S = e^{M}\sum_c U_t(c),\exp\left((a_c - M) + i\phi_c\right)
$$

and
$$
\log|S|^2 = 2M + \log\left|\sum_c U_t(c),\exp((a_c - M) + i\phi_c)\right|^2.
$$

**Implementation notes**

* compute `a = (-0.5 * F_am)` as `DTYPE`
* `M = a.max(dim=0, keepdim=True)`
* compute `A_shift = exp((a - M).to(cdouble) + (-0.5j * F_ph).to(cdouble))`
* sum `S_shift = (Ut * A_shift).sum(dim=0)`
* return `2*M.squeeze(0) + log(|S_shift|^2 + eps)`

**Expected outcome**

* numerically cleaner stabilization
* better interpretability (the “log-sum-exp analogue” is purely about amplitude)

---

#### Step 3: Make the $\epsilon$ handling explicit and tunable

**Problem**
When destructive interference makes $|S|^2 \approx 0$:

* `log(|S|^2 + eps)` is finite, but
* gradients can spike when `|S|^2` is tiny relative to `eps`

**Try these variants (one at a time)**

**3A - Increase `eps_rot` early**

* start with `eps_rot = 1e-4` or `1e-5`, decay to `1e-6` later

**3B - Clamp `S_abs2` before log**
$$
\log(\max(|S|^2, \epsilon))
$$
instead of $\log(|S|^2 + \epsilon)$

This changes gradients slightly but can prevent extreme spikes.

**3C - Track cancellation diagnostics**
Log these during training:

* `S_abs2.min()`, `S_abs2.median()`
* fraction of batch with `S_abs2 < 1e-12`

**Expected outcome**

* fewer sudden gradient explosions
* fewer “training got stuck after a spike” events

---

#### Step 4: Reduce cancellation variance by rephasing (optional)

**Reason**
Even with stable magnitudes, the complex sum can suffer from heavy cancellations.

A trick: factor out a reference phase to reduce oscillations.

Choose some reference $\phi_0$ (e.g. phase of the largest-magnitude branch), and rewrite:

$$
\sum_c z_c=e^{i \phi_0} \sum_c z_c e^{-i \phi_0}
$$

This does not change $|S|$, but can improve floating-point behavior.

**Implementation idea**

* pick index of max `a_c` for each sample
* use that branch phase as `phi0`
* multiply contributions by `exp(-1j*phi0)` before summing

**Expected outcome**

* small but sometimes noticeable stability improvement

---

#### Step 5: Replace “sum then square” with a ratio-form gradient (advanced)

**Goal**
Make the gradient behave like the classic expression:

$$
\nabla_\theta \log|S|^2 = 2\Re\left(\frac{\nabla_\theta S}{S}\right).
$$

Your explicit-gradient version is basically implementing this structure.

**Two options**

**5A - Keep autograd, but compute `S` and rely on it**

* often Step 1 + Step 2 is enough
* skip this unless you still see instability

**5B - Hybrid approach**

* compute `S` and `1/S` with a safeguard
* explicitly form a ratio-like expression for stability
* this starts looking like your explicit estimator again

**Expected outcome**

* lower-variance phase gradients in “hard interference” regimes
* more stable late-stage convergence

---

#### Step 6: Performance and scaling checks

This method enumerates branches:
$$
C = 2^S
$$
where $S$ is number of non-Z sites in the basis row.

For sliding windows (e.g. `XXZZ`, `XYZZ`) you’re fine. If you ever increase $S$, memory and compute blow up quickly.

**If you scale to larger $S$**

* consider sampling branches
* or restrict to local rotations

---

#### Recommended execution order

1. **Step 1**: Direct complex multiply (`Ut * exp(...)`)
2. **Step 2**: Shift by amplitude only (`M = max(-0.5 F_am)`)
3. **Step 3**: Tune `eps_rot` and/or clamp `S_abs2`
4. **Step 4** (optional): Rephase by max-branch phase
5. **Step 5** (only if needed): Ratio-form gradient stabilization
6. **Step 6**: Scaling guardrails if you increase non-Z support

Stop as soon as:

* fidelity improves or matches explicit-grad version
* training curves are smooth
* no NaNs/spikes without relying heavily on clipping

---

#### Checklist of what to log while testing

* `fidelity` every `log_every`
* `grad_norm` before clipping
* `S_abs2.min/median`
* fraction of batch with `S_abs2 < 1e-12`
* `M.mean()` (shift magnitude)

These diagnostics tell you *why* an improvement helped.

---

#### Notes specific to your current function

Your current line:

* `torch.log(Ut.abs().clamp_min(1e-300))` and `torch.angle(Ut)`

is exactly the polar split you should try to eliminate first.

Everything else (max-shift + eps) is already in the right direction.

---