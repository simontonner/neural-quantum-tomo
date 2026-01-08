#### TFIM diagnostics | ED-only story
# TFIM ED HARDNESS DIAGNOSTICS

#### What this is

This diagnostic is meant to explain why a Z-basis-trained generative model (RBM / cRBM / HyperRBM) can struggle in a specific field region, even when the physics "criticality" (as seen by fidelity susceptibility) sits somewhere else.

Everything here is computed from exact diagonalization (ED) ground states on a finite lattice.

#### Output artifacts

The script `tfim_ed_hardness_story.py` writes:

- `tfim_ed_hardness_story.csv`
- `tfim_ed_hardness_story_report.md`
- A single story plot with
    - left axis: chi_F(h)
    - right axis: Neff_norm, TC_norm, |Cnn_conn|

#### Column definitions

Let P_h(s) = |psi_h(s)|^2 in the computational (sigma^z) basis.

- `chiF_overlap`:
  Fidelity susceptibility from neighbor overlap (gauge-safe):
  chi_F(h) = 2 * (1 - |<psi(h)|psi(h+dh)>|) / dh^2

  Useful rule of thumb:
  1 - F(h) ≈ (1/2) * chi_F(h) * (delta h)^2

- `Neff`:
  Renyi-2 effective support:
  Neff(h) = 1 / sum_s P_h(s)^2

- `Neff_norm`:
  Support fraction: Neff / 2^N
  Interpretable as "what fraction of the whole basis effectively matters".

- `Cnn_conn`:
  Nearest-neighbor connected ZZ correlator:
  C_nn,conn = <z_i z_j> - <z_i><z_j> averaged over NN edges

- `Cnn_conn_abs_mean`:
  Mean absolute connected correlator over edges.
  Typically close to |Cnn_conn| if the sign structure is uniform.

- `H_joint_nats`:
  Shannon entropy H(P_h) = -sum_s P_h(s) log P_h(s), in nats.

- `H_single_sum_nats`:
  Sum of single-site entropies sum_i H(p_i), where p_i is the marginal for spin i in the Z basis.

- `TC_nats`:
  Total correlation (multi-information):
  TC = sum_i H(p_i) - H(P_h)

  TC is high when the distribution is both nontrivial at single-site level AND still strongly dependent across sites.

- `TC_norm`:
  TC normalized by the maximum possible value N log 2:
  TC_norm = TC / (N log 2) in [0,1] (up to finite-size / numerical noise)

- `gap_naive_E1_E0` (optional):
  Naive gap E1 - E0 from full Hilbert space.
  Warning: at small h on finite size, this is often dominated by even/odd cat splitting, so treat it as a sanity check only.

#### How to read the story plot

- chi_F(h) tells you "sensitivity to h":
  where small parametric errors delta h cause larger fidelity loss.

- Neff_norm tells you "support size":
  how broad P_h is in the Z basis (data burden + representational burden).

- TC_norm tells you "dependence":
  whether the broadness comes with nontrivial multi-site structure (harder than a near-i.i.d. broad distribution).

- |Cnn_conn| is a local sanity check:
  local structure is still present even if the distribution is broad.

#### Typical RBM-hard regime

A common RBM pain point is where:
- Neff_norm is already increasing (more configurations matter),
  but
- TC_norm and/or |Cnn_conn| are still substantial (still structured, not i.i.d.)

This is the "broad AND dependent" regime. It often sits slightly away from the chi_F peak on finite sizes.

#### One extra model-side diagnostic (optional but decisive)

If you have model overlaps F_model(h) against ED, define:

```
delta_eff(h) = sqrt( 2 * (1 - F_model(h)) / chi_F(h) )
```

- If delta_eff(h) is flat while F dips near chi_F peak: sensitivity amplification dominates.
- If delta_eff(h) spikes near some h (often ~3 in practice): real model/training mismatch concentrated there.
