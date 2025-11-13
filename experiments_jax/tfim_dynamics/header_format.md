FOR THE TFIM

HEADER | system=TFIM_3x3 | J=1.00 | h=0.5 | nqubits=9 | basis=computational | samples=10000 | seed=42


FOR THE XXZ HEISENBERG MODEL

HEADER | system=XXZ_3x3 | J=1.00 | delta=0.5 | nqubits=9 | basis=mixed | samples=10000 | seed=42


FOR THE PHASE AUGMENTED W STATE

HEADER | system=W_phase_augmented | nqubits=4 | basis=ZXYZ | samples=10000 | seed=42










------ how the old code did it:


Short: **Your loader mixes bases in each batch; the model re-groups per basis inside `gradient()` and processes each basis bucket separately.**

**Where it happens:**

* `RBMTomographyLoader.iter_epoch()` → yields `(pos_batch, neg_batch, bases_batch)` with **heterogeneous** `bases_batch` (it’s just a slice of the global shuffle).
* `ComplexWaveFunction.compute_batch_gradients(...)` → calls `positive_phase_gradients(...)` → `gradient(samples, bases=bases_batch)`.
* `gradient(...)`:

    * Builds `buckets = {basis_tuple: [row_indices]...}`.
    * For each bucket:

        * if the basis has any non-Z, calls `rotated_gradient(basis_t, samples[idxs])`.
        * if all-Z, uses `rbm_am.effective_energy_gradient(...)`.
    * Accumulates into `G_am`, `G_ph`.

**Net effect:** you **feed mixed bases**, but the **math is done per-basis** under the hood—basically the same as per-basis micro-batches, just without Python-side batching overhead.

**Negatives:** always drawn from the Z-only pool and used only for the amplitude RBM in the negative phase—independent of the measurement bases.

**If you want true per-basis micro-batches:** change the loader to yield one basis at a time; performance will be similar since `gradient()` already buckets—this would mainly simplify mental model, not speed.



THE NEW CODE:

Short: **Your newer JAX stack uses *homogeneous-basis* microbatches and accumulates grads.** No mixed bases inside a batch.

**Flow (where it happens):**

* **Grouping at load time:** `PerBasisTomographyDataset.groups()` builds index buckets per basis; `z_indices()` marks Z-only rows.
* **Homogeneous batches:** `RBMTomographyLoaderGrouped.iter_epoch()`

    * Randomizes basis order each epoch, shuffles rows *within* each basis.
    * Yields `(pos_batch, neg_batch, basis_tuple, is_z)` with **one basis per batch**.
    * Negatives are drawn epoch-wide from the Z-pool, independent of `basis_tuple`.
* **Per-basis kernels (compiled):** `Trainer.fit(...)`

    * Pulls cached meta via `nn.get_basis_meta(basis_tuple)` → `(Uc_flat, sites, combos)`; cache lives in `ComplexWaveFunction._basis_cache`.
    * Calls `@jit _grads_only(..., k, is_z)`; `is_z` is **static**, so XLA makes two traces: Z-path vs rotated-path.
* **Losses:**

    * Z batches: `_pos_loss_z(params_am, pos_batch)` (no rotations).
    * Non-Z batches: `_pos_loss_rot(params_am, params_ph, pos_batch, Uc_flat, sites, combos)` using the **stable log-sum** kernel `_stable_log_overlap_amp2_with_meta(...)`.
* **Negative phase:** inside `_grads_only` via `_rbm_gibbs_steps(params["am"], k, neg_batch, rng)`; only the amplitude RBM contributes to the negative term.
* **Gradient accumulation:** in `Trainer.fit` you sum grads for `accum_steps` microbatches, then average and `opt.update(...)` (Optax `multi_transform`: separate LR for phase). Any remainder is flushed at epoch end.

**Why this design:** It’s JIT/XLA-friendly (fixed shapes per call, fewer polymorphic branches), exploits vectorization across samples that share the same basis, and keeps rotations precomputed/cached. If you *wanted* mixed-basis batches, you’d need to re-introduce per-batch bucketing inside the loss—worse for compilation and usually slower in JAX.


#### HOW THE TORCH VERSION HANDLES IT

**Short:** Here it’s **mixed-basis batches**, then **bucketized per basis inside the loss**.

**Where it happens**

* **Loader (`RBMTomographyLoader.iter_epoch`)** shuffles *all* rows and yields
  `(pos_batch, neg_batch, bases_batch)` — `bases_batch` can contain many different bases.
* **Loss (`ComplexWaveFunction._positive_phase_loss`)**:

    * Builds `buckets[basis] -> indices` from `bases_batch`.
    * For each bucket:

        * **non-Z basis:** calls `_stable_log_overlap_amp2(basis, samples[idxs])` (rotated likelihood).
        * **Z-only:** adds `self.rbm_am.effective_energy(samples[idxs])`.
    * Sums both parts → data term.
* **Negative phase** uses `neg_batch` drawn from the **Z-only pool** in the loader (epoch-wide sample, chunked per batch) and does CD-k on the **amplitude** RBM.

**Implications**

* One optimizer step per batch; **no gradient accumulation** here.
* Compute cost scales with the **number of unique bases present in the batch** (one rotated kernel call per unique basis).
* Fine in PyTorch; if batches have many tiny basis buckets, consider **grouping by basis in the loader** (your JAX style) or use **smaller pos_bs + grad accumulation**.

