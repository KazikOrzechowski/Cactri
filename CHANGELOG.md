# Changelog

## 0.4.3 — 2026-08-13

### Packaging and documentation

- Synchronized package, import, citation, and validation metadata at version 0.4.3.
- Removed stale build products, Python/Numba caches, and generated artifacts from the source tree.
- Added release-oriented `.gitignore` rules.
- Moved old release/development documents into `docs/history/`.
- Consolidated the post-fit tree-pruning/refinement API in `cactri.pruning`.
- ACT and ACT-initializer modules are intentionally not included in this release.

### Compatibility

- No statistical-model, sampler, checkpoint-format, or public top-level model API changes.
- Existing 0.4.x checkpoint compatibility behavior is unchanged.

## 0.4.1

### Fixed

- `fit()` no longer eagerly constructs dense cell-by-cell co-assignment matrices; posterior co-assignment matrices are computed only when requested.
- Active residual Dirichlet simplexes remain strictly positive after floating-point underflow; exact zeros remain reserved for structurally pure hyperclusters.

## 0.4.0 — 2026-08-01

### Added

- Opt-in dominant-clone sparse-admixture model shared by Tree and Omega.
- Cell-level mutation clone assignments within BCR hyperclusters.
- Exact pure-hypercluster spike and sparse Beta-Dirichlet admixture.
- Uniform and tree-distance residual clone priors.
- Mixture-aware posterior summaries and cell clone co-clustering.
- Mixture-enabled CRP concentration sampling.
- v0.1/v0.2 checkpoint migration to a pure-mixture representation.

### Compatibility

- The release branches directly from 0.2.2.
- Mixtures are disabled by default and legacy seeded behavior is unchanged.
- The 0.3.x sampler line is discontinued and its checkpoints are rejected.

All notable changes are recorded by release. Existing release directories and
archives are immutable; new work is published as a new semantic version.

## 0.2.1 — Stage 2, maintenance milestone

### Added

- Cached split/merge sufficient statistics for repeated proposals in one sweep.
  Cached quantities include the cell-by-clone mutation likelihood matrix,
  cluster membership, BCR residue counts, clone-marginal likelihood sums, and
  collapsed cluster scores.
- Adaptive split-versus-merge anchor scheduling through
  `SplitMergeConfig(anchor_strategy="adaptive")`.
- State-dependent adaptive anchor probabilities and their reverse probabilities
  in the Metropolis-Hastings correction.
- Configurable adaptation interval, step size, probability bounds, and optional
  `adapt_until` cutoff.
- Exact checkpoint/resume through `save_checkpoint()` and `load_checkpoint()`.
  Checkpoints preserve model parameters, latent state, RNG state, adaptive
  scheduler state, diagnostics, and optionally retained traces.
- Backend override when loading a checkpoint, allowing exact continuation from
  NumPy under Numba or vice versa.
- Label-invariant hypercluster and clone partition-medoid helpers.
- Optional partition medoids in `posterior_summary()`.
- Global iteration accounting so tracking cadence remains unchanged when a fit
  is split across several calls or resumed from a checkpoint.
- Cache-build, cache-reuse, and adaptive split-probability diagnostics.

### Changed

- `SplitMergeConfig.cache_sufficient_statistics` defaults to `True`.
- The original uniformly sampled cell-pair proposal remains the default through
  `anchor_strategy="uniform_pair"`; therefore 0.2.0 proposal semantics are
  preserved unless adaptive scheduling is requested.
- `posterior_summary()` accepts `include_partition_medoids=False` to avoid the
  quadratic partition-loss calculation unless requested.

### Reproducibility

- Cached and uncached proposals consume the same random-number stream and
  produce identical seeded chains.
- Adaptive scheduling remains backend independent; NumPy and Numba receive the
  same model-owned random values and produce identical seeded chains.
- Checkpoint/resume is tested against uninterrupted execution, including
  tracking cadence and adaptive state.
- A checkpoint created under NumPy can continue under Numba with exact seeded
  algorithmic identity.

### Performance

- In the included 300-cell, 126-mutation synthetic proposal benchmark, caching
  reduced 50 global-proposal runtime from approximately 0.50 s to 0.40 s while
  producing the identical chain. Performance gains depend on cluster count,
  proposal count, and acceptance rate.

## 0.2.0 — Stage 2, milestone 1

### Added

- Collapsed mutation-informed restricted-Gibbs split/merge proposals.
- Exact clone-marginal mutation evidence in split/merge partition scoring.
- `SplitMergeConfig` with configurable local assignment transition and proposal count.
- `TrackingConfig.posterior()` for coherent posterior-summary traces.
- Trace-based posterior cell-clone probabilities.
- Coherent posterior cell-genotype probabilities, combining clone assignment and
  genotype state within each draw before averaging.
- Label-invariant posterior hypercluster co-assignment matrices.
- Split/merge acceptance diagnostics in the model and result dictionary.

### Changed

- Standard absent-genotype observation prior changed from `Beta(1, 99)` to
  `Beta(1, 999)`, matching the default simulated background rate of about 0.001.
- `posterior_cell_clone_probabilities()` and
  `posterior_cell_genotype_probabilities()` prefer retained draws when compatible
  traces exist. The former Stage-1 calculations remain available as
  `current_state_cell_clone_probabilities()` and
  `current_state_cell_genotype_probabilities()`.
- `assignment_sampler="split_merge"` now uses a vectorized approximate local
  sweep by default, followed by global proposals.

### Reproducibility

- NumPy remains the sole random-number source.
- The split/merge kernel is backend-independent and consumes the same model RNG
  stream under NumPy and Numba.
- Exact seeded NumPy/Numba chain identity is tested with split/merge enabled.

## 0.1.0 — Stage 1

- Introduce the abstract `Cactri` base class.
- Add `CactriTree`, `CactriOmega`, and standalone `BCRInitializer`.
- Add optional NumPy/Numba acceleration with model-owned random draws.
- Preserve legacy result aliases and explicit assignment-vector initialization.

## 0.2.2

- Added `CactriTree(observed_edge_probabilities=...)` for probabilistic supplied
  mutation-edge estimates.
- Added a learned direct-versus-distance-smoothed reliability mixture for
  probabilistic edge estimates.
- Added `SplitMergeConfig.max_restricted_cells` with an MH-correct eligible-pair
  proposal.
- Added NumPy/Numba algorithmic-identity and validation tests for the new tree
  prior.
