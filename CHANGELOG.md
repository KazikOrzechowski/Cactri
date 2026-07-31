# Changelog

All notable changes are recorded by release. Existing release directories and
archives are immutable; new work is published as a new semantic version.

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
  sweep by default, followed by global proposals. This avoids an implicit full
  sequential scan on large datasets.

### Fixed

- Removed a stale clone-label resize at the end of the sequential assignment
  sweep, which could use the partition from only the final cell update.

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
