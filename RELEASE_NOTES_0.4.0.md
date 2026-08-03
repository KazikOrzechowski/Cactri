# Cactri 0.4.0 release notes

Cactri 0.4.0 branches directly from 0.2.2. The 0.3.x hybrid partition-sampler
experiment is discontinued and is not part of this release.

## Added

- Opt-in `CloneMixtureConfig` shared by `CactriTree` and `CactriOmega`.
- Cell-level mutation-clone labels within BCR hyperclusters.
- Dominant-clone prior with sparse Beta-Dirichlet residual admixture.
- Optional exact pure-hypercluster spike with sampled global admixture rate.
- Uniform and optional tree-distance residual clone bases.
- Exact conditional cell-clone sampling and mixture-marginalized hypercluster
  assignment scores.
- Cell-level mutation-count aggregation for Tree/Omega genotype and observation
  updates.
- CRP concentration sampling for mixture-enabled models.
- Selectable mixture tracking and posterior summaries, including cell clone
  co-clustering probabilities.
- Trusted v0.1/v0.2 checkpoint migration to exact pure-mixture state.

## Compatibility

- Mixtures are disabled by default.
- Disabled mode is bit-for-bit identical to v0.2.2 for approximate and
  split/merge chains in characterization tests.
- `hypercluster_to_clone_` remains an alias for the dominant clone vector.
- `assignment_sampler="split_merge"` is retained for legacy mode and rejected
  when clone mixtures are enabled because its v0.2 target assumes one clone per
  hypercluster.
- v0.3.x checkpoints are intentionally rejected with a lineage-specific error.

## Validation

- 59 package tests pass.
- Exact NumPy/Numba seeded identity passes for Tree and Omega mixture chains.
- Interrupted/checkpoint-resumed mixture chains exactly match uninterrupted
  chains.
- Actual v0.2.2 checkpoints continue exactly under v0.4.0.
- A real v0.3.1 checkpoint is decoded and rejected with the documented error.
- Exact-zero pure states remain finite and normalized.
