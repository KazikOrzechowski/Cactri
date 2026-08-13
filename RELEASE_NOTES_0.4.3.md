# Cactri 0.4.3

Release-hygiene patch for the 0.4.x lineage.

## Included

- Core `Cactri`, `CactriTree`, `CactriOmega`, and `BCRInitializer` APIs.
- Existing dominant-clone sparse-admixture and posterior-summary functionality.
- Canonical post-fit pruning/refinement utilities under `cactri.pruning`:
  `GreedyTreePruner`, `ReadImpuritySplitBack`, and `PrunedTreeRefiner`, together
  with their result records.

## Excluded

- ACT and ACT-initializer classes/modules.
- Legacy standalone pruning module paths; pruning utilities live in
  `cactri.pruning`.

## Release hygiene

- Version metadata synchronized to 0.4.3.
- Stale wheels/sdists and generated Python/Numba caches removed before rebuild.
- Historical release/development documents moved under `docs/history/`.
- Added `.gitignore` rules for caches, build products, virtual environments, and
  editor artifacts.

## Compatibility

This release does not intentionally change the statistical model, sampler
semantics, checkpoint format, or existing top-level model API.
