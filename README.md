# Cactri

Cactri is a research package for joint BCR hypercluster inference and clonal
mutation-genotype inference from single-cell sequence and read-count data.

The package is organized around an abstract shared model:

- `CactriTree`: mutation origins on a fixed full binary tree, with an optional
  learned distance-aware error model for a supplied mutation tree.
- `CactriOmega`: independent clone-by-mutation genotypes, optionally centered
  on a supplied genotype matrix through a learned relaxation rate.
- `BCRInitializer`: standalone BCR-only CRP/Dirichlet initialization and
  multi-chain consensus fitting.

## Installation

```bash
pip install .
```

Optional Numba acceleration:

```bash
pip install ".[numba]"
```

## Basic use

```python
from cactri import (
    BCRInitializer,
    CactriTree,
    SplitMergeConfig,
    TrackingConfig,
)

initializer = BCRInitializer(random_state=1)
initializer.consensus_fit(
    bcr_sequences,
    n_chains=4,
    n_iter=500,
)

model = CactriTree(
    n_levels=3,
    observed_edge_assignment=wes_edge_assignment,
    random_state=2,
)
model.prefit(
    bcr_sequences,
    alt_counts,
    total_counts,
    init=initializer.consensus_assignments_,
)
result = model.fit(
    n_iter=1_000,
    assignment_sampler="split_merge",
    split_merge_config=SplitMergeConfig(
        local_sampler="approximate",
        proposals_per_sweep=2,
    ),
    tracking=TrackingConfig.posterior(every=5),
)

posterior = model.posterior_summary(burn_in=0.5)
cell_genotype_probability = posterior["cell_genotype_probabilities"]
```

`prefit(init="bcr_consensus")` can run `BCRInitializer` internally. Explicit
cell-to-hypercluster vectors and the named modes `one_cluster`, `random`, and
`identical_sequences` are also supported.

## Assignment samplers

- `approximate` — vectorized approximate CRP sweep; default.
- `sequential` — cell-by-cell reassignment.
- `split_merge` — a configurable local sweep followed by collapsed,
  mutation-informed restricted-Gibbs split/merge proposals.

The split/merge target integrates out BCR profiles and hypercluster-to-clone
labels. Mutation evidence therefore participates directly in proposals that can
split a BCR cluster into clone-homogeneous components.

## Posterior summaries

For posterior summaries, retain compatible traces:

```python
tracking = TrackingConfig.posterior(every=5)
model.fit(1_000, tracking=tracking)
summary = model.posterior_summary(burn_in=0.5)
```

The cell-genotype summary is coherent: for every retained draw, cell clone
labels are combined with the genotype state from that same draw before averaging.
Hypercluster uncertainty is summarized with a label-invariant cell
co-assignment matrix.

Stage-1 current-state calculations remain available through:

```python
model.current_state_cell_clone_probabilities()
model.current_state_cell_genotype_probabilities()
```

## Observation defaults

Version 0.2.0 standardizes the absent-genotype observation prior to
`p_unobs ~ Beta(1, 999)`. Pass `p_unobs_beta_prior=(1, 99)` to reproduce the
Stage-1 default.

## NumPy/Numba identity policy

Cactri owns all random-number generation. The same pre-generated uniforms are
passed to NumPy and Numba kernels. Exact seeded chain identity is tested for both
models, including split/merge sampling. Numba kernels do not maintain hidden
random state.

## Versioning

Cactri uses semantic versioning. Release source trees and wheels are immutable.
Model or transition-kernel changes are introduced in a new minor version and
recorded in `CHANGELOG.md` and `MIGRATION.md`.

## License

GNU General Public License v3.0 only.
