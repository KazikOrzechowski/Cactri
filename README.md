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
from cactri import BCRInitializer, CactriTree, TrackingConfig

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
    assignment_sampler="approximate",
    tracking=TrackingConfig(genotype_state=True),
)
```

`prefit(init="bcr_consensus")` can run `BCRInitializer` internally. Explicit
cell-to-hypercluster vectors and the legacy named modes `one_cluster`, `random`,
and `identical_sequences` are also supported.

## Assignment samplers

- `approximate` — parallel approximate CRP sweep; default.
- `sequential` — cell-by-cell reassignment.
- `split_merge` — sequential sweep followed by a Stage-1 global split/merge
  Metropolis proposal.

## NumPy/Numba identity policy

Cactri owns all random-number generation. The same pre-generated uniforms are
passed to NumPy and Numba kernels. Exact discrete decisions are tested where
practical; floating-point likelihoods are tested with tight numerical
tolerances. Numba kernels do not maintain hidden random state.

## Development stage

Version 0.1.0 is the behavior-preserving first stage of the refactor. More
advanced mutation-informed split/merge kernels and model changes will be added
as explicit later-stage changes.

## License

GNU General Public License v3.0 only.
