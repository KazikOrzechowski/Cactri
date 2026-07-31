# Migration guide

## From Cactri 0.1.x to 0.2.0

### Observation prior

The default absent-genotype prior is now `Beta(1, 999)` instead of
`Beta(1, 99)`.

To reproduce 0.1.x behavior:

```python
model = CactriTree(
    n_levels=3,
    p_unobs_beta_prior=(1.0, 99.0),
)
```

### Split/merge sampler

The Stage-1 heuristic has been replaced by a collapsed restricted-Gibbs kernel.
The default local transition is the approximate assignment sweep:

```python
from cactri import SplitMergeConfig

model.fit(
    1_000,
    assignment_sampler="split_merge",
    split_merge_config=SplitMergeConfig(
        local_sampler="approximate",
        proposals_per_sweep=2,
    ),
)
```

Use `local_sampler="sequential"` for a full cell-wise scan, or
`local_sampler="none"` when global proposals are scheduled separately.

### Posterior versus current-state probabilities

When retained traces are available, these methods now return trace-based
posterior probabilities:

```python
model.posterior_cell_clone_probabilities()
model.posterior_cell_genotype_probabilities()
```

The 0.1.x calculations have explicit names:

```python
model.current_state_cell_clone_probabilities()
model.current_state_cell_genotype_probabilities()
```

Use `TrackingConfig.posterior()` to retain all required traces.

## From legacy_PGM

| Previous class | Cactri class |
|---|---|
| robust tree fitter | `cactri.CactriTree` |
| independent/Omega fitter | `cactri.CactriOmega` |
| BCR-only fitter | `cactri.BCRInitializer` |

```python
from cactri import CactriTree

model = CactriTree(
    n_levels=3,
    observed_edge_assignment=input_edges,
    random_state=1,
)
model.prefit(sequences, alt_counts, total_counts, init=initial_partition)
result = model.fit(1_000)
```

Deletion and pruning models remain in the separate legacy archive.
