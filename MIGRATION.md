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

## From Cactri 0.2.0 to 0.2.1

No existing public call is removed and the default split/merge anchor proposal
remains `uniform_pair`.

### Cached global proposals

Caching is enabled by default and does not change the seeded chain:

```python
config = SplitMergeConfig(
    proposals_per_sweep=4,
    cache_sufficient_statistics=True,
)
```

Set `cache_sufficient_statistics=False` for profiling or characterization of the
uncached path.

### Adaptive split/merge scheduling

Adaptive scheduling is opt-in:

```python
config = SplitMergeConfig(
    anchor_strategy="adaptive",
    initial_split_probability=0.5,
    adaptation_interval=25,
    adaptation_step=0.25,
    adapt_until=1_000,
)
```

The proposal selection probability is included in the MH correction. For formal
posterior sampling, use `adapt_until` to stop adaptation after warm-up.

### Checkpoint and resume

```python
model.fit(500, tracking=TrackingConfig.posterior(every=5))
model.save_checkpoint("chain.cactri.gz")

resumed = CactriTree.load_checkpoint("chain.cactri.gz")
resumed.fit(500, tracking=TrackingConfig.posterior(every=5))
```

The global iteration counter preserves `TrackingConfig.every` across resumed or
repeated `fit()` calls. Checkpoints use pickle and must only be loaded from a
trusted source.

A backend may be selected at load time:

```python
resumed = CactriTree.load_checkpoint(
    "chain.cactri.gz",
    accelerator="numba",
)
```

### Partition medoids

```python
hypercluster_medoid = model.posterior_hypercluster_partition_medoid(
    burn_in=0.5
)
clone_medoid = model.posterior_clone_partition_medoid(burn_in=0.5)
```

Or request both through:

```python
summary = model.posterior_summary(
    burn_in=0.5,
    include_partition_medoids=True,
)
```

## Migrating from 0.2.1 to 0.2.2

No existing call requires modification.

To use a mutation-tree initializer posterior directly, replace MAP reduction:

```python
CactriTree(observed_edge_assignment=edge_posterior.argmax(axis=1))
```

with:

```python
CactriTree(observed_edge_probabilities=edge_posterior)
```

The two interpretations intentionally remain available for empirical
comparison. To bound the cost of global proposals, add
`max_restricted_cells` to `SplitMergeConfig`; omitting it retains the original
proposal.


## Migrating from 0.2.2 to 0.4.0

Version 0.4.0 is based directly on 0.2.2. Mixtures are opt-in, so existing
constructors and fits retain the exact v0.2.2 model and seeded transition.

Enable the extension with:

```python
from cactri import CloneMixtureConfig

model = CactriTree(
    n_levels=3,
    clone_mixture=CloneMixtureConfig(enabled=True),
)
```

The old `hypercluster_to_clone_` vector is now the dominant-clone vector. The
new sampled cell-level clone labels are returned by
`current_cell_clone_assignment()` and stored in `cell_clone_assignments_`.

New canonical state attributes:

```python
model.dominant_clones_
model.hypercluster_clone_proportions_
model.cell_clone_assignments_
model.admixture_mass_
model.residual_clone_proportions_
model.mixture_active_
```

For coherent mixture posterior summaries, use:

```python
tracking = TrackingConfig.posterior(mixture_diagnostics=True)
model.fit(1_000, tracking=tracking)
```

`split_merge` remains available in legacy mode. It is intentionally unavailable
with clone mixtures because the retained v0.2 collapsed target assumes one clone
label per hypercluster.

Trusted v0.1/v0.2 checkpoints migrate automatically to exact pure rows. v0.3.x
checkpoints are rejected because that experimental line is not an ancestor of
0.4.0.

## From the discontinued 0.3.x branch

There is no direct checkpoint migration. Recreate the model from source data or
export the required arrays under 0.3.x and initialize a new 0.4.0 model. The
blocked and hybrid partition kernels are not carried forward.
