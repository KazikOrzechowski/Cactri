# Checkpoints

Cactri checkpoints are gzip-compressed Python pickles. Load only trusted files.

## Exact continuation

The checkpoint contains the NumPy bit-generator state, current latent state,
mixture parameters, traces, CRP concentration, and completed-iteration count.
Temporary split/merge sufficient-statistic caches are excluded and rebuilt when
needed.

```python
model.save_checkpoint("chain.cactri.gz")
resumed = CactriTree.load_checkpoint("chain.cactri.gz")
```

A backend may be changed at load time without changing the seeded statistical
chain:

```python
resumed = CactriTree.load_checkpoint(
    "chain.cactri.gz",
    accelerator="numba",
)
```

## v0.1/v0.2 migration

Older checkpoints contain one clone label per hypercluster. Loading under 0.4.0
creates:

```text
dominant_clones[h] = old hypercluster_to_clone[h]
pi[h, dominant] = 1
cell_clone[i] = dominant_clones[z_i]
admixture_mass[h] = 0
mixture_active[h] = false
```

No random values are drawn during migration. Actual v0.2.2 interrupted chains
were validated to continue exactly to their uninterrupted v0.2.2 result.

## v0.3.x rejection

The 0.3.x line contains incompatible experimental hybrid-sampler state and is
not an ancestor of 0.4.0. Such checkpoints are decoded only far enough to issue
a lineage-specific `ValueError`; they are not migrated.
