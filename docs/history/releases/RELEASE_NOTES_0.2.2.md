# Cactri 0.2.2

Cactri 0.2.2 adds direct support for probabilistic mutation-tree initializer
outputs and an optional bounded split/merge proposal.

## Probabilistic supplied edge estimates

`CactriTree` now accepts a normalized or unnormalized mutation-by-edge matrix:

```python
model = CactriTree(
    n_levels=3,
    observed_edge_probabilities=edge_posterior,
    tree_prior="level_inverse",
    learn_edge_error_rate=True,
)
```

For initializer posterior row `P_j` and inverse-distance transition matrix `T`,
the mutation-origin prior is

```text
(1 - r) P_j + r (P_j @ T),
```

where `r` has the existing `edge_error_rate_prior`. A latent direct/smoothed
component indicator gives a conjugate Beta update for `r`. Random draws continue
to originate in the model's NumPy generator, preserving the package's
algorithmic-identity contract across NumPy and Numba backends.

Point estimates remain supported through `observed_edge_assignment`. The point
and probabilistic interfaces are mutually exclusive.

## Bounded split/merge proposals

`SplitMergeConfig` now accepts:

```python
SplitMergeConfig(
    max_restricted_cells=500,
    anchor_strategy="uniform_pair",
)
```

Only cell pairs whose split cluster or merged cluster union contains at most the
specified number of cells are eligible. Anchors are uniform over eligible cell
pairs, and the state-dependent eligible-pair probability is included in the
Metropolis-Hastings ratio. The option prevents a single global proposal from
performing an unexpectedly large restricted-Gibbs allocation.

The bounded option currently supports `anchor_strategy="uniform_pair"` only.
Leaving `max_restricted_cells=None` preserves 0.2.1 behavior.

## Compatibility

This release is backward compatible with 0.2.1. Existing point-edge, fixed
prior, uniform-prior, and level-inverse-prior models are unchanged by default.
