# Cactri

Cactri is a research package for joint BCR hypercluster inference and clonal
mutation-genotype inference from single-cell sequence and read-count data.

The public models are:

- `CactriTree`: mutation origins on a fixed full binary tree, optionally using
  a probabilistic supplied mutation-tree estimate and a learned edge-error rate.
- `CactriOmega`: independent clone-by-mutation genotypes, optionally centered
  on a supplied Omega matrix through a learned relaxation rate.
- `BCRInitializer`: standalone BCR-only CRP/Dirichlet initialization and
  multi-chain consensus fitting.

Version 0.4.0 adds an opt-in dominant-clone sparse-admixture model. One BCR
hypercluster may contain cells from several mutation clones without creating a
second BCR component.

## Installation

```bash
pip install .
```

Optional Numba acceleration:

```bash
pip install ".[numba]"
```

## Public imports

```python
from cactri import (
    BCRInitializer,
    Cactri,
    CactriOmega,
    CactriTree,
    CloneMixtureConfig,
    SplitMergeConfig,
    TrackingConfig,
)
```

## Dominant-clone sparse admixture

The extension is opt-in. Existing scripts retain the exact v0.2.2
one-clone-per-hypercluster transition when `clone_mixture` is omitted.

```python
from cactri import CactriTree, CloneMixtureConfig, TrackingConfig

mixture = CloneMixtureConfig(
    enabled=True,
    admixture_mass_prior=(1.0, 19.0),
    residual_concentration=0.2,
    residual_base="uniform",
    allow_pure_hyperclusters=True,
    mixture_presence_prior=(1.0, 9.0),
)

model = CactriTree(
    n_levels=3,
    observed_edge_probabilities=edge_posterior,
    tree_prior="level_inverse",
    learn_edge_error_rate=True,
    clone_mixture=mixture,
    random_state=2,
)
model.prefit(
    bcr_sequences,
    alt_counts,
    total_counts,
    init="bcr_consensus",
)
model.fit(
    n_iter=1_000,
    assignment_sampler="approximate",
    tracking=TrackingConfig.posterior(
        every=5,
        mixture_diagnostics=True,
    ),
)
```

For cell `i`, `z_i` remains its BCR hypercluster and `y_i` is its cell-level
mutation clone. Hypercluster `h` has clone proportions `pi_h`, shrunk toward one
dominant clone. Pure hyperclusters use an exact one-hot row; impossible clone
scores are represented as `-inf`, not by evaluating `log(0)`.

Canonical mixture state:

```python
model.dominant_clones_
model.hypercluster_clone_proportions_
model.cell_clone_assignments_
model.admixture_mass_
model.residual_clone_proportions_
model.mixture_active_
```

`hypercluster_to_clone_` and `hypercluster_to_clone` remain compatibility aliases
for `dominant_clones_`. `current_cell_clone_assignment()` returns the sampled
cell-level labels when mixtures are enabled.

## Posterior summaries

```python
summary = model.posterior_summary(burn_in=0.5)

cell_clone_probability = summary["cell_clone_probabilities"]
cell_genotype_probability = summary["cell_genotype_probabilities"]
cell_clone_coclustering = summary["cell_clone_coassignment"]
hypercluster_clone_proportions = summary["hypercluster_clone_proportions"]
dominant_clone_probability = summary["dominant_clone_probabilities"]
admixture_probability = summary["admixture_probabilities"]
```

Dedicated methods include:

```python
model.posterior_cell_clone_probabilities()
model.posterior_cell_genotype_probabilities()
model.posterior_cell_clone_coassignment()
model.posterior_cell_coclustering_probabilities()
model.posterior_hypercluster_clone_proportions()
model.posterior_dominant_clone_probabilities()
model.posterior_admixture_probabilities()
model.posterior_effective_clone_counts()
```

The cell-genotype summary is coherent: every retained cell-clone draw is
combined with the genotype state from the same iteration before averaging.
Tracked hypercluster-level quantities are aligned to the current partition by
maximum cell overlap before averaging, making summaries invariant to numeric
hypercluster label permutations.

## Assignment samplers

- `approximate`: vectorized approximate CRP sweep; default.
- `sequential`: cell-by-cell reassignment.
- `split_merge`: retained v0.2 collapsed restricted-Gibbs transition.

The mixture extension supports `approximate` and `sequential`. The v0.2
split/merge target integrates one clone label per hypercluster and is therefore
explicitly unavailable when `clone_mixture.enabled=True`.

## CRP concentration

Mixture-enabled models sample the CRP concentration by default with the
Escobar-West auxiliary update and `Gamma(1,1)` shape-rate prior. Legacy mode
leaves `alpha` fixed by default, preserving v0.2.2 behavior.

```python
model = CactriTree(
    n_levels=3,
    clone_mixture=mixture,
    alpha_prior=(1.0, 1.0),
    sample_alpha=True,
)
```

## Tree-aware residual admixture

`CactriTree` optionally weights non-dominant clones by tree distance:

```python
CloneMixtureConfig(
    enabled=True,
    residual_base="tree_distance",
    tree_distance_decay=1.0,
)
```

The default remains `uniform`, matching the existing simulation mechanism.
`CactriOmega` rejects `tree_distance` because it has no clone tree.

## Checkpoint and resume

```python
model.save_checkpoint("chain.cactri.gz")
resumed = CactriTree.load_checkpoint("chain.cactri.gz", accelerator="numba")
```

Trusted v0.1.x and v0.2.x checkpoints migrate to an exact pure-mixture state:
old hypercluster clone labels become dominant clones, clone proportions become
one-hot, and cell clone labels are inherited from their hyperclusters. The RNG
state and continuation remain exact.

The v0.3.x experimental sampler branch was discontinued after simulation
benchmarking and its checkpoints are intentionally rejected by the v0.4
lineage with a clear error. Checkpoints use pickle internally and must only be
loaded from trusted sources.

## NumPy/Numba identity

Cactri owns all random-number generation. Accelerator kernels receive
pre-generated random values and maintain no hidden RNG state. Exact seeded
NumPy/Numba identity is tested for Tree and Omega mixture chains, posterior
tracking, and checkpoint continuation.

## Release lineage

```text
0.2.2 ───────────────► 0.4.0
   \
    └──► 0.3.0 ─► 0.3.1  [experimental, discontinued]
```

Version 0.4.0 is based directly on 0.2.2. The 0.3.x artifacts remain available
only for reproducibility.

## License

GNU General Public License v3.0 only.
