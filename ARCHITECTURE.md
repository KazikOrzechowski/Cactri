# Architecture

## Release lineage

Cactri 0.4.0 branches from 0.2.2. The 0.3.x hybrid/blocked partition samplers
are retained only as immutable historical artifacts.

## Public classes

- `Cactri`: abstract shared CRP/BCR, observation, clone-mixture, tracking, and
  checkpoint implementation.
- `CactriTree`: full-binary-tree mutation-origin genotype model.
- `CactriOmega`: independent clone-by-mutation genotype model.
- `BCRInitializer`: standalone BCR-only initializer and consensus fitter.

## Shared latent state

Legacy state:

- `assignments_`: cell to BCR hypercluster.
- `bcr_profiles_`: hypercluster by BCR-position categorical probabilities.
- `p_obs_by_mutation_` and `p_unobs_`.

Mixture state:

- `dominant_clones_` / `hypercluster_to_clone_`: dominant clone per
  hypercluster.
- `cell_clone_assignments_`: cell-level mutation clone labels.
- `hypercluster_clone_proportions_`: hypercluster by clone mixture matrix.
- `admixture_mass_`: total residual mass per hypercluster.
- `residual_clone_proportions_`: residual distribution conditional on not being
  the dominant clone.
- `mixture_active_`: pure-versus-admixed indicator.
- `mixture_presence_rate_`: global admixture prevalence.

When mixtures are disabled, these arrays are maintained as an exact one-hot
view of the v0.2 state without consuming random numbers.

## Transition order

Mixture-enabled iterations use:

1. approximate or sequential cell-to-hypercluster transition;
2. conjugate BCR-profile refresh performed by the assignment transition;
3. exact parallel cell-level clone sampling;
4. collapsed pure/admixed and dominant-clone sampling;
5. conjugate admixture-parameter sampling;
6. Tree edge or Omega genotype update from counts aggregated by cell clone;
7. genotype-prior hyperparameter update;
8. observation-probability update;
9. optional CRP concentration update.

Legacy mode retains the exact 0.2.2 transition order.

## Partial collapsing

The hypercluster assignment score marginalizes over the destination clone
mixture. Cell clone labels are then sampled conditionally. Dominant clone and
pure/admixed structure are sampled with admixture mass and residual proportions
integrated out. This avoids creating a new BCR component merely to explain
mutation-clone heterogeneity.

## Accelerators

NumPy owns the RNG. Numba kernels receive explicit uniforms and do not create
hidden random state. Mixture-specific categorical draws use the same shared row
sampler, preserving exact backend identity.

## Tracking

`TrackingConfig` controls every potentially large mixture trace independently.
Hypercluster-level posterior summaries align each draw to the current partition
by maximum cell overlap. Cell clone co-clustering is computed directly from
cell-level label traces and is invariant to clone-label permutations only when
the biological clone labels themselves are considered fixed; the coassignment
matrix is invariant to numeric relabeling.
