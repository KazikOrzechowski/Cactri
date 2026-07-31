# Architecture

## Shared model

`Cactri` is abstract. It owns:

- BCR sequence coercion and Dirichlet profiles;
- CRP cell-to-hypercluster assignments;
- hypercluster-to-clone assignments;
- approximate, sequential, and Stage-1 split/merge assignment samplers;
- mutation read likelihoods through a common binary genotype-matrix interface;
- `p_obs_by_mutation` and `p_unobs` initialization and Gibbs updates;
- tracking, posterior summaries, validation, and legacy result aliases.

Subclasses implement genotype state initialization, sampling, prior evaluation,
and optional prior-hyperparameter updates.

## Tree model

`CactriTree` represents each mutation by one full-binary-tree origin vertex. A
supplied mutation-edge assignment enables a learned distance-aware mismatch
rate by default. Uniform and level-inverse base priors remain available.

## Omega model

`CactriOmega` samples clone-by-mutation entries independently. A supplied
`omega_prior` enables a learned matrix mismatch rate. `fix_reference_clone`
retains the previous clone-0 behavior.

## Acceleration

`_numba_accelerator.py` contains all optional kernels. Random values are always
created by the model's NumPy `Generator` and passed to the backend. Kernels do
not call a random-number generator.

## Compatibility

Canonical model data are ordinary attributes. `state_` is a thin compatibility
view, and result dictionaries retain previous aliases such as
`mutation_profile`, `genotype_matrix`, and `cell_clone_assignments`.
