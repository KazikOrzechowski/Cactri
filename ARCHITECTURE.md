# Architecture

## Shared model

`Cactri` is abstract. It owns:

- BCR sequence coercion and Dirichlet profiles;
- CRP cell-to-hypercluster assignments;
- hypercluster-to-clone assignments;
- approximate, sequential, and collapsed restricted-Gibbs split/merge samplers;
- mutation read likelihoods through a common binary genotype-matrix interface;
- `p_obs_by_mutation` and `p_unobs` initialization and Gibbs updates;
- coherent trace-based posterior summaries;
- tracking, validation, and legacy result aliases.

Subclasses implement genotype state initialization, sampling, prior evaluation,
and optional prior-hyperparameter updates.

## Stage-2 split/merge transition

For a proposed partition, the transition analytically integrates out:

1. BCR frequency profiles under the Dirichlet-multinomial model;
2. each hypercluster's clone label under the clone prior and mutation-read
   likelihood.

A split proposal anchors two cells in separate candidate groups and assigns the
remaining cells with restricted Gibbs probabilities containing:

- the CRP group-size factor;
- the BCR posterior predictive probability;
- the clone-marginal mutation posterior predictive probability.

The reverse split probability is evaluated for merge proposals, yielding a
Metropolis-Hastings transition with an explicit proposal correction. After an
accepted move, explicit BCR profiles and clone labels are sampled from their
conditional distributions.

## Tree model

`CactriTree` represents each mutation by one full-binary-tree origin vertex. A
supplied mutation-edge assignment enables a learned distance-aware mismatch
rate by default. Uniform and level-inverse base priors remain available.

## Omega model

`CactriOmega` samples clone-by-mutation entries independently. A supplied
`omega_prior` enables a learned matrix mismatch rate. `fix_reference_clone`
retains the previous clone-0 behavior.

## Acceleration and algorithmic identity

`_numba_accelerator.py` contains all optional kernels. Random values are always
created by the model's NumPy `Generator` and passed to the backend. Kernels do
not call a random-number generator. The split/merge transition is implemented
in shared deterministic NumPy/Python code and therefore follows the same RNG
stream under both backends.

## Compatibility

Canonical model data are ordinary attributes. `state_` is a thin compatibility
view, and result dictionaries retain previous aliases such as
`mutation_profile`, `genotype_matrix`, and `cell_clone_assignments`.
