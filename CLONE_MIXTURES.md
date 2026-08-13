# Dominant-clone sparse-admixture model

## Motivation

The v0.2 model assigns one mutation clone to each BCR hypercluster. Simulation
experiments with cell-level clone misspecification showed that splitting a
mutation-discordant subset into a new BCR hypercluster pays a large and usually
prohibitive BCR marginal-likelihood penalty. Version 0.4 changes the generative
model instead: BCR identity remains hypercluster-level, while mutation clone
identity becomes cell-level.

## Generative model

For cell `i`, let `z_i` be its BCR hypercluster and `y_i` its mutation clone.
Each hypercluster has one BCR profile `B_h` and one clone-mixture vector `pi_h`:

```text
z ~ CRP(alpha)
B_h,l ~ Dirichlet(eta_l)
x_i | z_i=h ~ product_l Categorical(B_h,l)
y_i | z_i=h, pi_h ~ Categorical(pi_h)
a_i,j | y_i, G, theta ~ Binomial(d_i,j, theta_i,j*)
```

The mutation observation probability is `p_obs_by_mutation_[j]` when clone
`y_i` carries mutation `j`, and `p_unobs_` otherwise.

## Dominant-plus-residual prior

Each hypercluster has dominant clone `d_h`:

```text
d_h ~ Categorical(clone_prior)
```

When the hypercluster is admixed:

```text
epsilon_h ~ Beta(a_mix, b_mix)
q_h,-d ~ Dirichlet(gamma * w_d)
pi_h,d = 1 - epsilon_h
pi_h,k = epsilon_h * q_h,k, k != d
```

The defaults are `Beta(1,19)` for total admixture and total residual Dirichlet
concentration `gamma=0.2`. Because `gamma < 1`, residual mass is sparse and
usually concentrates on one or a few secondary clones.

With `allow_pure_hyperclusters=True`, a global admixture prevalence is added:

```text
rho_mix ~ Beta(1,9)
s_h ~ Bernoulli(rho_mix)
```

If `s_h=0`, `pi_h` is exactly one-hot at `d_h`. If `s_h=1`, the
Beta-Dirichlet model is used.

## Exact-zero safety

Pure rows are stored as exact zeros and one exact one. Categorical log scores
are constructed with a masked transform:

```text
log(pi_h,k) = -inf  when pi_h,k = 0
```

No code evaluates `log(0)`, and zero components are excluded from entropy and
prior-density products. A pure cluster can become admixed because the
pure/admixed structure is updated before the next cell-clone draw; once active,
all residual components receive positive probability.

## Gibbs updates

### Cell clones

```text
log p(y_i=k | ...) = log pi_z_i,k + mutation_loglikelihood_i,k + constant
```

Cells are conditionally independent and sampled in parallel.

### Hypercluster assignments

Mutation evidence is marginalized over the destination hypercluster mixture:

```text
mutation_score_i,h = logsumexp_k(
    log pi_h,k + mutation_loglikelihood_i,k
)
```

This lets a mutation-discordant cell remain in its BCR hypercluster.

### Pure/admixed structure and dominant clone

For every candidate dominant clone, the admixture mass and residual proportions
are integrated out using Beta-binomial and Dirichlet-multinomial terms. Pure and
admixed candidate states are sampled jointly. A pure state is valid only when
all current cell-clone labels in that hypercluster equal its candidate dominant
clone.

### Mixture parameters

Conditional on an active state:

```text
epsilon_h | ... ~ Beta(a_mix + n_residual, b_mix + n_dominant)
q_h,-d | ... ~ Dirichlet(gamma * w_d + n_h,-d)
```

### Genotypes and observation probabilities

Tree edges or Omega genotypes, `p_obs_by_mutation_`, and `p_unobs_` aggregate
mutation reads by sampled cell-level clone labels `y_i`.

### CRP concentration

When enabled, `alpha` uses the Escobar-West auxiliary-variable Gamma-mixture
update. It defaults on for the mixture model and off for legacy mode.

## Residual bases

- `uniform`: equal prior mass over every non-dominant clone; default.
- `tree_distance`: `CactriTree` weights clones by `exp(-decay * tree_distance)`.
  The reference clone is treated as one level above the mutation-tree root for
  this prior. `CactriOmega` has no tree and rejects this option.

## Posterior interpretation

Hypercluster labels are not identifiable across draws. Hypercluster-level trace
summaries are aligned to the current partition by maximum cell overlap before
averaging. Cell clone co-clustering is label-invariant and is computed directly
from retained cell clone assignments.
