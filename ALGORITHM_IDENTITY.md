# Algorithmic identity

Cactri requires exact seeded NumPy/Numba identity.

## RNG ownership

All random values are generated through `model.rng`, a NumPy `Generator`.
Accelerator kernels receive explicit uniforms and never call a backend-specific
RNG.

## Mixture transitions

The following draws are backend-independent:

- cell-level clone categorical draws;
- joint dominant/pure-admixed structure draws;
- Beta admixture-mass draws;
- Gamma-normalized residual Dirichlet draws;
- global admixture-prevalence draws;
- CRP concentration auxiliary and Gamma draws.

The same Python-level code owns these draws for both backends. Numba is used for
shared BCR, grouped-count, genotype, and categorical kernels only.

## Exact-zero rows

Pure clone-mixture rows contain exact zeros. The score builder maps zeros to
`-inf` with a mask. It never evaluates `log(0)`. Entropy and prior calculations
operate only on positive entries. This policy is identical across backends.

## Characterization

Validation includes:

- exact full-chain Tree and Omega mixture identity;
- exact checkpoint-resume identity;
- bit-for-bit v0.2.2 behavior with mixtures disabled;
- exact v0.2.2 checkpoint continuation under v0.4.0.
