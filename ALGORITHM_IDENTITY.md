# NumPy/Numba algorithmic identity

Cactri targets algorithmic identity rather than universal bitwise identity.

1. The model creates every random variate with one NumPy `Generator`.
2. The exact same uniforms are passed to NumPy and Numba categorical or
   Bernoulli kernels.
3. Kernels use the same loop order for row sampling, count aggregation, clone
   assignment, and genotype sampling.
4. Exact equality is tested for integer outputs and grouped counts.
5. Floating-point likelihoods are tested with tight tolerances.
6. Full short seeded chains are required to produce the same assignments,
   clone labels, genotype state, and observation parameters on both backends.

Parallel reductions are intentionally not used in Stage 1 because their changed
summation order would weaken this guarantee. Later performance work must retain
these parity tests or document any deliberate relaxation.
