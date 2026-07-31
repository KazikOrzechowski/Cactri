# NumPy/Numba algorithmic identity

## Contract

Cactri targets algorithmic identity between NumPy and Numba backends:

- the model's NumPy `Generator` is the only source of randomness;
- random uniforms and other variates are generated before backend dispatch;
- kernels do not own hidden RNG state;
- count and discrete-state kernels are expected to agree exactly;
- floating-point likelihood kernels use matching operation order where practical;
- complete seeded chains are tested for exact assignments, clone mappings,
  genotype states, and observation parameters.

## Stage-2 global moves

The restricted-Gibbs split/merge transition is implemented in shared code. It
uses the same model RNG stream regardless of backend. The local approximate or
sequential sweep, accepted partition, conditional BCR-profile draw, clone-label
draw, genotype update, and observation update therefore remain exactly aligned
between NumPy and Numba for a fixed environment and seed.

## Limits

Bitwise portability across different NumPy, Numba, compiler, CPU, or BLAS
versions is not promised. The supported guarantee is backend identity within a
fixed software and hardware environment. Distributional tests remain necessary
for portability across environments.
