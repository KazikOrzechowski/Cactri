# Cactri 0.4.1

Patch release discovered during simulation testing.

- `fit()` no longer eagerly constructs dense cell-by-cell co-assignment matrices; call the posterior co-assignment methods explicitly.
- Active residual Dirichlet simplexes are kept strictly positive after floating-point underflow; exact zeros remain exclusive to structurally pure hyperclusters.
- 61 tests pass.
