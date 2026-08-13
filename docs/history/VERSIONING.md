# Versioning and release policy

Cactri follows semantic versioning and immutable release artifacts.

## Lineage

```text
0.2.2 ───────────────► 0.4.0
   \
    └──► 0.3.0 ─► 0.3.1  [experimental, discontinued]
```

The 0.3 branch tested mutation-informed partition proposals. Simulation
benchmarking found no improvement and zero accepted split-off moves across the
main diagnostic. Future work therefore returned to 0.2.2 and changed the
within-hypercluster clone model instead.

## Release classes

- Patch: bug fixes and computational changes that preserve the statistical
  model.
- Minor: additive models, new samplers, or changed opt-in semantics.
- Major: removal of deprecated APIs or incompatible result schemas.

Every release includes a changelog, migration guide, backend-identity tests,
clean-install wheel test, validation report, source archive, and wheel.

## 0.4.0

Opt-in cell-level clone mixtures within BCR hyperclusters. Legacy behavior is
unchanged by default. v0.3 checkpoints are not part of this lineage.
