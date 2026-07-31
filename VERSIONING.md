# Versioning and release policy

Cactri follows semantic versioning.

- **Patch release (`0.2.x`)**: bug fixes, documentation, and optimizations that
  do not intentionally change the target statistical model.
- **Minor release (`0.x.0`)**: new samplers, changed defaults, or additive model
  capabilities. Migration notes and characterization tests are required.
- **Major release (`x.0.0`)**: removal of deprecated public APIs or incompatible
  result-schema changes.

A release is immutable after publication. Development for the next release is
performed in a copied source tree, and every release must include:

1. a changelog entry;
2. migration notes for changed defaults or semantics;
3. unit and backend-identity tests;
4. a clean-install wheel smoke test;
5. a machine-readable validation report.
