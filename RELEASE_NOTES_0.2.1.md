# Cactri 0.2.1 release notes

Cactri 0.2.1 is a backward-compatible Stage-2 maintenance release.

## Highlights

- Reuses split/merge sufficient statistics across repeated global proposals.
- Adds opt-in adaptive split-versus-merge anchor scheduling with a complete MH
  correction for state-dependent anchor probabilities.
- Adds exact checkpoint/resume, including optional NumPy/Numba backend switching.
- Adds label-invariant posterior medoids for hypercluster and clone partitions.
- Preserves tracking cadence across repeated or resumed `fit()` calls.

## Compatibility

The default anchor strategy remains `uniform_pair`, matching Cactri 0.2.0.
Caching is enabled by default but does not change the seeded chain. No public
method or result alias from 0.2.0 is removed.

## Validation

- 37 tests passed.
- Cached and uncached global proposals produced identical seeded chains.
- Adaptive NumPy and Numba chains were exactly identical for both model classes.
- Interrupted and resumed chains were exactly identical to uninterrupted chains.
- NumPy checkpoints resumed under Numba with exact continuation.
- A clean installation of the wheel completed successfully.
