# Stage 2 roadmap

## Completed in 0.2.0

- collapsed mutation-informed restricted-Gibbs split/merge proposals;
- blocked partition changes with clone labels marginalized and then resampled;
- standardized absent-genotype observation prior;
- coherent trace-based posterior genotype summaries;
- label-invariant hypercluster co-assignment summaries;
- split/merge diagnostics;
- exact seeded NumPy/Numba identity tests with global moves enabled.

## Completed in 0.2.1

- cached sufficient statistics for repeated global proposals;
- adaptive split-versus-merge anchor scheduling with MH correction;
- exact checkpoint/resume, including backend switching;
- posterior hypercluster and clone partition medoids;
- fit-call-independent tracking cadence;
- cache and adaptive-scheduler diagnostics.

## Planned for later 0.2.x releases

- multiple restricted-Gibbs refinement scans with an auxiliary-path MH correction;
- incremental cache updates after accepted moves instead of complete cache rebuilds;
- optional disk-backed trace storage for long chains;
- checkpoint schema migration utilities;
- posterior credible sets for partitions and genotype probabilities.

## Candidate 0.3.x model extensions

- topology or active-leaf uncertainty;
- integrated clone-collapse/split moves for incomplete trees;
- deletion and hybrid tree/independent genotype states;
- mutation-specific supplied-tree confidence.
