# Stage 2 roadmap

## Completed in 0.2.0

- collapsed mutation-informed restricted-Gibbs split/merge proposals;
- blocked partition changes with clone labels marginalized and then resampled;
- standardized absent-genotype observation prior;
- coherent trace-based posterior genotype summaries;
- label-invariant hypercluster co-assignment summaries;
- split/merge diagnostics;
- exact seeded NumPy/Numba identity tests with global moves enabled.

## Planned for later 0.2.x releases

- multiple restricted-Gibbs refinement scans with an auxiliary-path MH correction;
- adaptive scheduling of split versus merge anchors;
- cached sufficient statistics for repeated global proposals;
- posterior partition medoid convenience methods;
- checkpoint/resume support.

## Candidate 0.3.x model extensions

- topology or active-leaf uncertainty;
- integrated clone-collapse/split moves for incomplete trees;
- deletion and hybrid tree/independent genotype states;
- mutation-specific supplied-tree confidence.
