# Migration from legacy_PGM

| Previous class | Cactri class |
|---|---|
| robust tree fitter | `cactri.CactriTree` |
| independent/Omega fitter | `cactri.CactriOmega` |
| BCR-only fitter | `cactri.BCRInitializer` |

```python
from cactri import CactriTree

model = CactriTree(
    n_levels=3,
    observed_edge_assignment=input_edges,
    random_state=1,
)
model.prefit(sequences, alt_counts, total_counts, init=initial_partition)
result = model.fit(1_000)
```

Important Stage-1 changes:

- the base class is abstract;
- both genotype models use `p_obs_by_mutation` and `p_unobs` names;
- all random draws come from the model NumPy generator;
- Numba is selected with `accelerator="auto"`, `"numpy"`, or `"numba"`;
- `BCRInitializer` is standalone but can be invoked internally with
  `init="bcr_consensus"`;
- deletion and pruning models remain in the separate legacy archive.
