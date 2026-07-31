from __future__ import annotations

import numpy as np


def coherent_cell_genotype_probabilities(
    cell_clone_trace: np.ndarray,
    genotype_trace: np.ndarray,
) -> np.ndarray:
    """Average cell genotypes after combining clone labels and profiles per draw."""
    clones = np.asarray(cell_clone_trace, dtype=np.int64)
    genotype = np.asarray(genotype_trace)
    if clones.ndim != 2 or genotype.ndim != 3:
        raise ValueError("expected clones (draws,cells) and genotypes (draws,clones,snv).")
    if clones.shape[0] != genotype.shape[0]:
        raise ValueError("trace lengths do not match.")
    draws = np.arange(clones.shape[0])[:, None]
    return genotype[draws, clones, :].mean(axis=0)
