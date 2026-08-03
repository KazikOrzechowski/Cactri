from __future__ import annotations

import numpy as np


def _trace_indices(n_draws: int, *, burn_in: int | float = 0, thin: int = 1) -> np.ndarray:
    if n_draws < 1:
        raise ValueError("at least one posterior draw is required.")
    if thin < 1:
        raise ValueError("thin must be at least 1.")
    if isinstance(burn_in, (float, np.floating)):
        if not 0.0 <= float(burn_in) < 1.0:
            raise ValueError("fractional burn_in must lie in [0, 1).")
        start = int(np.floor(float(burn_in) * n_draws))
    else:
        start = int(burn_in)
        if start < 0 or start >= n_draws:
            raise ValueError("integer burn_in must lie in [0, n_draws).")
    return np.arange(start, n_draws, thin, dtype=np.int64)


def coherent_cell_genotype_probabilities(
    cell_clone_trace: np.ndarray,
    genotype_trace: np.ndarray,
    *,
    burn_in: int | float = 0,
    thin: int = 1,
) -> np.ndarray:
    """Average cell genotypes after combining clone labels and profiles per draw."""

    clones = np.asarray(cell_clone_trace, dtype=np.int64)
    genotype = np.asarray(genotype_trace)
    if clones.ndim != 2 or genotype.ndim != 3:
        raise ValueError("expected clones (draws,cells) and genotypes (draws,clones,snv).")
    if clones.shape[0] != genotype.shape[0]:
        raise ValueError("trace lengths do not match.")
    indices = _trace_indices(clones.shape[0], burn_in=burn_in, thin=thin)
    clones = clones[indices]
    genotype = genotype[indices]
    if np.any(clones < 0) or np.any(clones >= genotype.shape[1]):
        raise ValueError("cell clone trace contains labels outside the genotype trace.")
    draws = np.arange(clones.shape[0])[:, None]
    return genotype[draws, clones, :].mean(axis=0)


def cell_clone_probabilities_from_trace(
    cell_clone_trace: np.ndarray,
    *,
    n_clones: int,
    burn_in: int | float = 0,
    thin: int = 1,
) -> np.ndarray:
    """Return posterior clone probabilities from fixed-identity clone labels."""

    clones = np.asarray(cell_clone_trace, dtype=np.int64)
    if clones.ndim != 2:
        raise ValueError("cell_clone_trace must have shape (draws, cells).")
    if n_clones < 1 or np.any(clones < 0) or np.any(clones >= n_clones):
        raise ValueError("cell clone labels must lie in [0, n_clones).")
    indices = _trace_indices(clones.shape[0], burn_in=burn_in, thin=thin)
    selected = clones[indices]
    one_hot = np.eye(n_clones, dtype=float)[selected]
    return one_hot.mean(axis=0)


def coassignment_probabilities_from_trace(
    assignment_trace: np.ndarray | list[np.ndarray],
    *,
    burn_in: int | float = 0,
    thin: int = 1,
) -> np.ndarray:
    """Return a label-invariant posterior cell co-assignment matrix."""

    draws = [np.asarray(row, dtype=np.int64) for row in assignment_trace]
    if not draws:
        raise ValueError("assignment_trace is empty.")
    n_cells = draws[0].size
    if any(row.ndim != 1 or row.size != n_cells for row in draws):
        raise ValueError("all assignment draws must be one-dimensional and equally sized.")
    indices = _trace_indices(len(draws), burn_in=burn_in, thin=thin)
    out = np.zeros((n_cells, n_cells), dtype=float)
    for idx in indices:
        row = draws[int(idx)]
        out += row[:, None] == row[None, :]
    return out / indices.size


def partition_medoid_from_trace(
    assignment_trace: np.ndarray | list[np.ndarray],
    *,
    burn_in: int | float = 0,
    thin: int = 1,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Return the retained partition minimizing squared co-assignment loss.

    The returned index refers to the original, unthinned trace. The loss is a
    label-invariant Binder-style squared loss against the posterior
    co-assignment matrix and is returned for every selected draw.
    """

    draws = [np.asarray(row, dtype=np.int64) for row in assignment_trace]
    if not draws:
        raise ValueError("assignment_trace is empty.")
    n_cells = draws[0].size
    if any(row.ndim != 1 or row.size != n_cells for row in draws):
        raise ValueError("all assignment draws must be one-dimensional and equally sized.")
    indices = _trace_indices(len(draws), burn_in=burn_in, thin=thin)
    target = coassignment_probabilities_from_trace(
        draws, burn_in=burn_in, thin=thin
    )
    losses = np.empty(indices.size, dtype=float)
    for out_index, trace_index in enumerate(indices):
        row = draws[int(trace_index)]
        matrix = row[:, None] == row[None, :]
        difference = matrix.astype(float) - target
        losses[out_index] = float(np.sum(difference * difference))
    selected = int(np.argmin(losses))
    trace_index = int(indices[selected])
    return draws[trace_index].copy(), trace_index, losses
