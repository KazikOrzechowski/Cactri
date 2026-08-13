from __future__ import annotations

import numpy as np


def coassignment_matrix(partitions: np.ndarray) -> np.ndarray:
    """Compute posterior pairwise co-assignment frequencies."""
    parts = np.asarray(partitions, dtype=np.int64)
    if parts.ndim != 2:
        raise ValueError("partitions must have shape draws x cells.")
    n_draws, n_cells = parts.shape
    if n_draws == 0:
        raise ValueError("at least one partition is required.")
    out = np.zeros((n_cells, n_cells), dtype=float)
    for z in parts:
        out += z[:, None] == z[None, :]
    return out / float(n_draws)


def partition_medoid(partitions: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    """Return the sampled partition closest to the posterior co-assignment matrix."""
    parts = np.asarray(partitions, dtype=np.int64)
    target = coassignment_matrix(parts)
    losses = np.empty(parts.shape[0], dtype=float)
    for i, z in enumerate(parts):
        matrix = z[:, None] == z[None, :]
        losses[i] = np.square(matrix.astype(float) - target).sum()
    index = int(np.argmin(losses))
    return parts[index].copy(), index, losses
