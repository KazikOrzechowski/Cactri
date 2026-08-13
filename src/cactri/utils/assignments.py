from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np

InitSpec: TypeAlias = str | Sequence[int] | np.ndarray


def reindex_assignments(assignments: np.ndarray) -> np.ndarray:
    """Map arbitrary nonnegative labels to contiguous labels in sorted order."""
    z = np.asarray(assignments, dtype=np.int64)
    if z.ndim != 1:
        raise ValueError("assignments must be one-dimensional.")
    _, inverse = np.unique(z, return_inverse=True)
    return inverse.astype(np.int64, copy=False)


def validate_assignment_vector(assignments: object, n_cells: int) -> np.ndarray:
    """Validate and normalize a cell-to-hypercluster assignment vector."""
    raw = np.asarray(assignments)
    if raw.ndim != 1:
        raise ValueError("init assignment vector must be one-dimensional.")
    if raw.shape[0] != n_cells:
        raise ValueError(
            f"init assignment vector must have length {n_cells}, got {raw.shape[0]}."
        )
    if raw.size == 0:
        raise ValueError("init assignment vector cannot be empty.")
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError("init assignment labels must be numeric integers.")
    numeric = raw.astype(float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("init assignment labels must be finite.")
    if not np.all(numeric == np.floor(numeric)):
        raise ValueError("init assignment labels must be integers.")
    if np.any(numeric < 0):
        raise ValueError("init assignment labels must be nonnegative.")
    return reindex_assignments(numeric.astype(np.int64))


def initialize_assignments(
    init: InitSpec,
    *,
    n_cells: int,
    sequences: np.ndarray,
    rng: np.random.Generator,
    random_init_clusters: int,
) -> np.ndarray:
    """Create initial assignments from a named mode or explicit vector."""
    if not isinstance(init, str):
        return validate_assignment_vector(init, n_cells)
    if init == "one_cluster":
        return np.zeros(n_cells, dtype=np.int64)
    if init == "random":
        k = min(max(1, int(random_init_clusters)), n_cells)
        return reindex_assignments(rng.integers(0, k, size=n_cells, dtype=np.int64))
    if init == "identical_sequences":
        _, z = np.unique(sequences, axis=0, return_inverse=True)
        return z.astype(np.int64)
    raise ValueError(
        "init must be 'one_cluster', 'random', 'identical_sequences', "
        "'bcr_consensus', or a one-dimensional assignment vector."
    )
