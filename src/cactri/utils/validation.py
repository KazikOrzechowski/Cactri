from __future__ import annotations

import numpy as np


def coerce_sequences(x: np.ndarray) -> np.ndarray:
    """Return integer encoded BCR sequences with shape cells x positions."""
    arr = np.asarray(x)
    if arr.ndim == 3:
        if arr.shape[-1] != 4:
            raise ValueError("one-hot BCR sequences must have final dimension 4.")
        arr = np.argmax(arr, axis=-1)
    if arr.ndim != 2:
        raise ValueError("bcr_sequences must be a 2D integer matrix or 3D one-hot array.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("bcr_sequences cannot be empty.")
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.isfinite(arr)) or not np.all(arr == np.floor(arr)):
            raise ValueError("BCR sequence values must be integer encoded.")
    arr = arr.astype(np.int64, copy=False)
    if np.any((arr < 0) | (arr > 3)):
        raise ValueError("BCR sequence values must be in {0, 1, 2, 3}.")
    return arr


def validate_read_counts(
    alt_counts: np.ndarray,
    total_counts: np.ndarray,
    *,
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    alt = np.asarray(alt_counts, dtype=np.int64)
    total = np.asarray(total_counts, dtype=np.int64)
    if alt.shape != total.shape or alt.ndim != 2:
        raise ValueError("alt_counts and total_counts must have the same 2D shape.")
    if alt.shape[0] != n_cells:
        raise ValueError("BCR and mutation matrices must have the same number of cells.")
    if np.any(alt < 0) or np.any(total < 0):
        raise ValueError("counts must be nonnegative.")
    if np.any(alt > total):
        raise ValueError("alt_counts cannot exceed total_counts.")
    return alt, total


def normalize_dirichlet_prior(prior: float | np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(prior, dtype=float)
    if arr.ndim == 0:
        out = np.full((length, 4), float(arr), dtype=float)
    elif arr.shape == (4,):
        out = np.broadcast_to(arr[None, :], (length, 4)).copy()
    elif arr.shape == (length, 4):
        out = arr.copy()
    else:
        raise ValueError("dirichlet_prior must be scalar, shape (4,), or shape (L, 4).")
    if np.any(out <= 0):
        raise ValueError("dirichlet_prior entries must be positive.")
    return out


def normalize_beta_prior(
    prior: tuple[float, float] | np.ndarray,
    n_snv: int,
    *,
    name: str,
) -> np.ndarray:
    arr = np.asarray(prior, dtype=float)
    if arr.shape == (2,):
        out = np.broadcast_to(arr[None, :], (n_snv, 2)).copy()
    elif arr.shape == (n_snv, 2):
        out = arr.copy()
    else:
        raise ValueError(f"{name} must have shape (2,) or ({n_snv}, 2).")
    if np.any(out <= 0):
        raise ValueError(f"{name} entries must be positive.")
    return out
