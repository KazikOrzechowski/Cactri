from __future__ import annotations

from typing import Literal

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - depends on optional dependency
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn

        return decorator


Backend = Literal["auto", "numpy", "numba"]


def _sample_rows_numpy(log_probs: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    n_rows, n_cols = log_probs.shape
    out = np.empty(n_rows, dtype=np.int64)
    for i in range(n_rows):
        max_lp = -1.0e300
        for j in range(n_cols):
            value = float(log_probs[i, j])
            if value > max_lp:
                max_lp = value
        total = 0.0
        for j in range(n_cols):
            total += float(np.exp(log_probs[i, j] - max_lp))
        threshold = float(uniforms[i]) * total
        running = 0.0
        chosen = n_cols - 1
        for j in range(n_cols):
            running += float(np.exp(log_probs[i, j] - max_lp))
            if running >= threshold:
                chosen = j
                break
        out[i] = chosen
    return out


@njit(cache=True)
def _sample_rows_numba(log_probs: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    n_rows, n_cols = log_probs.shape
    out = np.empty(n_rows, dtype=np.int64)
    for i in range(n_rows):
        max_lp = -1.0e300
        for j in range(n_cols):
            value = log_probs[i, j]
            if value > max_lp:
                max_lp = value
        total = 0.0
        for j in range(n_cols):
            total += np.exp(log_probs[i, j] - max_lp)
        threshold = uniforms[i] * total
        running = 0.0
        chosen = n_cols - 1
        for j in range(n_cols):
            running += np.exp(log_probs[i, j] - max_lp)
            if running >= threshold:
                chosen = j
                break
        out[i] = chosen
    return out


def _bcr_loglik_numpy(seqs: np.ndarray, log_profiles: np.ndarray) -> np.ndarray:
    n_cells, length = seqs.shape
    n_h = log_profiles.shape[0]
    out = np.zeros((n_cells, n_h), dtype=np.float64)
    for i in range(n_cells):
        for h in range(n_h):
            value = 0.0
            for pos in range(length):
                value += float(log_profiles[h, pos, seqs[i, pos]])
            out[i, h] = value
    return out


@njit(cache=True)
def _bcr_loglik_numba(seqs: np.ndarray, log_profiles: np.ndarray) -> np.ndarray:
    n_cells, length = seqs.shape
    n_h = log_profiles.shape[0]
    out = np.zeros((n_cells, n_h), dtype=np.float64)
    for i in range(n_cells):
        for h in range(n_h):
            value = 0.0
            for pos in range(length):
                value += log_profiles[h, pos, seqs[i, pos]]
            out[i, h] = value
    return out


def _aggregate_numpy(
    labels: np.ndarray,
    alt_counts: np.ndarray,
    total_counts: np.ndarray,
    n_labels: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_cells, n_snv = alt_counts.shape
    alt_out = np.zeros((n_labels, n_snv), dtype=np.float64)
    ref_out = np.zeros((n_labels, n_snv), dtype=np.float64)
    for i in range(n_cells):
        label = int(labels[i])
        for j in range(n_snv):
            alt = float(alt_counts[i, j])
            alt_out[label, j] += alt
            ref_out[label, j] += float(total_counts[i, j]) - alt
    return alt_out, ref_out


@njit(cache=True)
def _aggregate_numba(
    labels: np.ndarray,
    alt_counts: np.ndarray,
    total_counts: np.ndarray,
    n_labels: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_cells, n_snv = alt_counts.shape
    alt_out = np.zeros((n_labels, n_snv), dtype=np.float64)
    ref_out = np.zeros((n_labels, n_snv), dtype=np.float64)
    for i in range(n_cells):
        label = labels[i]
        for j in range(n_snv):
            alt = alt_counts[i, j]
            alt_out[label, j] += alt
            ref_out[label, j] += total_counts[i, j] - alt
    return alt_out, ref_out


def _sample_clone_labels_numpy(
    alt_by_h: np.ndarray,
    ref_by_h: np.ndarray,
    genotype: np.ndarray,
    log_p_obs: np.ndarray,
    log_1m_obs: np.ndarray,
    log_p_unobs: float,
    log_1m_unobs: float,
    log_clone_prior: np.ndarray,
    uniforms: np.ndarray,
) -> np.ndarray:
    n_h, n_snv = alt_by_h.shape
    n_clones = genotype.shape[0]
    log_probs = np.empty((n_h, n_clones), dtype=float)
    for h in range(n_h):
        for c in range(n_clones):
            value = float(log_clone_prior[c])
            for j in range(n_snv):
                if genotype[c, j] != 0:
                    value += alt_by_h[h, j] * log_p_obs[j]
                    value += ref_by_h[h, j] * log_1m_obs[j]
                else:
                    value += alt_by_h[h, j] * log_p_unobs
                    value += ref_by_h[h, j] * log_1m_unobs
            log_probs[h, c] = value
    return _sample_rows_numpy(log_probs, uniforms)


@njit(cache=True)
def _sample_clone_labels_numba(
    alt_by_h: np.ndarray,
    ref_by_h: np.ndarray,
    genotype: np.ndarray,
    log_p_obs: np.ndarray,
    log_1m_obs: np.ndarray,
    log_p_unobs: float,
    log_1m_unobs: float,
    log_clone_prior: np.ndarray,
    uniforms: np.ndarray,
) -> np.ndarray:
    n_h, n_snv = alt_by_h.shape
    n_clones = genotype.shape[0]
    log_probs = np.empty((n_h, n_clones), dtype=np.float64)
    for h in range(n_h):
        for c in range(n_clones):
            value = log_clone_prior[c]
            for j in range(n_snv):
                if genotype[c, j] != 0:
                    value += alt_by_h[h, j] * log_p_obs[j]
                    value += ref_by_h[h, j] * log_1m_obs[j]
                else:
                    value += alt_by_h[h, j] * log_p_unobs
                    value += ref_by_h[h, j] * log_1m_unobs
            log_probs[h, c] = value
    return _sample_rows_numba(log_probs, uniforms)


def _sample_genotypes_numpy(
    alt_by_clone: np.ndarray,
    ref_by_clone: np.ndarray,
    log_p_obs: np.ndarray,
    log_1m_obs: np.ndarray,
    log_p_unobs: float,
    log_1m_unobs: float,
    log_prior_1: np.ndarray,
    log_prior_0: np.ndarray,
    uniforms: np.ndarray,
    fix_reference_clone: bool,
) -> np.ndarray:
    n_clones, n_snv = alt_by_clone.shape
    out = np.empty((n_clones, n_snv), dtype=np.int8)
    for c in range(n_clones):
        for j in range(n_snv):
            log1 = log_prior_1[c, j]
            log1 += alt_by_clone[c, j] * log_p_obs[j]
            log1 += ref_by_clone[c, j] * log_1m_obs[j]
            log0 = log_prior_0[c, j]
            log0 += alt_by_clone[c, j] * log_p_unobs
            log0 += ref_by_clone[c, j] * log_1m_unobs
            delta = log0 - log1
            if delta > 700.0:
                p1 = 0.0
            elif delta < -700.0:
                p1 = 1.0
            else:
                p1 = 1.0 / (1.0 + float(np.exp(delta)))
            out[c, j] = 1 if uniforms[c, j] < p1 else 0
    if fix_reference_clone:
        out[0, :] = 0
    return out


@njit(cache=True)
def _sample_genotypes_numba(
    alt_by_clone: np.ndarray,
    ref_by_clone: np.ndarray,
    log_p_obs: np.ndarray,
    log_1m_obs: np.ndarray,
    log_p_unobs: float,
    log_1m_unobs: float,
    log_prior_1: np.ndarray,
    log_prior_0: np.ndarray,
    uniforms: np.ndarray,
    fix_reference_clone: bool,
) -> np.ndarray:
    n_clones, n_snv = alt_by_clone.shape
    out = np.empty((n_clones, n_snv), dtype=np.int8)
    for c in range(n_clones):
        for j in range(n_snv):
            log1 = log_prior_1[c, j]
            log1 += alt_by_clone[c, j] * log_p_obs[j]
            log1 += ref_by_clone[c, j] * log_1m_obs[j]
            log0 = log_prior_0[c, j]
            log0 += alt_by_clone[c, j] * log_p_unobs
            log0 += ref_by_clone[c, j] * log_1m_unobs
            delta = log0 - log1
            if delta > 700.0:
                p1 = 0.0
            elif delta < -700.0:
                p1 = 1.0
            else:
                p1 = 1.0 / (1.0 + np.exp(delta))
            out[c, j] = 1 if uniforms[c, j] < p1 else 0
    if fix_reference_clone:
        for j in range(n_snv):
            out[0, j] = 0
    return out


class Accelerator:
    """Dispatch deterministic kernels to NumPy or optional Numba.

    All random values are generated by the model and passed into these methods.
    Numba kernels never own RNG state, which provides algorithmic identity for
    categorical and Bernoulli decisions whenever floating-point scores agree.
    """

    def __init__(self, backend: Backend = "auto", *, deterministic: bool = False) -> None:
        if backend not in {"auto", "numpy", "numba"}:
            raise ValueError("accelerator must be 'auto', 'numpy', or 'numba'.")
        if backend == "numba" and not NUMBA_AVAILABLE:
            raise ImportError("Numba backend requested but numba is not installed.")
        self.backend: Literal["numpy", "numba"] = (
            "numba" if backend == "numba" or (backend == "auto" and NUMBA_AVAILABLE) else "numpy"
        )
        self.deterministic = bool(deterministic)

    @property
    def enabled(self) -> bool:
        return self.backend == "numba"

    def sample_rows(self, log_probs: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
        log_probs = np.asarray(log_probs, dtype=np.float64)
        uniforms = np.asarray(uniforms, dtype=np.float64)
        fn = _sample_rows_numba if self.enabled else _sample_rows_numpy
        return fn(log_probs, uniforms)

    def bcr_loglik(self, seqs: np.ndarray, log_profiles: np.ndarray) -> np.ndarray:
        fn = _bcr_loglik_numba if self.enabled else _bcr_loglik_numpy
        return fn(np.asarray(seqs, dtype=np.int64), np.asarray(log_profiles, dtype=np.float64))

    def aggregate_counts(
        self,
        labels: np.ndarray,
        alt_counts: np.ndarray,
        total_counts: np.ndarray,
        n_labels: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        fn = _aggregate_numba if self.enabled else _aggregate_numpy
        return fn(
            np.asarray(labels, dtype=np.int64),
            np.asarray(alt_counts, dtype=np.float64),
            np.asarray(total_counts, dtype=np.float64),
            int(n_labels),
        )

    def sample_clone_labels(
        self,
        alt_by_h: np.ndarray,
        ref_by_h: np.ndarray,
        genotype: np.ndarray,
        log_p_obs: np.ndarray,
        log_1m_obs: np.ndarray,
        log_p_unobs: float,
        log_1m_unobs: float,
        log_clone_prior: np.ndarray,
        uniforms: np.ndarray,
    ) -> np.ndarray:
        fn = _sample_clone_labels_numba if self.enabled else _sample_clone_labels_numpy
        return fn(
            np.asarray(alt_by_h, dtype=np.float64),
            np.asarray(ref_by_h, dtype=np.float64),
            np.asarray(genotype, dtype=np.int8),
            np.asarray(log_p_obs, dtype=np.float64),
            np.asarray(log_1m_obs, dtype=np.float64),
            float(log_p_unobs),
            float(log_1m_unobs),
            np.asarray(log_clone_prior, dtype=np.float64),
            np.asarray(uniforms, dtype=np.float64),
        )

    def sample_genotypes(
        self,
        alt_by_clone: np.ndarray,
        ref_by_clone: np.ndarray,
        log_p_obs: np.ndarray,
        log_1m_obs: np.ndarray,
        log_p_unobs: float,
        log_1m_unobs: float,
        log_prior_1: np.ndarray,
        log_prior_0: np.ndarray,
        uniforms: np.ndarray,
        fix_reference_clone: bool,
    ) -> np.ndarray:
        fn = _sample_genotypes_numba if self.enabled else _sample_genotypes_numpy
        return fn(
            np.asarray(alt_by_clone, dtype=np.float64),
            np.asarray(ref_by_clone, dtype=np.float64),
            np.asarray(log_p_obs, dtype=np.float64),
            np.asarray(log_1m_obs, dtype=np.float64),
            float(log_p_unobs),
            float(log_1m_unobs),
            np.asarray(log_prior_1, dtype=np.float64),
            np.asarray(log_prior_0, dtype=np.float64),
            np.asarray(uniforms, dtype=np.float64),
            bool(fix_reference_clone),
        )


__all__ = ["Accelerator", "Backend", "NUMBA_AVAILABLE"]
