from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import Cactri


class CactriOmega(Cactri):
    """Cactri with independent clone-by-mutation Bernoulli genotypes."""

    def __init__(
        self,
        *,
        n_clones: int,
        genotype_prior: float | np.ndarray = 0.5,
        omega_prior: np.ndarray | None = None,
        relax_rate_prior: tuple[float, float] = (1.0, 19.0),
        relax_rate_init: float | None = None,
        sample_relax_rate: bool = True,
        fix_reference_clone: bool = True,
        **kwargs: Any,
    ) -> None:
        if len(relax_rate_prior) != 2 or min(relax_rate_prior) <= 0:
            raise ValueError("relax_rate_prior must contain two positive values.")
        if relax_rate_init is not None and not 0 < relax_rate_init < 1:
            raise ValueError("relax_rate_init must lie in (0,1).")
        self.genotype_prior = np.asarray(genotype_prior, dtype=float)
        if np.any((self.genotype_prior <= 0) | (self.genotype_prior >= 1)):
            raise ValueError("genotype_prior values must lie in (0,1).")
        self.omega_prior_input = None if omega_prior is None else np.asarray(omega_prior).copy()
        self.relax_rate_prior = tuple(float(x) for x in relax_rate_prior)
        self.relax_rate_init = relax_rate_init
        self.sample_relax_rate = bool(sample_relax_rate)
        self.fix_reference_clone = bool(fix_reference_clone)
        self.genotype_matrix_: np.ndarray | None = None
        self.mutation_profile_: np.ndarray | None = None
        self.omega_prior_: np.ndarray | None = None
        self.relax_rate_: float | None = None
        self.relax_rate_trace_: list[float] = []
        super().__init__(n_clones=n_clones, **kwargs)
        self.genotype_matrix_trace_ = self.genotype_state_trace_

    def _initialize_genotype_state(self) -> None:
        self.omega_prior_ = self._normalize_omega_prior() if self.omega_prior_input is not None else None
        if self.omega_prior_ is not None:
            if self.relax_rate_init is None:
                a, b = self.relax_rate_prior
                self.relax_rate_ = float(np.clip(self.rng.beta(a, b), self.eps, 1.0 - self.eps))
            else:
                self.relax_rate_ = float(self.relax_rate_init)
        else:
            self.relax_rate_ = None
        prior = self._current_genotype_prior()
        genotype = (self.rng.random(prior.shape) < prior).astype(np.int8)
        if self.fix_reference_clone:
            genotype[0, :] = 0
        self.genotype_matrix_ = genotype
        self.mutation_profile_ = self.genotype_matrix_

    def _sample_genotype_state(self) -> None:
        alt_by_clone, ref_by_clone = self._aggregate_mutation_counts_by_clone()
        prior = self._current_genotype_prior()
        p_obs = np.clip(self.p_obs_by_mutation_, self.eps, 1.0 - self.eps)
        p_unobs = float(np.clip(self.p_unobs_, self.eps, 1.0 - self.eps))
        self.genotype_matrix_ = self.accelerator.sample_genotypes(
            alt_by_clone,
            ref_by_clone,
            np.log(p_obs),
            np.log1p(-p_obs),
            np.log(p_unobs),
            np.log1p(-p_unobs),
            np.log(np.clip(prior, self.eps, 1.0 - self.eps)),
            np.log(np.clip(1.0 - prior, self.eps, 1.0 - self.eps)),
            self.rng.random((self.n_clones, self.n_snv)),
            self.fix_reference_clone,
        )
        self.mutation_profile_ = self.genotype_matrix_

    def _genotype_matrix(self) -> np.ndarray:
        if self.genotype_matrix_ is None:
            raise RuntimeError("Call prefit before accessing genotypes.")
        return self.genotype_matrix_

    def _normalize_omega_prior(self) -> np.ndarray:
        omega = np.asarray(self.omega_prior_input)
        if omega.shape == (self.n_clones, self.n_snv):
            out = omega.copy()
        elif omega.shape == (self.n_snv, self.n_clones):
            out = omega.T.copy()
        else:
            raise ValueError(
                "omega_prior must have shape (n_clones,n_snv) or (n_snv,n_clones)."
            )
        if not np.all((out == 0) | (out == 1)):
            raise ValueError("omega_prior must be binary.")
        return out.astype(np.int8)

    def _normalize_genotype_prior(self) -> np.ndarray:
        prior = self.genotype_prior
        if prior.ndim == 0:
            out = np.full((self.n_clones, self.n_snv), float(prior), dtype=float)
        elif prior.shape == (self.n_snv,):
            out = np.broadcast_to(prior[None, :], (self.n_clones, self.n_snv)).copy()
        elif prior.shape == (self.n_clones, self.n_snv):
            out = prior.copy()
        else:
            raise ValueError(
                "genotype_prior must be scalar, shape (n_snv,), or shape (n_clones,n_snv)."
            )
        return out

    def _current_genotype_prior(self) -> np.ndarray:
        if self.omega_prior_ is None:
            prior = self._normalize_genotype_prior()
        else:
            rate = float(np.clip(self.relax_rate_, self.eps, 1.0 - self.eps))
            prior = np.where(self.omega_prior_ == 1, 1.0 - rate, rate).astype(float)
        if self.fix_reference_clone:
            prior[0, :] = self.eps
        return np.clip(prior, self.eps, 1.0 - self.eps)

    def _update_genotype_prior_parameters(self) -> None:
        if not (self.sample_relax_rate and self.omega_prior_ is not None):
            return
        mask = np.ones_like(self.genotype_matrix_, dtype=bool)
        if self.fix_reference_clone:
            mask[0, :] = False
        differences = int(np.sum(self.genotype_matrix_[mask] != self.omega_prior_[mask]))
        matches = int(np.sum(self.genotype_matrix_[mask] == self.omega_prior_[mask]))
        a, b = self.relax_rate_prior
        self.relax_rate_ = float(
            np.clip(self.rng.beta(a + differences, b + matches), self.eps, 1.0 - self.eps)
        )

    def _genotype_log_prior(self) -> float:
        prior = self._current_genotype_prior()
        genotype = self.genotype_matrix_.astype(bool)
        mask = np.ones_like(genotype, dtype=bool)
        if self.fix_reference_clone:
            mask[0, :] = False
        values = np.where(genotype, np.log(prior), np.log1p(-prior))
        return float(values[mask].sum())

    def _extra_log_prior(self) -> float:
        if self.relax_rate_ is None:
            return 0.0
        a, b = self.relax_rate_prior
        r = float(np.clip(self.relax_rate_, self.eps, 1.0 - self.eps))
        return float((a - 1.0) * math.log(r) + (b - 1.0) * math.log1p(-r))

    def _clear_subclass_traces(self) -> None:
        self.relax_rate_trace_.clear()

    def _append_subclass_tracking(self, config) -> None:
        self.relax_rate_trace_.append(
            np.nan if self.relax_rate_ is None else float(self.relax_rate_)
        )

    def _extra_to_dict(self) -> dict[str, Any]:
        return {
            "fix_reference_clone": self.fix_reference_clone,
            "omega_prior": None if self.omega_prior_ is None else self.omega_prior_.copy(),
            "relax_rate": None if self.relax_rate_ is None else float(self.relax_rate_),
            "relax_rate_prior": self.relax_rate_prior,
            "sample_relax_rate": self.sample_relax_rate,
            "genotype_prior": self._current_genotype_prior().copy(),
            "genotype_matrix_trace": np.asarray(self.genotype_state_trace_),
            "relax_rate_trace": np.asarray(self.relax_rate_trace_, dtype=float),
        }


PGM_fitter = CactriOmega
