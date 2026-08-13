from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from .base import Cactri
from .utils.tree import (
    edge_assignments_to_profiles,
    inverse_distance_transition,
    tree_vertex_levels,
    vertex_clone_presence,
)

TreePrior = Literal["uniform", "level_inverse"]


class CactriTree(Cactri):
    """Cactri with mutation origins constrained to a full binary clone tree.

    When ``observed_edge_assignment`` is provided, a learned distance-aware
    edge-error rate is enabled by default. The supplied assignment is treated as
    noisy evidence rather than a fixed truth.
    """

    def __init__(
        self,
        *,
        n_levels: int,
        tree_prior: TreePrior = "uniform",
        observed_edge_assignment: np.ndarray | None = None,
        observed_edge_probabilities: np.ndarray | None = None,
        mutation_edge_prior: np.ndarray | None = None,
        learn_edge_error_rate: bool | None = None,
        edge_error_rate_prior: tuple[float, float] = (1.0, 19.0),
        edge_error_rate_init: float | None = None,
        distance_power: float = 1.0,
        fixed_edge_confidence: float = 0.95,
        **kwargs: Any,
    ) -> None:
        if n_levels < 0:
            raise ValueError("n_levels must be nonnegative.")
        if tree_prior not in {"uniform", "level_inverse"}:
            raise ValueError("tree_prior must be 'uniform' or 'level_inverse'.")
        if len(edge_error_rate_prior) != 2 or min(edge_error_rate_prior) <= 0:
            raise ValueError("edge_error_rate_prior must contain two positive values.")
        if edge_error_rate_init is not None and not 0 < edge_error_rate_init < 1:
            raise ValueError("edge_error_rate_init must lie in (0,1).")
        if distance_power <= 0:
            raise ValueError("distance_power must be positive.")
        if not 0 < fixed_edge_confidence < 1:
            raise ValueError("fixed_edge_confidence must lie in (0,1).")

        self.n_levels = int(n_levels)
        self.n_leaf_clones = 2**self.n_levels
        self.n_tree_vertices = 2 ** (self.n_levels + 1) - 1
        self.tree_prior = tree_prior
        supplied_count = sum(
            value is not None
            for value in (
                observed_edge_assignment,
                observed_edge_probabilities,
                mutation_edge_prior,
            )
        )
        if supplied_count > 1:
            raise ValueError(
                "observed_edge_assignment, observed_edge_probabilities, and "
                "mutation_edge_prior are mutually exclusive."
            )
        self.observed_edge_assignment_input = (
            None
            if observed_edge_assignment is None
            else np.asarray(observed_edge_assignment, dtype=np.int64).copy()
        )
        self.observed_edge_probabilities_input = (
            None
            if observed_edge_probabilities is None
            else np.asarray(observed_edge_probabilities, dtype=float).copy()
        )
        self.mutation_edge_prior_input = (
            None if mutation_edge_prior is None else np.asarray(mutation_edge_prior, dtype=float).copy()
        )
        has_observed_edges = (
            observed_edge_assignment is not None
            or observed_edge_probabilities is not None
        )
        self.learn_edge_error_rate = (
            has_observed_edges
            if learn_edge_error_rate is None
            else bool(learn_edge_error_rate)
        )
        if self.learn_edge_error_rate and not has_observed_edges:
            raise ValueError(
                "learn_edge_error_rate requires observed_edge_assignment or "
                "observed_edge_probabilities."
            )
        self.edge_error_rate_prior = tuple(float(x) for x in edge_error_rate_prior)
        self.edge_error_rate_init = edge_error_rate_init
        self.distance_power = float(distance_power)
        self.fixed_edge_confidence = float(fixed_edge_confidence)

        self._vertex_clone_presence = vertex_clone_presence(self.n_levels, include_reference=True)
        self._vertex_levels = tree_vertex_levels(self.n_levels)
        self._base_edge_prior = self._make_base_edge_prior()
        self._edge_transition: np.ndarray | None = None
        self._fixed_mutation_edge_prior: np.ndarray | None = None
        self.observed_edge_assignment_: np.ndarray | None = None
        self.observed_edge_probabilities_: np.ndarray | None = None
        self._smoothed_observed_edge_probabilities: np.ndarray | None = None
        self.edge_error_component_trace_: list[int] = []
        self.mutation_tree_assignment_: np.ndarray | None = None
        self.mutation_profile_: np.ndarray | None = None
        self.genotype_matrix_: np.ndarray | None = None
        self.edge_error_rate_: float | None = None
        self.edge_error_rate_trace_: list[float] = []
        self.mutation_tree_assignment_trace_: list[np.ndarray] = []

        kwargs.setdefault("p_unobs_beta_prior", (1.0, 999.0))
        super().__init__(n_clones=self.n_leaf_clones + 1, **kwargs)
        self.mutation_profile_trace_ = self.genotype_state_trace_

    def _tree_distance_residual_weights(self, dominant: int) -> np.ndarray:
        """Return an unnormalized tree-distance base over non-dominant clones."""

        weights = np.zeros(self.n_clones, dtype=float)
        first_leaf = 2**self.n_levels - 1

        def node_distance(left: int, right: int) -> int:
            ancestors: dict[int, int] = {}
            node = left
            distance = 0
            while True:
                ancestors[node] = distance
                if node == 0:
                    break
                node = (node - 1) // 2
                distance += 1
            node = right
            distance = 0
            while node not in ancestors:
                node = (node - 1) // 2
                distance += 1
            return distance + ancestors[node]

        for clone in range(self.n_clones):
            if clone == dominant:
                continue
            if clone == 0 or dominant == 0:
                distance = self.n_levels + 1
            else:
                distance = node_distance(
                    first_leaf + dominant - 1,
                    first_leaf + clone - 1,
                )
            weights[clone] = math.exp(
                -self.clone_mixture.tree_distance_decay * float(distance)
            )
        return weights

    def _make_base_edge_prior(self) -> np.ndarray:
        if self.tree_prior == "uniform":
            return np.full(self.n_tree_vertices, 1.0 / self.n_tree_vertices, dtype=float)
        weights = 2.0 ** (-self._vertex_levels.astype(float))
        return weights / weights.sum()

    def _initialize_genotype_state(self) -> None:
        if self.observed_edge_probabilities_input is not None:
            probabilities = self._normalize_observed_edge_probabilities(
                self.observed_edge_probabilities_input
            )
            self.observed_edge_probabilities_ = probabilities
            self._initialize_edge_transition()
            self._smoothed_observed_edge_probabilities = (
                probabilities @ self._edge_transition
            )
            self._initialize_edge_error_rate()
            # The supplied posterior itself is the iteration-zero distribution.
            assignment = self._sample_rows(np.log(np.clip(probabilities, self.eps, 1.0)))
        elif self.observed_edge_assignment_input is not None:
            observed = self.observed_edge_assignment_input
            if observed.shape != (self.n_snv,):
                raise ValueError(
                    f"observed_edge_assignment must have shape ({self.n_snv},)."
                )
            if np.any((observed < 0) | (observed >= self.n_tree_vertices)):
                raise ValueError("observed_edge_assignment contains an invalid vertex.")
            self.observed_edge_assignment_ = observed.copy()
            self._initialize_edge_transition()
            self._initialize_edge_error_rate()
            assignment = observed.copy()
        elif self.mutation_edge_prior_input is not None:
            self._fixed_mutation_edge_prior = self._normalize_mutation_edge_prior(
                self.mutation_edge_prior_input
            )
            fixed = self._fixed_mutation_edge_prior
            fixed_matrix = (
                np.broadcast_to(fixed, (self.n_snv, self.n_tree_vertices))
                if fixed.ndim == 1
                else fixed
            )
            assignment = self._sample_rows(np.log(fixed_matrix))
        else:
            assignment = self._sample_rows(
                np.broadcast_to(
                    np.log(self._base_edge_prior),
                    (self.n_snv, self.n_tree_vertices),
                )
            )
        self.mutation_tree_assignment_ = np.asarray(assignment, dtype=np.int64)
        self._refresh_mutation_profile()

    def _initialize_edge_transition(self) -> None:
        transition = inverse_distance_transition(
            self.n_tree_vertices, power=self.distance_power
        )
        transition *= self._base_edge_prior[None, :]
        np.fill_diagonal(transition, 0.0)
        transition /= transition.sum(axis=1, keepdims=True)
        self._edge_transition = transition

    def _initialize_edge_error_rate(self) -> None:
        if self.learn_edge_error_rate:
            if self.edge_error_rate_init is None:
                a, b = self.edge_error_rate_prior
                self.edge_error_rate_ = float(
                    np.clip(self.rng.beta(a, b), self.eps, 1.0 - self.eps)
                )
            else:
                self.edge_error_rate_ = float(self.edge_error_rate_init)
        else:
            self.edge_error_rate_ = 1.0 - self.fixed_edge_confidence

    def _normalize_observed_edge_probabilities(
        self, probabilities: np.ndarray
    ) -> np.ndarray:
        arr = np.asarray(probabilities, dtype=float)
        if arr.shape != (self.n_snv, self.n_tree_vertices):
            raise ValueError(
                "observed_edge_probabilities must have shape "
                f"({self.n_snv}, {self.n_tree_vertices})."
            )
        if np.any(~np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(
                "observed_edge_probabilities must be finite and nonnegative."
            )
        row_sum = arr.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0):
            raise ValueError(
                "each observed_edge_probabilities row must have positive mass."
            )
        return arr / row_sum

    def _normalize_mutation_edge_prior(self, prior: np.ndarray) -> np.ndarray:
        arr = np.asarray(prior, dtype=float)
        if arr.ndim == 1:
            if arr.shape != (self.n_tree_vertices,):
                raise ValueError("mutation_edge_prior vector has the wrong length.")
            if np.any(arr < 0) or arr.sum() <= 0:
                raise ValueError("mutation_edge_prior must be nonnegative with positive mass.")
            return arr / arr.sum()
        if arr.ndim == 2:
            if arr.shape != (self.n_snv, self.n_tree_vertices):
                raise ValueError(
                    "mutation_edge_prior matrix must have shape (n_snv,n_tree_vertices)."
                )
            if np.any(arr < 0) or np.any(arr.sum(axis=1) <= 0):
                raise ValueError("each mutation_edge_prior row needs positive mass.")
            return arr / arr.sum(axis=1, keepdims=True)
        raise ValueError("mutation_edge_prior must be a vector or matrix.")

    def _current_edge_prior(self) -> np.ndarray:
        if self.observed_edge_probabilities_ is not None:
            rate = float(np.clip(self.edge_error_rate_, self.eps, 1.0 - self.eps))
            probabilities = (
                (1.0 - rate) * self.observed_edge_probabilities_
                + rate * self._smoothed_observed_edge_probabilities
            )
            probabilities = np.clip(probabilities, self.eps, None)
            return probabilities / probabilities.sum(axis=1, keepdims=True)
        if self.observed_edge_assignment_ is not None:
            rate = float(np.clip(self.edge_error_rate_, self.eps, 1.0 - self.eps))
            probabilities = rate * self._edge_transition[self.observed_edge_assignment_]
            probabilities[
                np.arange(self.n_snv), self.observed_edge_assignment_
            ] = 1.0 - rate
            return np.clip(probabilities, self.eps, 1.0)
        if self._fixed_mutation_edge_prior is not None:
            if self._fixed_mutation_edge_prior.ndim == 1:
                return np.broadcast_to(
                    self._fixed_mutation_edge_prior,
                    (self.n_snv, self.n_tree_vertices),
                )
            return self._fixed_mutation_edge_prior
        return np.broadcast_to(
            self._base_edge_prior, (self.n_snv, self.n_tree_vertices)
        )

    def _sample_genotype_state(self) -> None:
        alt_by_clone, ref_by_clone = self._aggregate_mutation_counts_by_clone()
        p_obs = np.clip(self.p_obs_by_mutation_, self.eps, 1.0 - self.eps)
        p_unobs = float(np.clip(self.p_unobs_, self.eps, 1.0 - self.eps))
        present = (
            alt_by_clone * np.log(p_obs)[None, :]
            + ref_by_clone * np.log1p(-p_obs)[None, :]
        )
        absent = alt_by_clone * np.log(p_unobs) + ref_by_clone * np.log1p(-p_unobs)
        base_absent = absent.sum(axis=0)
        delta = present - absent
        vertex_gain = self._vertex_clone_presence.astype(float) @ delta
        log_likelihood = (base_absent[None, :] + vertex_gain).T
        log_posterior = log_likelihood + np.log(self._current_edge_prior())
        self.mutation_tree_assignment_ = self._sample_rows(log_posterior)
        self._refresh_mutation_profile()

    def _refresh_mutation_profile(self) -> None:
        profile = edge_assignments_to_profiles(
            self.mutation_tree_assignment_, n_levels=self.n_levels, include_reference=True
        )
        self.mutation_profile_ = profile
        self.genotype_matrix_ = self.mutation_profile_

    def _genotype_matrix(self) -> np.ndarray:
        if self.mutation_profile_ is None:
            raise RuntimeError("Call prefit before accessing mutation profiles.")
        return self.mutation_profile_

    def _update_genotype_prior_parameters(self) -> None:
        if not self.learn_edge_error_rate:
            return
        a, b = self.edge_error_rate_prior
        if self.observed_edge_probabilities_ is not None:
            rows = np.arange(self.n_snv)
            assignment = self.mutation_tree_assignment_
            direct = self.observed_edge_probabilities_[rows, assignment]
            smoothed = self._smoothed_observed_edge_probabilities[rows, assignment]
            rate = float(np.clip(self.edge_error_rate_, self.eps, 1.0 - self.eps))
            denominator = (1.0 - rate) * direct + rate * smoothed
            corrupted_probability = np.divide(
                rate * smoothed, np.clip(denominator, self.eps, None)
            )
            corrupted = self.rng.random(self.n_snv) < corrupted_probability
            n_corrupted = int(corrupted.sum())
            self.edge_error_rate_ = float(
                np.clip(
                    self.rng.beta(a + n_corrupted, b + self.n_snv - n_corrupted),
                    self.eps,
                    1.0 - self.eps,
                )
            )
            self.edge_error_component_trace_.append(n_corrupted)
            return
        if self.observed_edge_assignment_ is not None:
            differences = int(
                np.sum(self.mutation_tree_assignment_ != self.observed_edge_assignment_)
            )
            matches = int(self.n_snv - differences)
            self.edge_error_rate_ = float(
                np.clip(
                    self.rng.beta(a + differences, b + matches),
                    self.eps,
                    1.0 - self.eps,
                )
            )

    def mutation_edge_assignment_log_prior(self) -> float:
        prior = self._current_edge_prior()
        return float(
            np.log(prior[np.arange(self.n_snv), self.mutation_tree_assignment_]).sum()
        )

    def _genotype_log_prior(self) -> float:
        return self.mutation_edge_assignment_log_prior()

    def _extra_log_prior(self) -> float:
        if not self.learn_edge_error_rate or self.edge_error_rate_ is None:
            return 0.0
        a, b = self.edge_error_rate_prior
        rate = float(np.clip(self.edge_error_rate_, self.eps, 1.0 - self.eps))
        return float((a - 1.0) * math.log(rate) + (b - 1.0) * math.log1p(-rate))

    def _genotype_state_for_tracking(self) -> np.ndarray:
        return self.mutation_profile_

    def _clear_subclass_traces(self) -> None:
        self.edge_error_rate_trace_.clear()
        self.edge_error_component_trace_.clear()
        self.mutation_tree_assignment_trace_.clear()

    def _append_subclass_tracking(self, config) -> None:
        self.edge_error_rate_trace_.append(
            np.nan if self.edge_error_rate_ is None else float(self.edge_error_rate_)
        )
        if config.genotype_state:
            self.mutation_tree_assignment_trace_.append(
                self.mutation_tree_assignment_.copy()
            )

    def _extra_to_dict(self) -> dict[str, Any]:
        return {
            "n_levels": self.n_levels,
            "n_leaf_clones": self.n_leaf_clones,
            "n_tree_vertices": self.n_tree_vertices,
            "tree_prior": self.tree_prior,
            "mutation_tree_assignment": self.mutation_tree_assignment_.copy(),
            "mutation_profile": self.mutation_profile_.copy(),
            "observed_edge_assignment": (
                None
                if self.observed_edge_assignment_ is None
                else self.observed_edge_assignment_.copy()
            ),
            "observed_edge_probabilities": (
                None
                if self.observed_edge_probabilities_ is None
                else self.observed_edge_probabilities_.copy()
            ),
            "edge_error_rate": (
                None if self.edge_error_rate_ is None else float(self.edge_error_rate_)
            ),
            "edge_error_rate_prior": self.edge_error_rate_prior,
            "learn_edge_error_rate": self.learn_edge_error_rate,
            "distance_power": self.distance_power,
            "edge_error_rate_trace": np.asarray(
                self.edge_error_rate_trace_, dtype=float
            ),
            "edge_error_component_trace": np.asarray(
                self.edge_error_component_trace_, dtype=np.int64
            ),
            "mutation_tree_assignment_trace": np.asarray(
                self.mutation_tree_assignment_trace_, dtype=np.int64
            ),
            "mutation_profile_trace": np.asarray(
                [
                    edge_assignments_to_profiles(
                        row, n_levels=self.n_levels, include_reference=True
                    )
                    for row in self.mutation_tree_assignment_trace_
                ],
                dtype=np.int8,
            ),
        }


PGM_fitter = CactriTree
