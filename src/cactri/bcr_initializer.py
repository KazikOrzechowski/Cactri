from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ._numba_accelerator import Accelerator, Backend
from .utils.assignments import InitSpec, initialize_assignments, reindex_assignments
from .utils.consensus import coassignment_matrix, partition_medoid
from .utils.validation import coerce_sequences, normalize_dirichlet_prior

BCRSampler = Literal["approximate", "sequential"]


class BCRInitializer:
    """Standalone CRP/Dirichlet model for BCR-based initialization."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        dirichlet_prior: float | np.ndarray = 0.5,
        accelerator: Backend = "auto",
        deterministic: bool = False,
        eps: float = 1e-12,
        random_state: int | None = None,
    ) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        self.alpha = float(alpha)
        self.dirichlet_prior = dirichlet_prior
        self.eps = float(eps)
        self.rng = np.random.default_rng(random_state)
        self.accelerator = Accelerator(accelerator, deterministic=deterministic)

        self.sequences_: np.ndarray | None = None
        self.assignments_: np.ndarray | None = None
        self.bcr_profiles_: np.ndarray | None = None
        self.assignment_trace_: list[np.ndarray] = []
        self.log_likelihood_trace_: list[float] = []

        self.consensus_assignments_: np.ndarray | None = None
        self.coassignment_matrix_: np.ndarray | None = None
        self.consensus_bcr_profiles_: np.ndarray | None = None
        self.chain_partitions_: np.ndarray | None = None
        self.diagnostics_: dict[str, Any] | None = None

    def prefit(
        self,
        bcr_sequences: np.ndarray,
        *,
        init: InitSpec = "random",
        random_init_clusters: int = 30,
    ) -> BCRInitializer:
        self.sequences_ = coerce_sequences(bcr_sequences)
        self.assignments_ = initialize_assignments(
            init,
            n_cells=self.n_cells,
            sequences=self.sequences_,
            rng=self.rng,
            random_init_clusters=random_init_clusters,
        )
        self.bcr_profiles_ = self._sample_profiles(self.assignments_)
        self.assignment_trace_.clear()
        self.log_likelihood_trace_.clear()
        return self

    def fit(
        self,
        bcr_sequences: np.ndarray | None = None,
        *,
        n_iter: int = 500,
        init: InitSpec = "random",
        random_init_clusters: int = 30,
        assignment_sampler: BCRSampler = "approximate",
        track_assignments: bool = True,
        verbose: bool = False,
        progress_every: int = 50,
    ) -> dict[str, Any]:
        if bcr_sequences is not None:
            self.prefit(
                bcr_sequences,
                init=init,
                random_init_clusters=random_init_clusters,
            )
        self._require_prefit()
        for iteration in range(n_iter):
            self.bcr_profiles_ = self._sample_profiles(self.assignments_)
            if assignment_sampler == "approximate":
                self.approximate_assignment_sweep()
            elif assignment_sampler == "sequential":
                self.sequential_assignment_sweep()
            else:
                raise ValueError("assignment_sampler must be 'approximate' or 'sequential'.")
            self.log_likelihood_trace_.append(self.log_likelihood())
            if track_assignments:
                self.assignment_trace_.append(self.assignments_.copy())
            if verbose and ((iteration + 1) % progress_every == 0 or iteration == 0):
                print(
                    f"iter={iteration + 1} K={self.n_hyperclusters} "
                    f"loglik={self.log_likelihood_trace_[-1]:.3f}"
                )
        return self.to_dict()

    def consensus_fit(
        self,
        bcr_sequences: np.ndarray,
        *,
        n_chains: int = 4,
        n_iter: int = 500,
        burn_in: int | float = 0.5,
        thin: int = 5,
        init: InitSpec = "random",
        random_init_clusters: int = 30,
        assignment_sampler: BCRSampler = "approximate",
        diagnostics: bool = True,
    ) -> dict[str, Any]:
        if n_chains < 1 or n_iter < 1 or thin < 1:
            raise ValueError("n_chains, n_iter, and thin must be positive.")
        if isinstance(burn_in, float):
            if not 0 <= burn_in < 1:
                raise ValueError("fractional burn_in must lie in [0,1).")
            burn = int(n_iter * burn_in)
        else:
            burn = int(burn_in)
        if not 0 <= burn < n_iter:
            raise ValueError("burn_in must be between 0 and n_iter-1.")

        sequences = coerce_sequences(bcr_sequences)
        partitions: list[np.ndarray] = []
        chain_summary: list[dict[str, Any]] = []
        for chain_index in range(n_chains):
            seed = int(self.rng.integers(0, 2**32 - 1))
            chain = BCRInitializer(
                alpha=self.alpha,
                dirichlet_prior=self.dirichlet_prior,
                accelerator=self.accelerator.backend,
                deterministic=self.accelerator.deterministic,
                eps=self.eps,
                random_state=seed,
            )
            chain.fit(
                sequences,
                n_iter=n_iter,
                init=init,
                random_init_clusters=random_init_clusters,
                assignment_sampler=assignment_sampler,
                track_assignments=True,
            )
            retained = chain.assignment_trace_[burn::thin]
            partitions.extend(z.copy() for z in retained)
            chain_summary.append(
                {
                    "chain": chain_index,
                    "seed": seed,
                    "final_n_hyperclusters": chain.n_hyperclusters,
                    "final_log_likelihood": chain.log_likelihood(),
                    "retained_draws": len(retained),
                }
            )

        all_partitions = np.asarray(partitions, dtype=np.int64)
        consensus, medoid_index, losses = partition_medoid(all_partitions)
        self.sequences_ = sequences
        self.assignments_ = consensus.copy()
        self.bcr_profiles_ = self._posterior_mean_profiles(consensus)
        self.consensus_assignments_ = consensus.copy()
        self.coassignment_matrix_ = coassignment_matrix(all_partitions)
        self.consensus_bcr_profiles_ = self.bcr_profiles_.copy()
        self.chain_partitions_ = all_partitions
        self.diagnostics_ = (
            {
                "chain_summary": chain_summary,
                "medoid_index": medoid_index,
                "medoid_loss": float(losses[medoid_index]),
                "n_retained_partitions": int(all_partitions.shape[0]),
                "n_consensus_hyperclusters": int(consensus.max()) + 1,
            }
            if diagnostics
            else None
        )
        return {
            "assignments": self.consensus_assignments_.copy(),
            "consensus_assignments": self.consensus_assignments_.copy(),
            "coassignment_matrix": self.coassignment_matrix_.copy(),
            "bcr_profiles": self.consensus_bcr_profiles_.copy(),
            "partitions": self.chain_partitions_.copy(),
            "diagnostics": self.diagnostics_,
        }

    def approximate_assignment_sweep(self) -> None:
        self._require_prefit()
        z = self.assignments_
        n_h = self.n_hyperclusters
        counts = np.bincount(z, minlength=n_h).astype(float)
        weights = counts[None, :] - np.eye(n_h)[z]
        log_prob = np.full((self.n_cells, n_h), -np.inf, dtype=float)
        valid = weights > 0
        log_prob[valid] = np.log(weights[valid])
        log_prob += self._cell_cluster_loglikelihood()
        new_score = np.log(self.alpha) + self._prior_predictive(self.sequences_)
        sampled = self._sample_rows(np.concatenate([log_prob, new_score[:, None]], axis=1))
        new = sampled == n_h
        if np.any(new):
            sampled[new] = np.arange(n_h, n_h + int(new.sum()))
        self.assignments_ = reindex_assignments(sampled)
        self.bcr_profiles_ = self._sample_profiles(self.assignments_)

    def sequential_assignment_sweep(self) -> None:
        self._require_prefit()
        for cell in self.rng.permutation(self.n_cells):
            z = self.assignments_.copy()
            current = z[cell]
            z[cell] = -1
            if np.sum(z == current) == 0:
                z[z > current] -= 1
            valid = z >= 0
            n_h = int(z[valid].max()) + 1 if np.any(valid) else 0
            profiles = self._posterior_mean_profiles(z[valid], self.sequences_[valid]) if n_h else np.empty((0, self.L, 4))
            scores = []
            for h in range(n_h):
                count = int(np.sum(z == h))
                value = np.log(count)
                for pos in range(self.L):
                    value += np.log(np.clip(profiles[h, pos, self.sequences_[cell, pos]], self.eps, 1.0))
                scores.append(value)
            scores.append(np.log(self.alpha) + float(self._prior_predictive(self.sequences_[cell : cell + 1])[0]))
            choice = int(self._sample_rows(np.asarray(scores)[None, :])[0])
            z[cell] = choice
            self.assignments_ = reindex_assignments(z)
        self.bcr_profiles_ = self._sample_profiles(self.assignments_)

    def posterior_assignment_probabilities(self) -> np.ndarray:
        ll = self._cell_cluster_loglikelihood()
        counts = np.bincount(self.assignments_, minlength=self.n_hyperclusters).astype(float)
        log_prob = np.log(counts[None, :] + self.eps) + ll
        maximum = np.max(log_prob, axis=1, keepdims=True)
        probability = np.exp(log_prob - maximum)
        return probability / probability.sum(axis=1, keepdims=True)

    def log_likelihood(self) -> float:
        rows = np.arange(self.n_cells)
        return float(self._cell_cluster_loglikelihood()[rows, self.assignments_].sum())

    def to_dict(self) -> dict[str, Any]:
        self._require_prefit()
        return {
            "assignments": self.assignments_.copy(),
            "bcr_profiles": self.bcr_profiles_.copy(),
            "n_hyperclusters": self.n_hyperclusters,
            "log_likelihood": self.log_likelihood(),
            "assignment_trace": np.asarray(self.assignment_trace_, dtype=object),
            "log_likelihood_trace": np.asarray(self.log_likelihood_trace_, dtype=float),
        }

    @property
    def n_cells(self) -> int:
        self._require_data()
        return int(self.sequences_.shape[0])

    @property
    def L(self) -> int:
        self._require_data()
        return int(self.sequences_.shape[1])

    @property
    def n_hyperclusters(self) -> int:
        self._require_prefit()
        return int(self.assignments_.max()) + 1

    def _cell_cluster_loglikelihood(self) -> np.ndarray:
        return self.accelerator.bcr_loglik(
            self.sequences_, np.log(np.clip(self.bcr_profiles_, self.eps, 1.0))
        )

    def _sample_profiles(self, assignments: np.ndarray) -> np.ndarray:
        z = np.asarray(assignments, dtype=np.int64)
        prior = normalize_dirichlet_prior(self.dirichlet_prior, self.L)
        counts = np.zeros((int(z.max()) + 1, self.L, 4), dtype=float)
        positions = np.arange(self.L)
        np.add.at(counts, (z[:, None], positions[None, :], self.sequences_), 1.0)
        draws = self.rng.gamma(counts + prior[None, :, :], 1.0)
        return draws / draws.sum(axis=-1, keepdims=True)

    def _posterior_mean_profiles(
        self, assignments: np.ndarray, sequences: np.ndarray | None = None
    ) -> np.ndarray:
        sequences = self.sequences_ if sequences is None else sequences
        z = np.asarray(assignments, dtype=np.int64)
        prior = normalize_dirichlet_prior(self.dirichlet_prior, sequences.shape[1])
        counts = np.zeros((int(z.max()) + 1, sequences.shape[1], 4), dtype=float)
        positions = np.arange(sequences.shape[1])
        np.add.at(counts, (z[:, None], positions[None, :], sequences), 1.0)
        posterior = counts + prior[None, :, :]
        return posterior / posterior.sum(axis=-1, keepdims=True)

    def _prior_predictive(self, sequences: np.ndarray) -> np.ndarray:
        prior = normalize_dirichlet_prior(self.dirichlet_prior, self.L)
        probability = prior / prior.sum(axis=1, keepdims=True)
        return np.log(probability[np.arange(self.L)[None, :], sequences]).sum(axis=1)

    def _sample_rows(self, log_prob: np.ndarray) -> np.ndarray:
        return self.accelerator.sample_rows(log_prob, self.rng.random(log_prob.shape[0]))

    def _require_data(self) -> None:
        if self.sequences_ is None:
            raise RuntimeError("Call prefit or fit with BCR sequences first.")

    def _require_prefit(self) -> None:
        self._require_data()
        if self.assignments_ is None or self.bcr_profiles_ is None:
            raise RuntimeError("Call prefit or fit first.")
