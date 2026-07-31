from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any, Literal, Mapping

import numpy as np

from ._numba_accelerator import Accelerator, Backend
from .config import SplitMergeConfig, TrackingConfig
from .state import StateView
from .utils.assignments import InitSpec, initialize_assignments, reindex_assignments
from .utils.posterior import (
    cell_clone_probabilities_from_trace,
    coassignment_probabilities_from_trace,
    coherent_cell_genotype_probabilities,
)
from .utils.validation import (
    coerce_sequences,
    normalize_beta_prior,
    normalize_dirichlet_prior,
    validate_read_counts,
)

AssignmentSampler = Literal["approximate", "sequential", "split_merge"]
AssignmentLikelihood = Literal["joint", "bcr"]
NewClusterLikelihood = Literal["auxiliary", "prior_predictive"]


class Cactri(ABC):
    """Abstract shared CRP/BCR and clone-assignment model.

    Subclasses implement a genotype representation and its prior. Random values
    are generated only by ``self.rng``; accelerator kernels receive those values
    as explicit arguments so NumPy and Numba follow the same sampling decisions.
    """

    def __init__(
        self,
        *,
        n_clones: int,
        alpha: float = 1.0,
        dirichlet_prior: float | np.ndarray = 0.5,
        p_obs_beta_prior: tuple[float, float] | np.ndarray = (50.0, 50.0),
        p_unobs_beta_prior: tuple[float, float] = (1.0, 999.0),
        p_obs_init: float | np.ndarray | None = None,
        p_unobs_init: float | None = None,
        clone_prior: np.ndarray | None = None,
        assignment_likelihood: AssignmentLikelihood = "joint",
        new_cluster_likelihood: NewClusterLikelihood = "auxiliary",
        accelerator: Backend = "auto",
        deterministic: bool = False,
        eps: float = 1e-12,
        random_state: int | None = None,
    ) -> None:
        if n_clones < 1:
            raise ValueError("n_clones must be at least 1.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if assignment_likelihood not in {"joint", "bcr"}:
            raise ValueError("assignment_likelihood must be 'joint' or 'bcr'.")
        if new_cluster_likelihood not in {"auxiliary", "prior_predictive"}:
            raise ValueError("new_cluster_likelihood must be 'auxiliary' or 'prior_predictive'.")
        if len(p_unobs_beta_prior) != 2 or min(p_unobs_beta_prior) <= 0:
            raise ValueError("p_unobs_beta_prior must contain two positive values.")
        if p_unobs_init is not None and not 0 < float(p_unobs_init) < 1:
            raise ValueError("p_unobs_init must lie in (0, 1).")

        self.n_clones = int(n_clones)
        self.alpha = float(alpha)
        self.alpha_ = float(alpha)
        self.dirichlet_prior = dirichlet_prior
        self.p_obs_beta_prior = np.asarray(p_obs_beta_prior, dtype=float)
        self.p_unobs_beta_prior = tuple(float(x) for x in p_unobs_beta_prior)
        self.p_obs_init = p_obs_init
        self.p_unobs_init = p_unobs_init
        self.assignment_likelihood = assignment_likelihood
        self.new_cluster_likelihood = new_cluster_likelihood
        self.eps = float(eps)
        self.rng = np.random.default_rng(random_state)
        self.accelerator = Accelerator(accelerator, deterministic=deterministic)
        self.accelerator_name = self.accelerator.backend
        self.deterministic = bool(deterministic)

        if clone_prior is None:
            self.clone_prior = np.full(self.n_clones, 1.0 / self.n_clones, dtype=float)
        else:
            prior = np.asarray(clone_prior, dtype=float)
            if prior.shape != (self.n_clones,) or np.any(prior < 0) or prior.sum() <= 0:
                raise ValueError("clone_prior must be a nonnegative vector of length n_clones.")
            self.clone_prior = prior / prior.sum()
        self.log_clone_prior = np.log(np.clip(self.clone_prior, self.eps, 1.0))

        self.sequences_: np.ndarray | None = None
        self.seqs_: np.ndarray | None = None  # legacy alias
        self.alt_counts_: np.ndarray | None = None
        self.total_counts_: np.ndarray | None = None
        self.assignments_: np.ndarray | None = None
        self.bcr_profiles_: np.ndarray | None = None
        self.hypercluster_to_clone_: np.ndarray | None = None
        self.p_obs_by_mutation_: np.ndarray | None = None
        self.p_unobs_: float | None = None
        self._p_obs_beta_prior_matrix: np.ndarray | None = None
        self.state_ = StateView(self)

        self.log_likelihood_trace_: list[float] = []
        self.bcr_log_likelihood_trace_: list[float] = []
        self.mutation_log_likelihood_trace_: list[float] = []
        self.crp_logprior_trace_: list[float] = []
        self.clone_assignment_logprior_trace_: list[float] = []
        self.genotype_logprior_trace_: list[float] = []
        self.log_posterior_trace_: list[float] = []
        self.n_hyperclusters_trace_: list[int] = []
        self.alpha_trace_: list[float] = []
        self.p_unobs_trace_: list[float] = []
        self.p_obs_mean_trace_: list[float] = []
        self.assignment_trace_: list[np.ndarray] = []
        self.cell_clone_assignment_trace_: list[np.ndarray] = []
        self.genotype_state_trace_: list[np.ndarray] = []
        self.p_obs_by_mutation_trace_: list[np.ndarray] = []

        self.split_merge_attempts_: int = 0
        self.split_merge_accepts_: int = 0
        self.split_attempts_: int = 0
        self.split_accepts_: int = 0
        self.merge_attempts_: int = 0
        self.merge_accepts_: int = 0

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def prefit(
        self,
        bcr_sequences: np.ndarray,
        alt_counts: np.ndarray,
        total_counts: np.ndarray,
        *,
        init: InitSpec = "identical_sequences",
        random_init_clusters: int = 3,
        bcr_initializer_config: Mapping[str, Any] | None = None,
    ) -> Cactri:
        sequences = coerce_sequences(bcr_sequences)
        alt, total = validate_read_counts(alt_counts, total_counts, n_cells=sequences.shape[0])
        self.sequences_ = self.seqs_ = sequences
        self.alt_counts_ = alt
        self.total_counts_ = total
        self._p_obs_beta_prior_matrix = normalize_beta_prior(
            self.p_obs_beta_prior,
            self.n_snv,
            name="p_obs_beta_prior",
        )

        if isinstance(init, str) and init == "bcr_consensus":
            from .bcr_initializer import BCRInitializer

            config = dict(bcr_initializer_config or {})
            initializer = BCRInitializer(
                alpha=float(config.pop("alpha", self.alpha)),
                dirichlet_prior=config.pop("dirichlet_prior", self.dirichlet_prior),
                accelerator=config.pop("accelerator", self.accelerator_name),
                deterministic=config.pop("deterministic", self.deterministic),
                random_state=config.pop("random_state", int(self.rng.integers(0, 2**32 - 1))),
            )
            initializer.consensus_fit(sequences, **config)
            z = initializer.consensus_assignments_.copy()
            self.bcr_initializer_ = initializer
        else:
            z = initialize_assignments(
                init,
                n_cells=self.n_cells,
                sequences=sequences,
                rng=self.rng,
                random_init_clusters=random_init_clusters,
            )

        self.assignments_ = z
        self.bcr_profiles_ = self._sample_bcr_profiles(z)
        self.hypercluster_to_clone_ = self.rng.choice(
            self.n_clones,
            size=self.n_hyperclusters,
            replace=True,
            p=self.clone_prior,
        ).astype(np.int64)
        self.p_obs_by_mutation_ = self._initialize_p_obs()
        self.p_unobs_ = self._initialize_p_unobs()
        self._initialize_genotype_state()
        self._clear_traces()
        return self

    def fit(
        self,
        n_iter: int = 1000,
        *,
        assignment_sampler: AssignmentSampler = "approximate",
        update_assignments: bool = True,
        update_bcr_profiles: bool = True,
        update_hypercluster_clones: bool = True,
        update_genotypes: bool = True,
        update_observation_probabilities: bool = True,
        tracking: TrackingConfig | None = None,
        split_merge_config: SplitMergeConfig | None = None,
        verbose: bool = False,
        progress_every: int = 50,
    ) -> dict[str, Any]:
        self._require_prefit()
        if n_iter < 0:
            raise ValueError("n_iter must be nonnegative.")
        if assignment_sampler not in {"approximate", "sequential", "split_merge"}:
            raise ValueError("unknown assignment_sampler.")
        tracking = tracking or TrackingConfig()

        for iteration in range(n_iter):
            if update_hypercluster_clones:
                self.hypercluster_to_clone_ = self._sample_hypercluster_clone_assignments()
            if update_bcr_profiles:
                self.bcr_profiles_ = self._sample_bcr_profiles(self.assignments_)
            if update_assignments:
                self.assignment_sweep(
                    assignment_sampler,
                    split_merge_config=split_merge_config,
                )
            if update_genotypes:
                self._sample_genotype_state()
                self._update_genotype_prior_parameters()
            if update_observation_probabilities:
                self._resample_observation_probabilities()
            if (iteration + 1) % tracking.every == 0:
                self._append_tracking(tracking)
            if verbose and ((iteration + 1) % progress_every == 0 or iteration == 0):
                print(
                    f"iter={iteration + 1} K={self.n_hyperclusters} "
                    f"loglik={self.log_likelihood():.3f} "
                    f"p_unobs={self.p_unobs_:.5g}"
                )
        return self.to_dict()

    def assignment_sweep(
        self,
        sampler: AssignmentSampler = "approximate",
        *,
        split_merge_config: SplitMergeConfig | None = None,
    ) -> None:
        if sampler == "approximate":
            self.approximate_assignment_sweep()
        elif sampler == "sequential":
            self.sequential_assignment_sweep()
        elif sampler == "split_merge":
            config = split_merge_config or SplitMergeConfig()
            if config.local_sampler == "approximate":
                self.approximate_assignment_sweep()
            elif config.local_sampler == "sequential":
                self.sequential_assignment_sweep()
            for _ in range(config.proposals_per_sweep):
                self.split_merge_assignment_sweep()
        else:  # pragma: no cover
            raise ValueError("unknown assignment sampler.")

    # ------------------------------------------------------------------
    # Assignment samplers
    # ------------------------------------------------------------------

    def approximate_assignment_sweep(self) -> None:
        self._require_prefit()
        z_old = self.assignments_.copy()
        n_h = self.n_hyperclusters
        bcr_ll = self._cell_hypercluster_bcr_loglikelihood()
        counts = np.bincount(z_old, minlength=n_h).astype(float)
        weights = counts[None, :] - np.eye(n_h, dtype=float)[z_old]
        existing = np.full((self.n_cells, n_h), -np.inf, dtype=float)
        valid = weights > 0
        existing[valid] = np.log(weights[valid])
        existing += bcr_ll

        cell_clone_ll = None
        if self.assignment_likelihood == "joint":
            existing += self._cell_hypercluster_mut_loglikelihood()
            cell_clone_ll = self._cell_clone_mut_loglikelihood()

        aux_clones = None
        if self.new_cluster_likelihood == "auxiliary":
            aux_profiles = self._sample_prior_bcr_profiles(self.n_cells)
            new_bcr = self._cellwise_bcr_loglikelihood_for_profiles(self.sequences_, aux_profiles)
            if self.assignment_likelihood == "joint":
                aux_clones = self.rng.choice(
                    self.n_clones, size=self.n_cells, p=self.clone_prior
                ).astype(np.int64)
                new_mut = cell_clone_ll[np.arange(self.n_cells), aux_clones]
            else:
                new_mut = 0.0
        else:
            new_bcr = self._bcr_prior_predictive_loglikelihood(self.sequences_)
            new_mut = (
                self._new_hypercluster_mutation_loglikelihood()
                if self.assignment_likelihood == "joint"
                else 0.0
            )

        new_score = np.log(self.alpha_) + new_bcr + new_mut
        sampled = self._sample_rows(np.concatenate([existing, new_score[:, None]], axis=1))
        new_mask = sampled == n_h
        z_tmp = sampled.copy()
        if np.any(new_mask):
            z_tmp[new_mask] = np.arange(n_h, n_h + int(new_mask.sum()))
        unique, z_new = np.unique(z_tmp, return_inverse=True)
        z_new = z_new.astype(np.int64)
        new_h_to_clone = self._resize_hypercluster_clones(z_old, z_new)
        if aux_clones is not None and np.any(new_mask):
            compact = np.searchsorted(unique, z_tmp[new_mask])
            new_h_to_clone[compact] = aux_clones[new_mask]
        self.assignments_ = z_new
        self.hypercluster_to_clone_ = new_h_to_clone
        self.bcr_profiles_ = self._sample_bcr_profiles(z_new)

    def sequential_assignment_sweep(self) -> None:
        """Cell-by-cell Gibbs-style reassignment using current profile parameters."""
        self._require_prefit()
        order = self.rng.permutation(self.n_cells)
        for cell in order:
            old_z = self.assignments_.copy()
            current = int(old_z[cell])
            keep = np.ones(self.n_cells, dtype=bool)
            keep[cell] = False
            remaining = old_z[keep]
            if np.sum(old_z == current) == 1:
                remaining = reindex_assignments(remaining)
                z_without = np.empty_like(old_z)
                z_without[keep] = remaining
                z_without[cell] = -1
                old_profiles = self.bcr_profiles_
                old_clones = self.hypercluster_to_clone_
                active_old = [h for h in range(len(old_profiles)) if h != current]
                self.bcr_profiles_ = old_profiles[active_old]
                self.hypercluster_to_clone_ = old_clones[active_old]
            else:
                z_without = old_z.copy()
                z_without[cell] = -1
            self.assignments_ = z_without

            n_h = self.bcr_profiles_.shape[0]
            counts = np.bincount(z_without[z_without >= 0], minlength=n_h).astype(float)
            bcr_scores = self._bcr_loglik_for_cells(self.sequences_[cell : cell + 1])[0]
            scores = np.log(np.clip(counts, self.eps, None)) + bcr_scores
            if self.assignment_likelihood == "joint":
                clone_scores = self._cell_clone_mut_loglikelihood()[cell]
                scores += clone_scores[self.hypercluster_to_clone_]
            new_bcr = float(self._bcr_prior_predictive_loglikelihood(self.sequences_[cell : cell + 1])[0])
            if self.assignment_likelihood == "joint":
                new_mut = float(
                    self._logsumexp(
                        clone_scores[None, :] + self.log_clone_prior[None, :], axis=1
                    )[0]
                )
            else:
                new_mut = 0.0
            all_scores = np.concatenate([scores, [np.log(self.alpha_) + new_bcr + new_mut]])
            choice = int(self._sample_rows(all_scores[None, :])[0])
            if choice == n_h:
                self.assignments_[cell] = n_h
                self.hypercluster_to_clone_ = np.append(
                    self.hypercluster_to_clone_,
                    self.rng.choice(self.n_clones, p=self.clone_prior),
                ).astype(np.int64)
                new_profile = self._sample_bcr_profiles(np.array([0], dtype=np.int64), sequences=self.sequences_[cell : cell + 1])
                self.bcr_profiles_ = np.concatenate([self.bcr_profiles_, new_profile], axis=0)
            else:
                self.assignments_[cell] = choice
        self.assignments_ = reindex_assignments(self.assignments_)
        self.bcr_profiles_ = self._sample_bcr_profiles(self.assignments_)

    def split_merge_assignment_sweep(self) -> bool:
        """Run one collapsed mutation-informed split-or-merge proposal.

        BCR profiles and hypercluster-to-clone labels are analytically
        marginalized when the partition is scored. Split proposals use a
        restricted Gibbs allocation whose probabilities combine the CRP size
        term, Dirichlet-multinomial BCR evidence, and clone-marginalized
        mutation evidence. Explicit profiles and clone labels are resampled
        only after an accepted move.
        """

        self._require_prefit()
        if self.n_cells < 2:
            return False

        first, second = self.rng.choice(self.n_cells, size=2, replace=False)
        label_first = int(self.assignments_[first])
        label_second = int(self.assignments_[second])
        old_assignments = self.assignments_.copy()
        cell_clone_loglik = self._cell_clone_mut_loglikelihood()

        self.split_merge_attempts_ += 1
        if label_first == label_second:
            self.split_attempts_ += 1
            members = np.flatnonzero(old_assignments == label_first)
            if members.size < 2:
                return False
            remaining = members[(members != first) & (members != second)]
            order = self.rng.permutation(remaining)
            group_a, group_b, log_q_forward = self._restricted_split_allocation(
                first,
                second,
                order,
                cell_clone_loglik=cell_clone_loglik,
            )
            candidate = old_assignments.copy()
            new_label = int(old_assignments.max()) + 1
            candidate[group_a] = label_first
            candidate[group_b] = new_label
            candidate = reindex_assignments(candidate)
            old_cluster_score = self._collapsed_cluster_log_score(
                members, cell_clone_loglik=cell_clone_loglik
            )
            split_score = (
                self._collapsed_cluster_log_score(
                    group_a, cell_clone_loglik=cell_clone_loglik
                )
                + self._collapsed_cluster_log_score(
                    group_b, cell_clone_loglik=cell_clone_loglik
                )
            )
            log_target_ratio = (
                math.log(self.alpha_)
                + math.lgamma(float(group_a.size))
                + math.lgamma(float(group_b.size))
                - math.lgamma(float(members.size))
                + split_score
                - old_cluster_score
            )
            log_acceptance = log_target_ratio - log_q_forward
            move = "split"
        else:
            self.merge_attempts_ += 1
            members_a = np.flatnonzero(old_assignments == label_first)
            members_b = np.flatnonzero(old_assignments == label_second)
            remaining = np.concatenate(
                [
                    members_a[members_a != first],
                    members_b[members_b != second],
                ]
            )
            order = self.rng.permutation(remaining)
            target_group = np.full(self.n_cells, -1, dtype=np.int8)
            target_group[members_a] = 0
            target_group[members_b] = 1
            _, _, log_q_reverse = self._restricted_split_allocation(
                first,
                second,
                order,
                cell_clone_loglik=cell_clone_loglik,
                target_group=target_group,
            )
            candidate = old_assignments.copy()
            candidate[candidate == label_second] = label_first
            candidate = reindex_assignments(candidate)
            union = np.concatenate([members_a, members_b])
            separate_score = (
                self._collapsed_cluster_log_score(
                    members_a, cell_clone_loglik=cell_clone_loglik
                )
                + self._collapsed_cluster_log_score(
                    members_b, cell_clone_loglik=cell_clone_loglik
                )
            )
            merged_score = self._collapsed_cluster_log_score(
                union, cell_clone_loglik=cell_clone_loglik
            )
            log_target_ratio = (
                -math.log(self.alpha_)
                + math.lgamma(float(union.size))
                - math.lgamma(float(members_a.size))
                - math.lgamma(float(members_b.size))
                + merged_score
                - separate_score
            )
            log_acceptance = log_target_ratio + log_q_reverse
            move = "merge"

        accepted = bool(np.log(self.rng.random()) < min(0.0, log_acceptance))
        if accepted:
            self.assignments_ = candidate
            self.bcr_profiles_ = self._sample_bcr_profiles(candidate)
            self.hypercluster_to_clone_ = self._sample_hypercluster_clone_assignments()
            self.split_merge_accepts_ += 1
            if move == "split":
                self.split_accepts_ += 1
            else:
                self.merge_accepts_ += 1
        return accepted

    def split_merge_diagnostics(self) -> dict[str, float | int]:
        """Return proposal counts and acceptance rates for global moves."""

        def rate(accepted: int, attempted: int) -> float:
            return float(accepted / attempted) if attempted else 0.0

        return {
            "attempts": self.split_merge_attempts_,
            "accepts": self.split_merge_accepts_,
            "acceptance_rate": rate(self.split_merge_accepts_, self.split_merge_attempts_),
            "split_attempts": self.split_attempts_,
            "split_accepts": self.split_accepts_,
            "split_acceptance_rate": rate(self.split_accepts_, self.split_attempts_),
            "merge_attempts": self.merge_attempts_,
            "merge_accepts": self.merge_accepts_,
            "merge_acceptance_rate": rate(self.merge_accepts_, self.merge_attempts_),
        }

    def _restricted_split_allocation(
        self,
        first: int,
        second: int,
        order: np.ndarray,
        *,
        cell_clone_loglik: np.ndarray,
        target_group: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Sample or score one sequential restricted-Gibbs split allocation."""

        prior = normalize_dirichlet_prior(self.dirichlet_prior, self.L)
        prior_sum = prior.sum(axis=1)

        def initial_stats(cell: int) -> tuple[np.ndarray, np.ndarray]:
            counts = np.zeros((self.L, 4), dtype=np.int64)
            counts[np.arange(self.L), self.sequences_[cell]] = 1
            return counts, cell_clone_loglik[cell].copy()

        group_a: list[int] = [int(first)]
        group_b: list[int] = [int(second)]
        counts_a, clone_sum_a = initial_stats(int(first))
        counts_b, clone_sum_b = initial_stats(int(second))
        log_q = 0.0

        for raw_cell in order:
            cell = int(raw_cell)
            residues = self.sequences_[cell]
            positions = np.arange(self.L)
            bcr_a = np.log(
                (counts_a[positions, residues] + prior[positions, residues])
                / (len(group_a) + prior_sum)
            ).sum()
            bcr_b = np.log(
                (counts_b[positions, residues] + prior[positions, residues])
                / (len(group_b) + prior_sum)
            ).sum()
            if self.assignment_likelihood == "joint":
                old_a = float(
                    self._logsumexp(
                        (clone_sum_a + self.log_clone_prior)[None, :], axis=1
                    )[0]
                )
                new_a = float(
                    self._logsumexp(
                        (clone_sum_a + cell_clone_loglik[cell] + self.log_clone_prior)[
                            None, :
                        ],
                        axis=1,
                    )[0]
                )
                old_b = float(
                    self._logsumexp(
                        (clone_sum_b + self.log_clone_prior)[None, :], axis=1
                    )[0]
                )
                new_b = float(
                    self._logsumexp(
                        (clone_sum_b + cell_clone_loglik[cell] + self.log_clone_prior)[
                            None, :
                        ],
                        axis=1,
                    )[0]
                )
                mutation_a = new_a - old_a
                mutation_b = new_b - old_b
            else:
                mutation_a = mutation_b = 0.0

            scores = np.asarray(
                [
                    math.log(len(group_a)) + float(bcr_a) + mutation_a,
                    math.log(len(group_b)) + float(bcr_b) + mutation_b,
                ],
                dtype=float,
            )
            log_norm = float(self._logsumexp(scores[None, :], axis=1)[0])
            if target_group is None:
                choice = int(self._sample_rows(scores[None, :])[0])
            else:
                choice = int(target_group[cell])
                if choice not in (0, 1):
                    raise ValueError(
                        "target_group must assign every restricted cell to group 0 or 1."
                    )
            log_q += float(scores[choice] - log_norm)
            if choice == 0:
                group_a.append(cell)
                counts_a[positions, residues] += 1
                clone_sum_a += cell_clone_loglik[cell]
            else:
                group_b.append(cell)
                counts_b[positions, residues] += 1
                clone_sum_b += cell_clone_loglik[cell]

        return (
            np.asarray(group_a, dtype=np.int64),
            np.asarray(group_b, dtype=np.int64),
            float(log_q),
        )

    def _collapsed_partition_log_target(
        self,
        assignments: np.ndarray,
        *,
        cell_clone_loglik: np.ndarray | None = None,
    ) -> float:
        z = reindex_assignments(np.asarray(assignments, dtype=np.int64))
        _, counts = np.unique(z, return_counts=True)
        score = len(counts) * math.log(self.alpha_)
        score += math.lgamma(self.alpha_) - math.lgamma(self.alpha_ + self.n_cells)
        score += sum(math.lgamma(float(count)) for count in counts)
        if cell_clone_loglik is None:
            cell_clone_loglik = self._cell_clone_mut_loglikelihood()
        for label in range(int(z.max()) + 1):
            members = np.flatnonzero(z == label)
            score += self._collapsed_cluster_log_score(
                members, cell_clone_loglik=cell_clone_loglik
            )
        return float(score)

    def _collapsed_cluster_log_score(
        self,
        members: np.ndarray,
        *,
        cell_clone_loglik: np.ndarray,
    ) -> float:
        members = np.asarray(members, dtype=np.int64)
        if members.ndim != 1 or members.size == 0:
            raise ValueError("a collapsed cluster must contain at least one cell.")
        score = self._collapsed_bcr_cluster_log_marginal(members)
        if self.assignment_likelihood == "joint":
            clone_scores = cell_clone_loglik[members].sum(axis=0) + self.log_clone_prior
            score += float(self._logsumexp(clone_scores[None, :], axis=1)[0])
        return float(score)

    def _collapsed_bcr_cluster_log_marginal(self, members: np.ndarray) -> float:
        prior = normalize_dirichlet_prior(self.dirichlet_prior, self.L)
        sequences = self.sequences_[members]
        score = 0.0
        for position in range(self.L):
            counts = np.bincount(sequences[:, position], minlength=4).astype(float)
            alpha = prior[position]
            score += math.lgamma(float(alpha.sum()))
            score -= math.lgamma(float(alpha.sum() + counts.sum()))
            score += sum(
                math.lgamma(float(alpha[residue] + counts[residue]))
                - math.lgamma(float(alpha[residue]))
                for residue in range(4)
            )
        return float(score)

    # ------------------------------------------------------------------
    # Shared likelihoods and updates
    # ------------------------------------------------------------------

    def current_cell_clone_assignment(self) -> np.ndarray:
        self._require_prefit()
        return self.hypercluster_to_clone_[self.assignments_]

    def current_state_cell_clone_probabilities(self) -> np.ndarray:
        """Return clone probabilities conditional on the current sampler state."""

        self._require_prefit()
        bcr = self._cell_hypercluster_bcr_loglikelihood()
        mut = self._cell_hypercluster_mut_loglikelihood()
        counts = np.bincount(self.assignments_, minlength=self.n_hyperclusters).astype(float)
        log_prob = np.log(counts[None, :] + self.eps) + bcr + mut
        prob = np.exp(log_prob - self._logsumexp(log_prob, axis=1)[:, None])
        indicator = np.eye(self.n_clones, dtype=float)[self.hypercluster_to_clone_]
        out = prob @ indicator
        return out / np.clip(out.sum(axis=1, keepdims=True), self.eps, None)

    def posterior_cell_clone_probabilities(
        self,
        *,
        burn_in: int | float = 0,
        thin: int = 1,
        use_trace: bool | None = None,
    ) -> np.ndarray:
        """Return posterior clone probabilities.

        When cell-clone draws are available, trace averaging is the default.
        Set ``use_trace=False`` for the Stage-1 current-state calculation.
        """

        has_trace = len(self.cell_clone_assignment_trace_) > 0
        if use_trace is True and not has_trace:
            raise RuntimeError("cell-clone tracking is required for trace-based probabilities.")
        if use_trace is not False and has_trace:
            return cell_clone_probabilities_from_trace(
                np.asarray(self.cell_clone_assignment_trace_, dtype=np.int64),
                n_clones=self.n_clones,
                burn_in=burn_in,
                thin=thin,
            )
        return self.current_state_cell_clone_probabilities()

    def current_state_cell_genotype_probabilities(self) -> np.ndarray:
        """Return cell genotype probabilities conditional on the current state."""

        clone_prob = self.current_state_cell_clone_probabilities()
        return clone_prob @ self._genotype_matrix().astype(float)

    def posterior_cell_genotype_probabilities(
        self,
        *,
        burn_in: int | float = 0,
        thin: int = 1,
        use_trace: bool | None = None,
    ) -> np.ndarray:
        """Return coherent posterior cell-genotype probabilities.

        With compatible cell-clone and genotype traces, each draw's clone labels
        are combined with that same draw's genotype state before averaging.
        """

        has_trace = (
            len(self.cell_clone_assignment_trace_) > 0
            and len(self.cell_clone_assignment_trace_) == len(self.genotype_state_trace_)
        )
        if use_trace is True and not has_trace:
            raise RuntimeError(
                "cell-clone and genotype-state tracking are required for a coherent posterior summary."
            )
        if use_trace is not False and has_trace:
            return coherent_cell_genotype_probabilities(
                np.asarray(self.cell_clone_assignment_trace_, dtype=np.int64),
                np.asarray(self.genotype_state_trace_),
                burn_in=burn_in,
                thin=thin,
            )
        return self.current_state_cell_genotype_probabilities()

    def posterior_hypercluster_coassignment(
        self,
        *,
        burn_in: int | float = 0,
        thin: int = 1,
    ) -> np.ndarray:
        """Return a label-invariant posterior cell co-assignment matrix."""

        if not self.assignment_trace_:
            raise RuntimeError("assignment tracking is required for co-assignment summaries.")
        return coassignment_probabilities_from_trace(
            self.assignment_trace_, burn_in=burn_in, thin=thin
        )

    def posterior_summary(
        self,
        *,
        burn_in: int | float = 0,
        thin: int = 1,
    ) -> dict[str, np.ndarray]:
        """Return canonical coherent posterior summaries from retained draws."""

        return {
            "cell_clone_probabilities": self.posterior_cell_clone_probabilities(
                burn_in=burn_in, thin=thin, use_trace=True
            ),
            "cell_genotype_probabilities": self.posterior_cell_genotype_probabilities(
                burn_in=burn_in, thin=thin, use_trace=True
            ),
            "hypercluster_coassignment": self.posterior_hypercluster_coassignment(
                burn_in=burn_in, thin=thin
            ),
        }

    def log_likelihood_components(self) -> dict[str, float]:
        rows = np.arange(self.n_cells)
        bcr = float(self._cell_hypercluster_bcr_loglikelihood()[rows, self.assignments_].sum())
        mutation = float(self._cell_hypercluster_mut_loglikelihood()[rows, self.assignments_].sum())
        return {"bcr": bcr, "mutation": mutation, "total": bcr + mutation}

    def log_likelihood(self) -> float:
        return float(self.log_likelihood_components()["total"])

    def crp_assignment_log_prior(self) -> float:
        _, counts = np.unique(self.assignments_, return_counts=True)
        out = len(counts) * math.log(self.alpha_)
        out += math.lgamma(self.alpha_) - math.lgamma(self.alpha_ + self.n_cells)
        out += sum(math.lgamma(float(count)) for count in counts)
        return float(out)

    def clone_assignment_log_prior(self) -> float:
        return float(np.sum(self.log_clone_prior[self.hypercluster_to_clone_]))

    def log_posterior(self) -> float:
        return float(
            self.log_likelihood()
            + self.crp_assignment_log_prior()
            + self.clone_assignment_log_prior()
            + self._genotype_log_prior()
            + self._extra_log_prior()
        )

    def _cell_hypercluster_bcr_loglikelihood(self) -> np.ndarray:
        return self._bcr_loglik_for_cells(self.sequences_)

    def _bcr_loglik_for_cells(self, sequences: np.ndarray) -> np.ndarray:
        log_profiles = np.log(np.clip(self.bcr_profiles_, self.eps, 1.0))
        return self.accelerator.bcr_loglik(sequences, log_profiles)

    def _cell_clone_mut_loglikelihood(self) -> np.ndarray:
        alt = self.alt_counts_.astype(float)
        ref = self.total_counts_.astype(float) - alt
        genotype = self._genotype_matrix().astype(float)
        p_obs = np.clip(self.p_obs_by_mutation_, self.eps, 1.0 - self.eps)
        p_unobs = float(np.clip(self.p_unobs_, self.eps, 1.0 - self.eps))
        present = alt * np.log(p_obs)[None, :] + ref * np.log1p(-p_obs)[None, :]
        absent = alt * np.log(p_unobs) + ref * np.log1p(-p_unobs)
        return absent.sum(axis=1)[:, None] + (present - absent) @ genotype.T

    def _cell_hypercluster_mut_loglikelihood(self) -> np.ndarray:
        return self._cell_clone_mut_loglikelihood()[:, self.hypercluster_to_clone_]

    def _new_hypercluster_mutation_loglikelihood(self) -> np.ndarray:
        return self._logsumexp(
            self._cell_clone_mut_loglikelihood() + self.log_clone_prior[None, :], axis=1
        )

    def _sample_bcr_profiles(
        self,
        assignments: np.ndarray,
        *,
        sequences: np.ndarray | None = None,
    ) -> np.ndarray:
        sequences = self.sequences_ if sequences is None else sequences
        z = np.asarray(assignments, dtype=np.int64)
        prior = normalize_dirichlet_prior(self.dirichlet_prior, sequences.shape[1])
        n_h = int(z.max()) + 1
        counts = np.zeros((n_h, sequences.shape[1], 4), dtype=float)
        positions = np.arange(sequences.shape[1], dtype=np.int64)
        np.add.at(counts, (z[:, None], positions[None, :], sequences), 1.0)
        draws = self.rng.gamma(shape=counts + prior[None, :, :], scale=1.0)
        draws = np.clip(draws, self.eps, None)
        return draws / draws.sum(axis=-1, keepdims=True)

    def _sample_prior_bcr_profiles(self, n_profiles: int) -> np.ndarray:
        prior = normalize_dirichlet_prior(self.dirichlet_prior, self.L)
        shape = np.broadcast_to(prior[None, :, :], (n_profiles, self.L, 4))
        draws = self.rng.gamma(shape=shape, scale=1.0)
        return draws / np.clip(draws.sum(axis=-1, keepdims=True), self.eps, None)

    def _cellwise_bcr_loglikelihood_for_profiles(
        self, sequences: np.ndarray, profiles: np.ndarray
    ) -> np.ndarray:
        rows = np.arange(sequences.shape[0])[:, None]
        positions = np.arange(sequences.shape[1])[None, :]
        return np.log(np.clip(profiles[rows, positions, sequences], self.eps, 1.0)).sum(axis=1)

    def _bcr_prior_predictive_loglikelihood(self, sequences: np.ndarray) -> np.ndarray:
        prior = normalize_dirichlet_prior(self.dirichlet_prior, self.L)
        probabilities = prior / prior.sum(axis=1, keepdims=True)
        return np.log(probabilities[np.arange(self.L)[None, :], sequences]).sum(axis=1)

    def _sample_hypercluster_clone_assignments(self) -> np.ndarray:
        alt_by_h, ref_by_h = self.accelerator.aggregate_counts(
            self.assignments_, self.alt_counts_, self.total_counts_, self.n_hyperclusters
        )
        p_obs = np.clip(self.p_obs_by_mutation_, self.eps, 1.0 - self.eps)
        p_unobs = float(np.clip(self.p_unobs_, self.eps, 1.0 - self.eps))
        return self.accelerator.sample_clone_labels(
            alt_by_h,
            ref_by_h,
            self._genotype_matrix(),
            np.log(p_obs),
            np.log1p(-p_obs),
            np.log(p_unobs),
            np.log1p(-p_unobs),
            self.log_clone_prior,
            self.rng.random(self.n_hyperclusters),
        )

    def _resize_hypercluster_clones(
        self, old_assignments: np.ndarray, new_assignments: np.ndarray
    ) -> np.ndarray:
        old_h = self.hypercluster_to_clone_.copy()
        new_n_h = int(new_assignments.max()) + 1
        contingency = np.zeros((new_n_h, int(old_assignments.max()) + 1), dtype=np.int64)
        np.add.at(contingency, (new_assignments, old_assignments), 1)
        old_mode = contingency.argmax(axis=1)
        out = old_h[np.clip(old_mode, 0, old_h.size - 1)].copy()
        empty = contingency.sum(axis=1) == 0
        if np.any(empty):
            out[empty] = self.rng.choice(
                self.n_clones, size=int(empty.sum()), p=self.clone_prior
            )
        return out.astype(np.int64)

    def _aggregate_mutation_counts_by_clone(self) -> tuple[np.ndarray, np.ndarray]:
        return self.accelerator.aggregate_counts(
            self.current_cell_clone_assignment(),
            self.alt_counts_,
            self.total_counts_,
            self.n_clones,
        )

    def _resample_observation_probabilities(self) -> None:
        genotype = self._genotype_matrix().astype(bool)
        alt_by_clone, ref_by_clone = self._aggregate_mutation_counts_by_clone()
        prior = self._p_obs_beta_prior_matrix
        successes = np.sum(np.where(genotype, alt_by_clone, 0.0), axis=0)
        failures = np.sum(np.where(genotype, ref_by_clone, 0.0), axis=0)
        self.p_obs_by_mutation_ = np.clip(
            self.rng.beta(prior[:, 0] + successes, prior[:, 1] + failures),
            self.eps,
            1.0 - self.eps,
        )
        absent_success = float(np.sum(np.where(~genotype, alt_by_clone, 0.0)))
        absent_failure = float(np.sum(np.where(~genotype, ref_by_clone, 0.0)))
        a, b = self.p_unobs_beta_prior
        self.p_unobs_ = float(
            np.clip(
                self.rng.beta(a + absent_success, b + absent_failure),
                self.eps,
                1.0 - self.eps,
            )
        )

    def _initialize_p_obs(self) -> np.ndarray:
        if self.p_obs_init is None:
            prior = self._p_obs_beta_prior_matrix
            return np.clip(
                self.rng.beta(prior[:, 0], prior[:, 1]), self.eps, 1.0 - self.eps
            )
        value = np.asarray(self.p_obs_init, dtype=float)
        if value.ndim == 0:
            value = np.full(self.n_snv, float(value))
        if value.shape != (self.n_snv,) or np.any((value <= 0) | (value >= 1)):
            raise ValueError("p_obs_init must be scalar or shape (n_snv,) with values in (0,1).")
        return value.copy()

    def _initialize_p_unobs(self) -> float:
        if self.p_unobs_init is not None:
            return float(self.p_unobs_init)
        a, b = self.p_unobs_beta_prior
        return float(np.clip(self.rng.beta(a, b), self.eps, 1.0 - self.eps))

    # ------------------------------------------------------------------
    # Tracking and serialization
    # ------------------------------------------------------------------

    def _clear_traces(self) -> None:
        for name in (
            "log_likelihood_trace_",
            "bcr_log_likelihood_trace_",
            "mutation_log_likelihood_trace_",
            "crp_logprior_trace_",
            "clone_assignment_logprior_trace_",
            "genotype_logprior_trace_",
            "log_posterior_trace_",
            "n_hyperclusters_trace_",
            "alpha_trace_",
            "p_unobs_trace_",
            "p_obs_mean_trace_",
            "assignment_trace_",
            "cell_clone_assignment_trace_",
            "genotype_state_trace_",
            "p_obs_by_mutation_trace_",
        ):
            getattr(self, name).clear()
        self.split_merge_attempts_ = 0
        self.split_merge_accepts_ = 0
        self.split_attempts_ = 0
        self.split_accepts_ = 0
        self.merge_attempts_ = 0
        self.merge_accepts_ = 0
        self._clear_subclass_traces()

    def _append_tracking(self, config: TrackingConfig) -> None:
        components = self.log_likelihood_components()
        if config.log_likelihood:
            self.log_likelihood_trace_.append(components["total"])
            self.bcr_log_likelihood_trace_.append(components["bcr"])
            self.mutation_log_likelihood_trace_.append(components["mutation"])
        crp = self.crp_assignment_log_prior()
        clone = self.clone_assignment_log_prior()
        genotype = self._genotype_log_prior()
        if config.log_posterior:
            self.crp_logprior_trace_.append(crp)
            self.clone_assignment_logprior_trace_.append(clone)
            self.genotype_logprior_trace_.append(genotype)
            self.log_posterior_trace_.append(
                components["total"] + crp + clone + genotype + self._extra_log_prior()
            )
        self.n_hyperclusters_trace_.append(self.n_hyperclusters)
        self.alpha_trace_.append(self.alpha_)
        self.p_unobs_trace_.append(self.p_unobs_)
        self.p_obs_mean_trace_.append(float(np.mean(self.p_obs_by_mutation_)))
        if config.assignments:
            self.assignment_trace_.append(self.assignments_.copy())
        if config.cell_clone_assignments:
            self.cell_clone_assignment_trace_.append(self.current_cell_clone_assignment().copy())
        if config.genotype_state:
            self.genotype_state_trace_.append(self._genotype_state_for_tracking().copy())
        if config.observation_probabilities:
            self.p_obs_by_mutation_trace_.append(self.p_obs_by_mutation_.copy())
        self._append_subclass_tracking(config)

    def to_dict(self) -> dict[str, Any]:
        self._require_prefit()
        genotype = self._genotype_matrix().copy()
        out: dict[str, Any] = {
            "assignments": self.assignments_.copy(),
            "bcr_profiles": self.bcr_profiles_.copy(),
            "hypercluster_to_clone": self.hypercluster_to_clone_.copy(),
            "cell_clone_assignment": self.current_cell_clone_assignment().copy(),
            "cell_clone_assignments": self.current_cell_clone_assignment().copy(),
            "genotype_matrix": genotype,
            "mutation_profile": genotype.copy(),
            "p_obs_by_mutation": self.p_obs_by_mutation_.copy(),
            "p_unobs": float(self.p_unobs_),
            "alpha": float(self.alpha_),
            "n_hyperclusters": self.n_hyperclusters,
            "n_clones": self.n_clones,
            "clone_prior": self.clone_prior.copy(),
            "accelerator": self.accelerator_name,
            "numba_enabled": self.accelerator.enabled,
            "assignment_likelihood": self.assignment_likelihood,
            "new_cluster_likelihood": self.new_cluster_likelihood,
            "log_likelihood": self.log_likelihood(),
            "bcr_log_likelihood": self.log_likelihood_components()["bcr"],
            "mutation_log_likelihood": self.log_likelihood_components()["mutation"],
            "log_posterior": self.log_posterior(),
            "log_likelihood_trace": np.asarray(self.log_likelihood_trace_, dtype=float),
            "bcr_log_likelihood_trace": np.asarray(self.bcr_log_likelihood_trace_, dtype=float),
            "mutation_log_likelihood_trace": np.asarray(self.mutation_log_likelihood_trace_, dtype=float),
            "log_posterior_trace": np.asarray(self.log_posterior_trace_, dtype=float),
            "n_hyperclusters_trace": np.asarray(self.n_hyperclusters_trace_, dtype=int),
            "alpha_trace": np.asarray(self.alpha_trace_, dtype=float),
            "p_unobs_trace": np.asarray(self.p_unobs_trace_, dtype=float),
            "p_obs_mean_trace": np.asarray(self.p_obs_mean_trace_, dtype=float),
            "assignment_trace": np.asarray(self.assignment_trace_, dtype=object),
            "cell_clone_assignment_trace": np.asarray(
                self.cell_clone_assignment_trace_, dtype=np.int64
            ),
            "genotype_state_trace": np.asarray(self.genotype_state_trace_),
            "p_obs_by_mutation_trace": np.asarray(self.p_obs_by_mutation_trace_, dtype=float),
            "split_merge_diagnostics": self.split_merge_diagnostics(),
            "current_state_cell_clone_probabilities": self.current_state_cell_clone_probabilities(),
            "current_state_cell_genotype_probabilities": self.current_state_cell_genotype_probabilities(),
        }
        if self.cell_clone_assignment_trace_:
            out["posterior_cell_clone_probabilities"] = self.posterior_cell_clone_probabilities()
        if (
            self.cell_clone_assignment_trace_
            and len(self.cell_clone_assignment_trace_) == len(self.genotype_state_trace_)
        ):
            out["posterior_cell_genotype_probabilities"] = (
                self.posterior_cell_genotype_probabilities()
            )
        if self.assignment_trace_:
            out["posterior_hypercluster_coassignment"] = (
                self.posterior_hypercluster_coassignment()
            )
        out.update(self._extra_to_dict())
        return out

    # ------------------------------------------------------------------
    # Properties and helpers
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        self._require_data()
        return int(self.sequences_.shape[0])

    @property
    def L(self) -> int:
        self._require_data()
        return int(self.sequences_.shape[1])

    @property
    def n_snv(self) -> int:
        self._require_data()
        return int(self.alt_counts_.shape[1])

    @property
    def n_hyperclusters(self) -> int:
        self._require_prefit()
        return int(self.assignments_.max()) + 1

    def _sample_rows(self, log_probs: np.ndarray) -> np.ndarray:
        return self.accelerator.sample_rows(
            np.asarray(log_probs, dtype=float), self.rng.random(log_probs.shape[0])
        )

    @staticmethod
    def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
        maximum = np.max(x, axis=axis, keepdims=True)
        finite = np.isfinite(maximum)
        shifted = np.where(finite, x - maximum, -np.inf)
        summed = np.sum(np.exp(shifted), axis=axis)
        return np.squeeze(maximum, axis=axis) + np.log(summed)

    def _require_data(self) -> None:
        if self.sequences_ is None or self.alt_counts_ is None or self.total_counts_ is None:
            raise RuntimeError("Call prefit before using the model.")

    def _require_prefit(self) -> None:
        self._require_data()
        if self.assignments_ is None or self.bcr_profiles_ is None:
            raise RuntimeError("Call prefit before using the model.")

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _initialize_genotype_state(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _sample_genotype_state(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _genotype_matrix(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _genotype_log_prior(self) -> float:
        raise NotImplementedError

    def _extra_log_prior(self) -> float:
        return 0.0

    def _update_genotype_prior_parameters(self) -> None:
        return None

    def _genotype_state_for_tracking(self) -> np.ndarray:
        return self._genotype_matrix()

    def _extra_to_dict(self) -> dict[str, Any]:
        return {}

    def _clear_subclass_traces(self) -> None:
        return None

    def _append_subclass_tracking(self, config: TrackingConfig) -> None:
        return None
