from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
import gzip
import math
from pathlib import Path
import pickle
from typing import Any, Literal, Mapping

import numpy as np

from ._numba_accelerator import Accelerator, Backend
from .config import CloneMixtureConfig, SplitMergeConfig, TrackingConfig
from .state import StateView
from .utils.assignments import InitSpec, initialize_assignments, reindex_assignments
from .utils.posterior import (
    cell_clone_probabilities_from_trace,
    coassignment_probabilities_from_trace,
    coherent_cell_genotype_probabilities,
    partition_medoid_from_trace,
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
_CHECKPOINT_PACKAGE_VERSION = "0.4.0"


@dataclass(slots=True)
class _ClusterStats:
    members: np.ndarray
    bcr_counts: np.ndarray
    clone_loglik_sum: np.ndarray
    score: float


@dataclass(slots=True)
class _SplitMergeCache:
    cell_clone_loglik: np.ndarray
    prior: np.ndarray
    prior_sum: np.ndarray
    clusters: dict[int, _ClusterStats]


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
        alpha_prior: tuple[float, float] = (1.0, 1.0),
        sample_alpha: bool | None = None,
        dirichlet_prior: float | np.ndarray = 0.5,
        p_obs_beta_prior: tuple[float, float] | np.ndarray = (50.0, 50.0),
        p_unobs_beta_prior: tuple[float, float] = (1.0, 999.0),
        p_obs_init: float | np.ndarray | None = None,
        p_unobs_init: float | None = None,
        clone_prior: np.ndarray | None = None,
        clone_mixture: CloneMixtureConfig | None = None,
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
        if len(alpha_prior) != 2 or min(alpha_prior) <= 0:
            raise ValueError("alpha_prior must contain positive shape and rate values.")
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
        self.alpha_prior = tuple(float(x) for x in alpha_prior)
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
        self.clone_mixture = clone_mixture or CloneMixtureConfig()
        self.sample_alpha = (
            self.clone_mixture.enabled if sample_alpha is None else bool(sample_alpha)
        )

        self.sequences_: np.ndarray | None = None
        self.seqs_: np.ndarray | None = None  # legacy alias
        self.alt_counts_: np.ndarray | None = None
        self.total_counts_: np.ndarray | None = None
        self.assignments_: np.ndarray | None = None
        self.bcr_profiles_: np.ndarray | None = None
        self.hypercluster_to_clone_: np.ndarray | None = None
        self.cell_clone_assignments_: np.ndarray | None = None
        self.hypercluster_clone_proportions_: np.ndarray | None = None
        self.admixture_mass_: np.ndarray | None = None
        self.residual_clone_proportions_: np.ndarray | None = None
        self.mixture_active_: np.ndarray | None = None
        self.mixture_presence_rate_: float | None = None
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
        self.dominant_clone_trace_: list[np.ndarray] = []
        self.hypercluster_clone_proportions_trace_: list[np.ndarray] = []
        self.admixture_mass_trace_: list[np.ndarray] = []
        self.mixture_active_trace_: list[np.ndarray] = []
        self.effective_clone_count_trace_: list[np.ndarray] = []
        self.admixture_entropy_trace_: list[np.ndarray] = []
        self.dominant_fraction_trace_: list[np.ndarray] = []

        self.split_merge_attempts_: int = 0
        self.split_merge_accepts_: int = 0
        self.split_attempts_: int = 0
        self.split_accepts_: int = 0
        self.merge_attempts_: int = 0
        self.merge_accepts_: int = 0
        self.split_merge_cache_builds_: int = 0
        self.split_merge_cache_reuses_: int = 0
        self.adaptive_split_probability_: float = 0.5
        self._adaptive_initialized_: bool = False
        self._adaptive_window_split_attempts: int = 0
        self._adaptive_window_split_accepts: int = 0
        self._adaptive_window_merge_attempts: int = 0
        self._adaptive_window_merge_accepts: int = 0
        self._split_merge_cache_: _SplitMergeCache | None = None
        self._iterations_completed_: int = 0

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
        if not self.clone_mixture.enabled:
            # Preserve the exact v0.2.2 random-number order when mixtures are off.
            self.hypercluster_to_clone_ = self.rng.choice(
                self.n_clones,
                size=self.n_hyperclusters,
                replace=True,
                p=self.clone_prior,
            ).astype(np.int64)
            self.p_obs_by_mutation_ = self._initialize_p_obs()
            self.p_unobs_ = self._initialize_p_unobs()
            self._initialize_genotype_state()
            self._sync_disabled_clone_mixture_state()
        else:
            self._validate_clone_mixture_support()
            self.p_obs_by_mutation_ = self._initialize_p_obs()
            self.p_unobs_ = self._initialize_p_unobs()
            self._initialize_genotype_state()
            self._initialize_clone_mixture_state()
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
        update_cell_clones: bool | None = None,
        update_clone_mixtures: bool | None = None,
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
        update_cell_clones = (
            update_hypercluster_clones if update_cell_clones is None else bool(update_cell_clones)
        )
        update_clone_mixtures = (
            update_hypercluster_clones
            if update_clone_mixtures is None
            else bool(update_clone_mixtures)
        )

        for iteration in range(n_iter):
            if self.clone_mixture.enabled:
                # Partially collapsed Gibbs order: z, B, y, mixture, G, theta.
                if update_assignments:
                    self.assignment_sweep(
                        assignment_sampler,
                        split_merge_config=split_merge_config,
                    )
                if update_bcr_profiles and not update_assignments:
                    self.bcr_profiles_ = self._sample_bcr_profiles(self.assignments_)
                if update_cell_clones:
                    self._sample_cell_clone_assignments()
                if update_clone_mixtures:
                    self._sample_clone_mixture_state()
            else:
                # Exact v0.2.2 transition order and random-number stream.
                if update_hypercluster_clones:
                    self.hypercluster_to_clone_ = self._sample_hypercluster_clone_assignments()
                if update_bcr_profiles:
                    self.bcr_profiles_ = self._sample_bcr_profiles(self.assignments_)
                if update_assignments:
                    self.assignment_sweep(
                        assignment_sampler,
                        split_merge_config=split_merge_config,
                    )
                self._sync_disabled_clone_mixture_state()
            if update_genotypes:
                self._sample_genotype_state()
                self._update_genotype_prior_parameters()
            if update_observation_probabilities:
                self._resample_observation_probabilities()
            if self.sample_alpha:
                self._resample_alpha()
            self._iterations_completed_ += 1
            if self._iterations_completed_ % tracking.every == 0:
                self._append_tracking(tracking)
            if verbose and (
                self._iterations_completed_ % progress_every == 0 or iteration == 0
            ):
                print(
                    f"iter={self._iterations_completed_} K={self.n_hyperclusters} "
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
        if self.clone_mixture.enabled and sampler == "split_merge":
            raise ValueError(
                "assignment_sampler='split_merge' is not defined for clone mixtures; "
                "use 'approximate' or 'sequential'."
            )
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
            self._split_merge_cache_ = (
                self._build_split_merge_cache()
                if config.cache_sufficient_statistics
                else None
            )
            try:
                for _ in range(config.proposals_per_sweep):
                    self.split_merge_assignment_sweep(config=config)
            finally:
                self._split_merge_cache_ = None
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
        if self.clone_mixture.enabled:
            self._resize_clone_mixture_state(
                z_old,
                z_new,
                dominant_hint=new_h_to_clone,
            )
        else:
            self.hypercluster_to_clone_ = new_h_to_clone
        self.bcr_profiles_ = self._sample_bcr_profiles(z_new)
        if self.clone_mixture.enabled:
            self._enforce_pure_clone_assignments()

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
                if self.clone_mixture.enabled:
                    self._subset_clone_mixture_rows(active_old)
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
                if self.clone_mixture.enabled:
                    scores += self._mixture_log_marginal_for_cell(clone_scores)
                else:
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
                new_dominant = int(self.rng.choice(self.n_clones, p=self.clone_prior))
                self.hypercluster_to_clone_ = np.append(
                    self.hypercluster_to_clone_, new_dominant
                ).astype(np.int64)
                if self.clone_mixture.enabled:
                    self._append_clone_mixture_cluster(new_dominant)
                new_profile = self._sample_bcr_profiles(np.array([0], dtype=np.int64), sequences=self.sequences_[cell : cell + 1])
                self.bcr_profiles_ = np.concatenate([self.bcr_profiles_, new_profile], axis=0)
            else:
                self.assignments_[cell] = choice
        self.assignments_ = reindex_assignments(self.assignments_)
        self.bcr_profiles_ = self._sample_bcr_profiles(self.assignments_)
        if not self.clone_mixture.enabled:
            self._sync_disabled_clone_mixture_state()
        else:
            self._enforce_pure_clone_assignments()

    def split_merge_assignment_sweep(
        self,
        *,
        config: SplitMergeConfig | None = None,
    ) -> bool:
        """Run one collapsed mutation-informed split-or-merge proposal.

        Version 0.2.1 can cache the cell/cluster sufficient statistics used by
        repeated proposals. Under ``anchor_strategy="adaptive"`` split and
        merge anchor families are scheduled separately; their state-dependent
        selection probabilities are included in the Metropolis-Hastings ratio.
        """

        self._require_prefit()
        if self.n_cells < 2:
            return False
        config = config or SplitMergeConfig()
        if config.anchor_strategy == "adaptive" and not self._adaptive_initialized_:
            self.adaptive_split_probability_ = float(config.initial_split_probability)
            self._adaptive_initialized_ = True

        cache = self._split_merge_cache_
        if cache is None:
            cache = self._build_split_merge_cache()
            if config.cache_sufficient_statistics:
                self._split_merge_cache_ = cache
        else:
            self.split_merge_cache_reuses_ += 1

        old_assignments = self.assignments_.copy()
        if config.max_restricted_cells is not None:
            _options, _split_total, _merge_total = self._bounded_pair_summary(
                old_assignments, max_cells=config.max_restricted_cells
            )
            if _split_total + _merge_total <= 0.0:
                return False
        first, second, anchor_log_forward = self._select_split_merge_anchors(
            old_assignments, config=config
        )
        label_first = int(old_assignments[first])
        label_second = int(old_assignments[second])
        cell_clone_loglik = cache.cell_clone_loglik

        self.split_merge_attempts_ += 1
        if label_first == label_second:
            move = "split"
            self.split_attempts_ += 1
            members = cache.clusters[label_first].members
            remaining = members[(members != first) & (members != second)]
            order = self.rng.permutation(remaining)
            group_a, group_b, log_q_forward, stats_a, stats_b = (
                self._restricted_split_allocation(
                    first,
                    second,
                    order,
                    cell_clone_loglik=cell_clone_loglik,
                    prior=cache.prior,
                    prior_sum=cache.prior_sum,
                )
            )
            candidate = old_assignments.copy()
            new_label = int(old_assignments.max()) + 1
            candidate[group_a] = label_first
            candidate[group_b] = new_label
            candidate = reindex_assignments(candidate)
            old_cluster_score = cache.clusters[label_first].score
            split_score = stats_a.score + stats_b.score
            log_target_ratio = (
                math.log(self.alpha_)
                + math.lgamma(float(group_a.size))
                + math.lgamma(float(group_b.size))
                - math.lgamma(float(members.size))
                + split_score
                - old_cluster_score
            )
            anchor_log_reverse = self._anchor_log_probability(
                candidate,
                move="merge",
                config=config,
            )
            log_acceptance = (
                log_target_ratio
                - log_q_forward
                + anchor_log_reverse
                - anchor_log_forward
            )
        else:
            move = "merge"
            self.merge_attempts_ += 1
            stats_a_old = cache.clusters[label_first]
            stats_b_old = cache.clusters[label_second]
            members_a = stats_a_old.members
            members_b = stats_b_old.members
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
            _, _, log_q_reverse, _, _ = self._restricted_split_allocation(
                first,
                second,
                order,
                cell_clone_loglik=cell_clone_loglik,
                prior=cache.prior,
                prior_sum=cache.prior_sum,
                target_group=target_group,
            )
            candidate = old_assignments.copy()
            candidate[candidate == label_second] = label_first
            candidate = reindex_assignments(candidate)
            merged_stats = self._combine_cluster_stats(stats_a_old, stats_b_old)
            separate_score = stats_a_old.score + stats_b_old.score
            merged_score = merged_stats.score
            log_target_ratio = (
                -math.log(self.alpha_)
                + math.lgamma(float(merged_stats.members.size))
                - math.lgamma(float(members_a.size))
                - math.lgamma(float(members_b.size))
                + merged_score
                - separate_score
            )
            anchor_log_reverse = self._anchor_log_probability(
                candidate,
                move="split",
                config=config,
            )
            log_acceptance = (
                log_target_ratio
                + log_q_reverse
                + anchor_log_reverse
                - anchor_log_forward
            )

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
            if config.cache_sufficient_statistics:
                self._split_merge_cache_ = self._build_split_merge_cache()

        self._record_adaptive_proposal(move=move, accepted=accepted, config=config)
        return accepted

    def split_merge_diagnostics(self) -> dict[str, float | int]:
        """Return global proposal, cache, and adaptive-scheduler diagnostics."""

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
            "adaptive_split_probability": float(self.adaptive_split_probability_),
            "cache_builds": self.split_merge_cache_builds_,
            "cache_reuses": self.split_merge_cache_reuses_,
        }

    def _build_split_merge_cache(self) -> _SplitMergeCache:
        prior = normalize_dirichlet_prior(self.dirichlet_prior, self.L)
        prior_sum = prior.sum(axis=1)
        cell_clone_loglik = self._cell_clone_mut_loglikelihood()
        clusters: dict[int, _ClusterStats] = {}
        for label in range(self.n_hyperclusters):
            members = np.flatnonzero(self.assignments_ == label)
            clusters[label] = self._cluster_stats_from_members(
                members,
                cell_clone_loglik=cell_clone_loglik,
                prior=prior,
            )
        self.split_merge_cache_builds_ += 1
        return _SplitMergeCache(
            cell_clone_loglik=cell_clone_loglik,
            prior=prior,
            prior_sum=prior_sum,
            clusters=clusters,
        )

    def _cluster_stats_from_members(
        self,
        members: np.ndarray,
        *,
        cell_clone_loglik: np.ndarray,
        prior: np.ndarray,
    ) -> _ClusterStats:
        members = np.asarray(members, dtype=np.int64)
        if members.ndim != 1 or members.size == 0:
            raise ValueError("a collapsed cluster must contain at least one cell.")
        counts = np.zeros((self.L, 4), dtype=np.int64)
        sequences = self.sequences_[members]
        for position in range(self.L):
            counts[position] = np.bincount(
                sequences[:, position], minlength=4
            ).astype(np.int64)
        clone_sum = cell_clone_loglik[members].sum(axis=0)
        score = self._collapsed_score_from_stats(counts, clone_sum, prior=prior)
        return _ClusterStats(
            members=members.copy(),
            bcr_counts=counts,
            clone_loglik_sum=clone_sum,
            score=score,
        )

    def _combine_cluster_stats(
        self,
        left: _ClusterStats,
        right: _ClusterStats,
    ) -> _ClusterStats:
        members = np.concatenate([left.members, right.members])
        counts = left.bcr_counts + right.bcr_counts
        clone_sum = left.clone_loglik_sum + right.clone_loglik_sum
        prior = normalize_dirichlet_prior(self.dirichlet_prior, self.L)
        score = self._collapsed_score_from_stats(counts, clone_sum, prior=prior)
        return _ClusterStats(members, counts, clone_sum, score)

    def _collapsed_score_from_stats(
        self,
        bcr_counts: np.ndarray,
        clone_loglik_sum: np.ndarray,
        *,
        prior: np.ndarray,
    ) -> float:
        score = 0.0
        for position in range(self.L):
            counts = bcr_counts[position]
            alpha = prior[position]
            score += math.lgamma(float(alpha.sum()))
            score -= math.lgamma(float(alpha.sum() + counts.sum()))
            for residue in range(4):
                score += math.lgamma(float(alpha[residue] + counts[residue]))
                score -= math.lgamma(float(alpha[residue]))
        if self.assignment_likelihood == "joint":
            score += float(
                self._logsumexp(
                    (clone_loglik_sum + self.log_clone_prior)[None, :], axis=1
                )[0]
            )
        return float(score)

    @staticmethod
    def _bounded_pair_summary(
        assignments: np.ndarray, *, max_cells: int
    ) -> tuple[list[tuple[str, int, int | None, float]], float, float]:
        labels, counts = np.unique(assignments, return_counts=True)
        options: list[tuple[str, int, int | None, float]] = []
        split_total = 0.0
        merge_total = 0.0
        for label, count in zip(labels, counts):
            if 2 <= count <= max_cells:
                weight = float(count * (count - 1) // 2)
                options.append(("split", int(label), None, weight))
                split_total += weight
        for left in range(labels.size - 1):
            for right in range(left + 1, labels.size):
                if counts[left] + counts[right] <= max_cells:
                    weight = float(counts[left] * counts[right])
                    options.append(
                        ("merge", int(labels[left]), int(labels[right]), weight)
                    )
                    merge_total += weight
        return options, split_total, merge_total

    @staticmethod
    def _pair_type_counts(assignments: np.ndarray) -> tuple[int, int]:
        _, counts = np.unique(assignments, return_counts=True)
        total = int(assignments.size * (assignments.size - 1) // 2)
        same = int(np.sum(counts * (counts - 1) // 2))
        return same, total - same

    def _effective_split_probability(
        self,
        assignments: np.ndarray,
        *,
        config: SplitMergeConfig,
    ) -> float:
        same, cross = self._pair_type_counts(assignments)
        if same == 0:
            return 0.0
        if cross == 0:
            return 1.0
        return float(self.adaptive_split_probability_)

    def _anchor_log_probability(
        self,
        assignments: np.ndarray,
        *,
        move: Literal["split", "merge"],
        config: SplitMergeConfig,
    ) -> float:
        if config.anchor_strategy == "uniform_pair":
            if config.max_restricted_cells is None:
                total_pairs = assignments.size * (assignments.size - 1) // 2
                return -math.log(float(total_pairs))
            _options, split_total, merge_total = self._bounded_pair_summary(
                assignments, max_cells=config.max_restricted_cells
            )
            family_total = split_total if move == "split" else merge_total
            total_pairs = split_total + merge_total
            if family_total <= 0.0 or total_pairs <= 0.0:
                return -np.inf
            return -math.log(float(total_pairs))
        same, cross = self._pair_type_counts(assignments)
        p_split = self._effective_split_probability(assignments, config=config)
        if move == "split":
            if same < 1 or p_split <= 0.0:
                return -np.inf
            return math.log(p_split) - math.log(float(same))
        if cross < 1 or p_split >= 1.0:
            return -np.inf
        return math.log1p(-p_split) - math.log(float(cross))

    def _select_split_merge_anchors(
        self,
        assignments: np.ndarray,
        *,
        config: SplitMergeConfig,
    ) -> tuple[int, int, float]:
        if config.anchor_strategy == "uniform_pair":
            if config.max_restricted_cells is None:
                first, second = self.rng.choice(self.n_cells, size=2, replace=False)
                log_probability = self._anchor_log_probability(
                    assignments,
                    move=(
                        "split"
                        if assignments[first] == assignments[second]
                        else "merge"
                    ),
                    config=config,
                )
                return int(first), int(second), float(log_probability)

            options, split_total, merge_total = self._bounded_pair_summary(
                assignments, max_cells=config.max_restricted_cells
            )
            if not options:
                raise RuntimeError("no bounded split/merge anchor pairs are available")
            weights = np.asarray([option[3] for option in options], dtype=float)
            move, left_label, right_label, _weight = options[
                self._weighted_index(weights)
            ]
            if move == "split":
                members = np.flatnonzero(assignments == left_label)
                first, second = self.rng.choice(members, size=2, replace=False)
            else:
                first = int(
                    self.rng.choice(np.flatnonzero(assignments == left_label))
                )
                second = int(
                    self.rng.choice(np.flatnonzero(assignments == right_label))
                )
            return (
                int(first),
                int(second),
                float(-math.log(split_total + merge_total)),
            )

        labels, counts = np.unique(assignments, return_counts=True)
        p_split = self._effective_split_probability(assignments, config=config)
        request_split = bool(self.rng.random() < p_split)
        if request_split:
            pair_weights = counts * (counts - 1) // 2
            chosen_label = int(
                labels[self._weighted_index(pair_weights.astype(float))]
            )
            members = np.flatnonzero(assignments == chosen_label)
            first, second = self.rng.choice(members, size=2, replace=False)
            move = "split"
        else:
            cluster_pairs: list[tuple[int, int]] = []
            weights: list[float] = []
            for left in range(labels.size - 1):
                for right in range(left + 1, labels.size):
                    cluster_pairs.append((int(labels[left]), int(labels[right])))
                    weights.append(float(counts[left] * counts[right]))
            pair_index = self._weighted_index(np.asarray(weights, dtype=float))
            label_left, label_right = cluster_pairs[pair_index]
            first = int(self.rng.choice(np.flatnonzero(assignments == label_left)))
            second = int(self.rng.choice(np.flatnonzero(assignments == label_right)))
            move = "merge"
        return (
            int(first),
            int(second),
            float(self._anchor_log_probability(assignments, move=move, config=config)),
        )

    def _weighted_index(self, weights: np.ndarray) -> int:
        weights = np.asarray(weights, dtype=float)
        total = float(weights.sum())
        if weights.ndim != 1 or weights.size == 0 or total <= 0.0:
            raise ValueError("weighted selection requires positive one-dimensional weights.")
        threshold = float(self.rng.random()) * total
        running = 0.0
        for index, weight in enumerate(weights):
            running += float(weight)
            if running > threshold:
                return int(index)
        return int(weights.size - 1)

    def _record_adaptive_proposal(
        self,
        *,
        move: Literal["split", "merge"],
        accepted: bool,
        config: SplitMergeConfig,
    ) -> None:
        if config.anchor_strategy != "adaptive":
            return
        if move == "split":
            self._adaptive_window_split_attempts += 1
            self._adaptive_window_split_accepts += int(accepted)
        else:
            self._adaptive_window_merge_attempts += 1
            self._adaptive_window_merge_accepts += int(accepted)
        window_attempts = (
            self._adaptive_window_split_attempts
            + self._adaptive_window_merge_attempts
        )
        if window_attempts < config.adaptation_interval:
            return
        if config.adapt_until is not None and self.split_merge_attempts_ > config.adapt_until:
            return
        split_rate = (
            self._adaptive_window_split_accepts + 0.5
        ) / (self._adaptive_window_split_attempts + 1.0)
        merge_rate = (
            self._adaptive_window_merge_accepts + 0.5
        ) / (self._adaptive_window_merge_attempts + 1.0)
        probability = float(
            np.clip(
                self.adaptive_split_probability_,
                config.min_split_probability,
                config.max_split_probability,
            )
        )
        logit = math.log(probability) - math.log1p(-probability)
        logit += config.adaptation_step * (merge_rate - split_rate)
        updated = 1.0 / (1.0 + math.exp(-logit))
        self.adaptive_split_probability_ = float(
            np.clip(
                updated,
                config.min_split_probability,
                config.max_split_probability,
            )
        )
        self._adaptive_window_split_attempts = 0
        self._adaptive_window_split_accepts = 0
        self._adaptive_window_merge_attempts = 0
        self._adaptive_window_merge_accepts = 0

    def _restricted_split_allocation(
        self,
        first: int,
        second: int,
        order: np.ndarray,
        *,
        cell_clone_loglik: np.ndarray,
        prior: np.ndarray,
        prior_sum: np.ndarray,
        target_group: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, _ClusterStats, _ClusterStats]:
        """Sample or score one sequential restricted-Gibbs split allocation."""

        def initial_stats(cell: int) -> tuple[np.ndarray, np.ndarray]:
            counts = np.zeros((self.L, 4), dtype=np.int64)
            counts[np.arange(self.L), self.sequences_[cell]] = 1
            return counts, cell_clone_loglik[cell].copy()

        group_a: list[int] = [int(first)]
        group_b: list[int] = [int(second)]
        counts_a, clone_sum_a = initial_stats(int(first))
        counts_b, clone_sum_b = initial_stats(int(second))
        log_q = 0.0
        positions = np.arange(self.L)

        for raw_cell in order:
            cell = int(raw_cell)
            residues = self.sequences_[cell]
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

        group_a_array = np.asarray(group_a, dtype=np.int64)
        group_b_array = np.asarray(group_b, dtype=np.int64)
        stats_a = _ClusterStats(
            group_a_array,
            counts_a,
            clone_sum_a,
            self._collapsed_score_from_stats(counts_a, clone_sum_a, prior=prior),
        )
        stats_b = _ClusterStats(
            group_b_array,
            counts_b,
            clone_sum_b,
            self._collapsed_score_from_stats(counts_b, clone_sum_b, prior=prior),
        )
        return group_a_array, group_b_array, float(log_q), stats_a, stats_b

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
        prior = normalize_dirichlet_prior(self.dirichlet_prior, self.L)
        for label in range(int(z.max()) + 1):
            members = np.flatnonzero(z == label)
            score += self._cluster_stats_from_members(
                members, cell_clone_loglik=cell_clone_loglik, prior=prior
            ).score
        return float(score)

    def _collapsed_cluster_log_score(
        self,
        members: np.ndarray,
        *,
        cell_clone_loglik: np.ndarray,
    ) -> float:
        prior = normalize_dirichlet_prior(self.dirichlet_prior, self.L)
        return self._cluster_stats_from_members(
            members, cell_clone_loglik=cell_clone_loglik, prior=prior
        ).score

    def _collapsed_bcr_cluster_log_marginal(self, members: np.ndarray) -> float:
        prior = normalize_dirichlet_prior(self.dirichlet_prior, self.L)
        dummy = np.zeros(self.n_clones, dtype=float)
        stats = self._cluster_stats_from_members(
            members,
            cell_clone_loglik=np.zeros((self.n_cells, self.n_clones), dtype=float),
            prior=prior,
        )
        if self.assignment_likelihood == "joint":
            clone_component = float(
                self._logsumexp(
                    (dummy + self.log_clone_prior)[None, :], axis=1
                )[0]
            )
            return float(stats.score - clone_component)
        return float(stats.score)


    # ------------------------------------------------------------------
    # Shared likelihoods and updates
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Dominant-clone sparse-admixture model
    # ------------------------------------------------------------------

    @staticmethod
    def _log_beta_function(a: float, b: float) -> float:
        return float(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))

    @staticmethod
    def _safe_log_probabilities(probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=float)
        out = np.full(probabilities.shape, -np.inf, dtype=float)
        positive = probabilities > 0.0
        out[positive] = np.log(probabilities[positive])
        return out

    def _validate_clone_mixture_support(self) -> None:
        if self.clone_mixture.residual_base == "tree_distance":
            self._residual_base_weights(0)

    def _tree_distance_residual_weights(self, dominant: int) -> np.ndarray:
        raise ValueError(
            "residual_base='tree_distance' is supported only by CactriTree."
        )

    def _residual_base_weights(self, dominant: int) -> np.ndarray:
        if not 0 <= int(dominant) < self.n_clones:
            raise ValueError("dominant clone is out of range.")
        if self.n_clones == 1:
            return np.zeros(1, dtype=float)
        if self.clone_mixture.residual_base == "tree_distance":
            weights = np.asarray(
                self._tree_distance_residual_weights(int(dominant)), dtype=float
            )
        else:
            weights = np.ones(self.n_clones, dtype=float)
            weights[int(dominant)] = 0.0
        weights = np.where(np.arange(self.n_clones) == int(dominant), 0.0, weights)
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("residual clone base must have positive non-dominant mass.")
        return weights / weights.sum()

    def _sync_disabled_clone_mixture_state(self) -> None:
        if self.assignments_ is None or self.hypercluster_to_clone_ is None:
            return
        dominant = np.asarray(self.hypercluster_to_clone_, dtype=np.int64)
        self.cell_clone_assignments_ = dominant[self.assignments_].copy()
        self.hypercluster_clone_proportions_ = np.eye(
            self.n_clones, dtype=float
        )[dominant]
        self.admixture_mass_ = np.zeros(dominant.size, dtype=float)
        self.residual_clone_proportions_ = np.zeros(
            (dominant.size, self.n_clones), dtype=float
        )
        self.mixture_active_ = np.zeros(dominant.size, dtype=bool)
        self.mixture_presence_rate_ = 0.0

    def _initialize_clone_mixture_state(self) -> None:
        self._validate_clone_mixture_support()
        cell_clone_ll = self._cell_clone_mut_loglikelihood()
        aggregate = np.zeros((self.n_hyperclusters, self.n_clones), dtype=float)
        np.add.at(aggregate, self.assignments_, cell_clone_ll)
        dominant = self._sample_rows(aggregate + self.log_clone_prior[None, :])
        self.hypercluster_to_clone_ = dominant.astype(np.int64)
        self.cell_clone_assignments_ = dominant[self.assignments_].astype(np.int64)
        self.hypercluster_clone_proportions_ = np.eye(
            self.n_clones, dtype=float
        )[dominant]
        self.admixture_mass_ = np.zeros(self.n_hyperclusters, dtype=float)
        self.residual_clone_proportions_ = np.zeros(
            (self.n_hyperclusters, self.n_clones), dtype=float
        )
        if self.clone_mixture.allow_pure_hyperclusters or self.n_clones == 1:
            self.mixture_active_ = np.zeros(self.n_hyperclusters, dtype=bool)
            a, b = self.clone_mixture.mixture_presence_prior
            self.mixture_presence_rate_ = float(
                np.clip(self.rng.beta(a, b), self.eps, 1.0 - self.eps)
            )
        else:
            self.mixture_active_ = np.ones(self.n_hyperclusters, dtype=bool)
            self.mixture_presence_rate_ = 1.0
            self._sample_active_mixture_parameters(
                self._clone_counts_by_hypercluster(), dominant
            )

    def _enforce_pure_clone_assignments(self) -> None:
        """Keep exact-zero pure states internally valid after z transitions."""

        for h in np.flatnonzero(~self.mixture_active_):
            self.cell_clone_assignments_[self.assignments_ == h] = int(
                self.dominant_clones_[h]
            )

    def _clone_counts_by_hypercluster(self) -> np.ndarray:
        counts = np.zeros((self.n_hyperclusters, self.n_clones), dtype=np.int64)
        np.add.at(counts, (self.assignments_, self.cell_clone_assignments_), 1)
        return counts

    def _collapsed_active_mixture_log_mass(
        self, counts: np.ndarray, dominant: int
    ) -> float:
        counts = np.asarray(counts, dtype=np.int64)
        n_dominant = int(counts[dominant])
        n_residual = int(counts.sum() - n_dominant)
        a, b = self.clone_mixture.admixture_mass_prior
        score = self._log_beta_function(a + n_residual, b + n_dominant)
        score -= self._log_beta_function(a, b)
        if self.n_clones == 1:
            return float(score)
        weights = self._residual_base_weights(dominant)
        mask = np.arange(self.n_clones) != dominant
        alpha = self.clone_mixture.residual_concentration * weights[mask]
        residual_counts = counts[mask].astype(float)
        total_alpha = float(alpha.sum())
        score += math.lgamma(total_alpha) - math.lgamma(
            total_alpha + float(residual_counts.sum())
        )
        score += float(
            np.sum(
                [
                    math.lgamma(float(x + n)) - math.lgamma(float(x))
                    for x, n in zip(alpha, residual_counts, strict=True)
                ]
            )
        )
        return float(score)

    def _sample_cell_clone_assignments(self) -> None:
        scores = self._cell_clone_mut_loglikelihood()
        scores = scores + self._safe_log_probabilities(
            self.hypercluster_clone_proportions_[self.assignments_]
        )
        self.cell_clone_assignments_ = self._sample_rows(scores).astype(np.int64)

    def _sample_clone_mixture_state(self) -> None:
        counts = self._clone_counts_by_hypercluster()
        n_h = self.n_hyperclusters
        k = self.n_clones
        scores = np.full((n_h, 2 * k), -np.inf, dtype=float)
        rho = 1.0
        if self.clone_mixture.allow_pure_hyperclusters and k > 1:
            rho = float(np.clip(self.mixture_presence_rate_, self.eps, 1.0 - self.eps))
        for h in range(n_h):
            for dominant in range(k):
                if k > 1:
                    scores[h, dominant] = (
                        math.log(rho)
                        + self.log_clone_prior[dominant]
                        + self._collapsed_active_mixture_log_mass(
                            counts[h], dominant
                        )
                    )
            if self.clone_mixture.allow_pure_hyperclusters or k == 1:
                occupied = np.flatnonzero(counts[h] > 0)
                if occupied.size == 1:
                    dominant = int(occupied[0])
                    pure_probability = 1.0 if k == 1 else 1.0 - rho
                    scores[h, k + dominant] = (
                        math.log(max(pure_probability, self.eps))
                        + self.log_clone_prior[dominant]
                    )
        choice = self._sample_rows(scores)
        active = choice < k
        dominant = np.where(active, choice, choice - k).astype(np.int64)
        self.hypercluster_to_clone_ = dominant
        self.mixture_active_ = active.astype(bool)
        if self.clone_mixture.allow_pure_hyperclusters and k > 1:
            a, b = self.clone_mixture.mixture_presence_prior
            self.mixture_presence_rate_ = float(
                np.clip(
                    self.rng.beta(a + int(active.sum()), b + int((~active).sum())),
                    self.eps,
                    1.0 - self.eps,
                )
            )
        else:
            self.mixture_presence_rate_ = 1.0 if k > 1 else 0.0
        self._sample_active_mixture_parameters(counts, dominant)

    def _sample_active_mixture_parameters(
        self, counts: np.ndarray, dominant: np.ndarray
    ) -> None:
        proportions = np.zeros((self.n_hyperclusters, self.n_clones), dtype=float)
        residual = np.zeros_like(proportions)
        admixture = np.zeros(self.n_hyperclusters, dtype=float)
        a, b = self.clone_mixture.admixture_mass_prior
        for h in range(self.n_hyperclusters):
            d = int(dominant[h])
            if not bool(self.mixture_active_[h]) or self.n_clones == 1:
                proportions[h, d] = 1.0
                continue
            n_dominant = int(counts[h, d])
            n_residual = int(counts[h].sum() - n_dominant)
            epsilon = float(
                np.clip(
                    self.rng.beta(a + n_residual, b + n_dominant),
                    self.eps,
                    1.0 - self.eps,
                )
            )
            mask = np.arange(self.n_clones) != d
            weights = self._residual_base_weights(d)[mask]
            alpha = (
                self.clone_mixture.residual_concentration * weights
                + counts[h, mask]
            )
            draw = self.rng.gamma(shape=alpha, scale=1.0)
            draw_sum = float(draw.sum())
            if not np.isfinite(draw_sum) or draw_sum <= 0.0:
                draw = weights.copy()
            else:
                draw = draw / draw_sum
            # Active Dirichlet components have support on the open simplex.
            # Gamma draws may underflow to exact zero for sparse parameters;
            # keep those structural zeros exclusive to pure hyperclusters.
            draw = np.clip(draw, self.eps, None)
            draw = draw / draw.sum()
            residual[h, mask] = draw
            admixture[h] = epsilon
            proportions[h, d] = 1.0 - epsilon
            proportions[h, mask] = epsilon * draw
        self.admixture_mass_ = admixture
        self.residual_clone_proportions_ = residual
        self.hypercluster_clone_proportions_ = proportions

    def _new_hypercluster_clone_prior(self) -> np.ndarray:
        if self.n_clones == 1:
            return np.ones(1, dtype=float)
        a, b = self.clone_mixture.admixture_mass_prior
        mean_admixture = float(a / (a + b))
        if self.clone_mixture.allow_pure_hyperclusters:
            rho = float(np.clip(self.mixture_presence_rate_, 0.0, 1.0))
        else:
            rho = 1.0
        out = np.zeros(self.n_clones, dtype=float)
        for dominant in range(self.n_clones):
            base = float(self.clone_prior[dominant])
            out[dominant] += base * (
                (1.0 - rho) + rho * (1.0 - mean_admixture)
            )
            out += base * rho * mean_admixture * self._residual_base_weights(
                dominant
            )
        return out / out.sum()

    def _mixture_log_marginal_for_cell(self, cell_clone_scores: np.ndarray) -> np.ndarray:
        out = np.empty(self.n_hyperclusters, dtype=float)
        log_pi = self._safe_log_probabilities(
            self.hypercluster_clone_proportions_
        )
        for h in range(self.n_hyperclusters):
            out[h] = float(
                self._logsumexp(
                    (cell_clone_scores + log_pi[h])[None, :], axis=1
                )[0]
            )
        return out

    def _resize_clone_mixture_state(
        self,
        old_assignments: np.ndarray,
        new_assignments: np.ndarray,
        *,
        dominant_hint: np.ndarray,
    ) -> None:
        old_dominant = self.hypercluster_to_clone_.copy()
        old_proportions = self.hypercluster_clone_proportions_.copy()
        old_admixture = self.admixture_mass_.copy()
        old_residual = self.residual_clone_proportions_.copy()
        old_active = self.mixture_active_.copy()
        new_n_h = int(new_assignments.max()) + 1
        contingency = np.zeros(
            (new_n_h, int(old_assignments.max()) + 1), dtype=np.int64
        )
        np.add.at(contingency, (new_assignments, old_assignments), 1)
        source = contingency.argmax(axis=1)
        dominant = np.asarray(dominant_hint, dtype=np.int64).copy()
        proportions = old_proportions[source].copy()
        admixture = old_admixture[source].copy()
        residual = old_residual[source].copy()
        active = old_active[source].copy()
        changed = dominant != old_dominant[source]
        for h in np.flatnonzero(changed):
            proportions[h] = 0.0
            proportions[h, dominant[h]] = 1.0
            residual[h] = 0.0
            admixture[h] = 0.0
            active[h] = False
        self.hypercluster_to_clone_ = dominant
        self.hypercluster_clone_proportions_ = proportions
        self.admixture_mass_ = admixture
        self.residual_clone_proportions_ = residual
        self.mixture_active_ = active

    def _subset_clone_mixture_rows(self, active_rows: list[int]) -> None:
        rows = np.asarray(active_rows, dtype=np.int64)
        self.hypercluster_clone_proportions_ = self.hypercluster_clone_proportions_[rows]
        self.admixture_mass_ = self.admixture_mass_[rows]
        self.residual_clone_proportions_ = self.residual_clone_proportions_[rows]
        self.mixture_active_ = self.mixture_active_[rows]

    def _append_clone_mixture_cluster(self, dominant: int) -> None:
        row = np.zeros((1, self.n_clones), dtype=float)
        row[0, int(dominant)] = 1.0
        self.hypercluster_clone_proportions_ = np.concatenate(
            [self.hypercluster_clone_proportions_, row], axis=0
        )
        self.admixture_mass_ = np.append(self.admixture_mass_, 0.0)
        self.residual_clone_proportions_ = np.concatenate(
            [
                self.residual_clone_proportions_,
                np.zeros((1, self.n_clones), dtype=float),
            ],
            axis=0,
        )
        self.mixture_active_ = np.append(self.mixture_active_, False)

    def _clone_mixture_log_prior(self) -> float:
        counts = self._clone_counts_by_hypercluster()
        rho = float(np.clip(self.mixture_presence_rate_, self.eps, 1.0 - self.eps))
        a, b = self.clone_mixture.admixture_mass_prior
        total = 0.0
        for h in range(self.n_hyperclusters):
            d = int(self.dominant_clones_[h])
            total += float(self.log_clone_prior[d])
            if not bool(self.mixture_active_[h]):
                if np.any(self.cell_clone_assignments_[self.assignments_ == h] != d):
                    return -np.inf
                if self.clone_mixture.allow_pure_hyperclusters:
                    total += math.log1p(-rho)
                continue
            if self.clone_mixture.allow_pure_hyperclusters:
                total += math.log(rho)
            epsilon = float(self.admixture_mass_[h])
            total += (a - 1.0) * math.log(epsilon)
            total += (b - 1.0) * math.log1p(-epsilon)
            total -= self._log_beta_function(a, b)
            mask = np.arange(self.n_clones) != d
            weights = self._residual_base_weights(d)[mask]
            alpha = self.clone_mixture.residual_concentration * weights
            q = np.clip(
                self.residual_clone_proportions_[h, mask], self.eps, 1.0
            )
            q = q / q.sum()
            total += math.lgamma(float(alpha.sum()))
            total -= sum(math.lgamma(float(x)) for x in alpha)
            total += float(np.sum((alpha - 1.0) * np.log(q)))
            log_pi = self._safe_log_probabilities(
                self.hypercluster_clone_proportions_[h]
            )
            occupied = counts[h] > 0
            total += float(np.sum(counts[h, occupied] * log_pi[occupied]))
        return float(total)

    def _clone_mixture_hyperprior_log_prior(self) -> float:
        if not (
            self.clone_mixture.enabled
            and self.clone_mixture.allow_pure_hyperclusters
            and self.n_clones > 1
        ):
            return 0.0
        a, b = self.clone_mixture.mixture_presence_prior
        rho = float(np.clip(self.mixture_presence_rate_, self.eps, 1.0 - self.eps))
        return float(
            (a - 1.0) * math.log(rho)
            + (b - 1.0) * math.log1p(-rho)
            - self._log_beta_function(a, b)
        )

    def current_effective_clone_counts(self) -> np.ndarray:
        return 1.0 / np.sum(self.hypercluster_clone_proportions_ ** 2, axis=1)

    def current_admixture_entropy(self) -> np.ndarray:
        p = self.hypercluster_clone_proportions_
        out = np.zeros(p.shape[0], dtype=float)
        for h in range(p.shape[0]):
            positive = p[h] > 0.0
            out[h] = -float(np.sum(p[h, positive] * np.log(p[h, positive])))
        return out

    def current_dominant_fractions(self) -> np.ndarray:
        return np.max(self.hypercluster_clone_proportions_, axis=1)

    def _migrate_clone_mixture_state(self) -> None:
        if not hasattr(self, "clone_mixture"):
            self.clone_mixture = CloneMixtureConfig(enabled=False)
        for name in (
            "dominant_clone_trace_",
            "hypercluster_clone_proportions_trace_",
            "admixture_mass_trace_",
            "mixture_active_trace_",
            "effective_clone_count_trace_",
            "admixture_entropy_trace_",
            "dominant_fraction_trace_",
        ):
            if not hasattr(self, name):
                setattr(self, name, [])
        if not hasattr(self, "_iterations_completed_"):
            self._iterations_completed_ = 0
        if not hasattr(self, "alpha_prior"):
            self.alpha_prior = (1.0, 1.0)
        if not hasattr(self, "sample_alpha"):
            self.sample_alpha = False
        if getattr(self, "assignments_", None) is None:
            return
        if getattr(self, "hypercluster_to_clone_", None) is None:
            return
        required = (
            "cell_clone_assignments_",
            "hypercluster_clone_proportions_",
            "admixture_mass_",
            "residual_clone_proportions_",
            "mixture_active_",
            "mixture_presence_rate_",
        )
        if any(not hasattr(self, name) or getattr(self, name) is None for name in required):
            # v0.1/v0.2 checkpoints become an exact pure-mixture state.
            self.cell_clone_assignments_ = self.hypercluster_to_clone_[
                self.assignments_
            ].copy()
            self.hypercluster_clone_proportions_ = np.eye(
                self.n_clones, dtype=float
            )[self.hypercluster_to_clone_]
            self.admixture_mass_ = np.zeros(self.n_hyperclusters, dtype=float)
            self.residual_clone_proportions_ = np.zeros(
                (self.n_hyperclusters, self.n_clones), dtype=float
            )
            self.mixture_active_ = np.zeros(self.n_hyperclusters, dtype=bool)
            self.mixture_presence_rate_ = 0.0

    @property
    def dominant_clones_(self) -> np.ndarray | None:
        """Canonical dominant clone per hypercluster.

        ``hypercluster_to_clone_`` remains the deprecated v0.2 compatibility
        storage and always aliases this property.
        """

        return self.hypercluster_to_clone_

    @dominant_clones_.setter
    def dominant_clones_(self, value: np.ndarray | None) -> None:
        self.hypercluster_to_clone_ = value

    @property
    def hypercluster_to_clone(self) -> np.ndarray | None:
        """Deprecated alias for :attr:`dominant_clones_`."""

        return self.dominant_clones_

    @hypercluster_to_clone.setter
    def hypercluster_to_clone(self, value: np.ndarray | None) -> None:
        self.dominant_clones_ = value

    def current_cell_clone_assignment(self) -> np.ndarray:
        self._require_prefit()
        if self.clone_mixture.enabled:
            return self.cell_clone_assignments_
        return self.hypercluster_to_clone_[self.assignments_]

    def current_state_cell_clone_probabilities(self) -> np.ndarray:
        """Return clone probabilities conditional on the current sampler state."""

        self._require_prefit()
        if self.clone_mixture.enabled:
            scores = self._safe_log_probabilities(
                self.hypercluster_clone_proportions_[self.assignments_]
            ) + self._cell_clone_mut_loglikelihood()
            normalizer = self._logsumexp(scores, axis=1)
            return np.exp(scores - normalizer[:, None])
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

    def posterior_cell_clone_coassignment(
        self,
        *,
        burn_in: int | float = 0,
        thin: int = 1,
    ) -> np.ndarray:
        """Return posterior probabilities that pairs of cells share a clone."""

        if not self.cell_clone_assignment_trace_:
            raise RuntimeError(
                "cell-clone tracking is required for clone co-clustering summaries."
            )
        return coassignment_probabilities_from_trace(
            self.cell_clone_assignment_trace_, burn_in=burn_in, thin=thin
        )

    def posterior_cell_coclustering_probabilities(
        self,
        *,
        burn_in: int | float = 0,
        thin: int = 1,
    ) -> np.ndarray:
        """Alias for :meth:`posterior_cell_clone_coassignment`."""

        return self.posterior_cell_clone_coassignment(
            burn_in=burn_in, thin=thin
        )

    @staticmethod
    def _selected_trace_indices(
        n_draws: int, burn_in: int | float, thin: int
    ) -> np.ndarray:
        if thin < 1:
            raise ValueError("thin must be at least 1.")
        if isinstance(burn_in, float):
            if not 0.0 <= burn_in < 1.0:
                raise ValueError("fractional burn_in must lie in [0,1).")
            start = int(math.floor(n_draws * burn_in))
        else:
            start = int(burn_in)
            if start < 0:
                raise ValueError("burn_in must be nonnegative.")
        indices = np.arange(start, n_draws, thin, dtype=np.int64)
        if indices.size == 0:
            raise ValueError("no posterior draws remain after burn-in and thinning.")
        return indices

    def _map_hypercluster_trace_to_current_partition(
        self,
        values_trace: list[np.ndarray],
        *,
        burn_in: int | float,
        thin: int,
    ) -> np.ndarray:
        if not values_trace or len(values_trace) != len(self.assignment_trace_):
            raise RuntimeError(
                "matching assignment and hypercluster-mixture tracking are required."
            )
        indices = self._selected_trace_indices(
            len(values_trace), burn_in, thin
        )
        current = np.asarray(self.assignments_, dtype=np.int64)
        current_h = self.n_hyperclusters
        mapped: list[np.ndarray] = []
        for index in indices:
            draw_z = np.asarray(self.assignment_trace_[int(index)], dtype=np.int64)
            draw_values = np.asarray(values_trace[int(index)])
            draw_h = int(draw_z.max()) + 1
            contingency = np.zeros((current_h, draw_h), dtype=np.int64)
            np.add.at(contingency, (current, draw_z), 1)
            match = contingency.argmax(axis=1)
            mapped.append(draw_values[match])
        return np.asarray(mapped)

    def posterior_hypercluster_clone_proportions(
        self,
        *,
        burn_in: int | float = 0,
        thin: int = 1,
    ) -> np.ndarray:
        """Return clone proportions for the current hyperclusters.

        Tracked hyperclusters are matched to the current partition by maximum
        cell overlap before averaging, making the result invariant to numeric
        hypercluster-label permutations.
        """

        if self.hypercluster_clone_proportions_trace_:
            mapped = self._map_hypercluster_trace_to_current_partition(
                self.hypercluster_clone_proportions_trace_,
                burn_in=burn_in,
                thin=thin,
            )
            return np.mean(mapped, axis=0)
        clone_probability = self.posterior_cell_clone_probabilities(
            burn_in=burn_in, thin=thin
        )
        out = np.zeros((self.n_hyperclusters, self.n_clones), dtype=float)
        for h in range(self.n_hyperclusters):
            out[h] = clone_probability[self.assignments_ == h].mean(axis=0)
        return out

    def posterior_dominant_clone_probabilities(
        self,
        *,
        burn_in: int | float = 0,
        thin: int = 1,
    ) -> np.ndarray:
        """Return posterior dominant-clone probabilities for current hyperclusters."""

        if self.dominant_clone_trace_:
            mapped = self._map_hypercluster_trace_to_current_partition(
                self.dominant_clone_trace_, burn_in=burn_in, thin=thin
            ).astype(np.int64)
            out = np.zeros((self.n_hyperclusters, self.n_clones), dtype=float)
            for draw in mapped:
                out[np.arange(self.n_hyperclusters), draw] += 1.0
            return out / mapped.shape[0]
        proportions = self.posterior_hypercluster_clone_proportions(
            burn_in=burn_in, thin=thin
        )
        out = np.zeros_like(proportions)
        out[np.arange(self.n_hyperclusters), proportions.argmax(axis=1)] = 1.0
        return out

    def posterior_admixture_probabilities(
        self,
        *,
        burn_in: int | float = 0,
        thin: int = 1,
    ) -> np.ndarray:
        """Return posterior probabilities that current hyperclusters are admixed."""

        if self.mixture_active_trace_:
            mapped = self._map_hypercluster_trace_to_current_partition(
                self.mixture_active_trace_, burn_in=burn_in, thin=thin
            )
            return np.mean(mapped.astype(float), axis=0)
        return self.mixture_active_.astype(float)

    def posterior_effective_clone_counts(
        self,
        *,
        burn_in: int | float = 0,
        thin: int = 1,
    ) -> np.ndarray:
        """Return posterior effective clone count for each current hypercluster."""

        if self.effective_clone_count_trace_:
            mapped = self._map_hypercluster_trace_to_current_partition(
                self.effective_clone_count_trace_, burn_in=burn_in, thin=thin
            )
            return np.mean(mapped, axis=0)
        proportions = self.posterior_hypercluster_clone_proportions(
            burn_in=burn_in, thin=thin
        )
        return 1.0 / np.sum(proportions**2, axis=1)

    def posterior_hypercluster_partition_medoid(
        self,
        *,
        burn_in: int | float = 0,
        thin: int = 1,
    ) -> dict[str, Any]:
        """Return the sampled hypercluster partition nearest the posterior mean.

        The partition is label invariant because the loss is computed from
        pairwise co-assignment matrices. ``trace_index`` refers to the original
        retained assignment trace before burn-in and thinning.
        """

        if not self.assignment_trace_:
            raise RuntimeError("assignment tracking is required for a partition medoid.")
        assignment, trace_index, losses = partition_medoid_from_trace(
            self.assignment_trace_, burn_in=burn_in, thin=thin
        )
        return {
            "assignment": assignment,
            "trace_index": trace_index,
            "losses": losses,
        }

    def posterior_clone_partition_medoid(
        self,
        *,
        burn_in: int | float = 0,
        thin: int = 1,
    ) -> dict[str, Any]:
        """Return the sampled cell-clone partition nearest its posterior mean."""

        if not self.cell_clone_assignment_trace_:
            raise RuntimeError("cell-clone tracking is required for a partition medoid.")
        assignment, trace_index, losses = partition_medoid_from_trace(
            self.cell_clone_assignment_trace_, burn_in=burn_in, thin=thin
        )
        return {
            "assignment": assignment,
            "trace_index": trace_index,
            "losses": losses,
        }

    def posterior_partition_medoid(
        self,
        kind: Literal["hypercluster", "clone"] = "hypercluster",
        *,
        burn_in: int | float = 0,
        thin: int = 1,
    ) -> dict[str, Any]:
        """Return a label-invariant posterior medoid for either partition trace."""

        if kind == "hypercluster":
            return self.posterior_hypercluster_partition_medoid(
                burn_in=burn_in, thin=thin
            )
        if kind == "clone":
            return self.posterior_clone_partition_medoid(
                burn_in=burn_in, thin=thin
            )
        raise ValueError("kind must be 'hypercluster' or 'clone'.")

    def posterior_summary(
        self,
        *,
        burn_in: int | float = 0,
        thin: int = 1,
        include_partition_medoids: bool = False,
    ) -> dict[str, Any]:
        """Return canonical coherent posterior summaries from retained draws."""

        summary: dict[str, Any] = {
            "cell_clone_probabilities": self.posterior_cell_clone_probabilities(
                burn_in=burn_in, thin=thin, use_trace=True
            ),
            "cell_genotype_probabilities": self.posterior_cell_genotype_probabilities(
                burn_in=burn_in, thin=thin, use_trace=True
            ),
            "hypercluster_coassignment": self.posterior_hypercluster_coassignment(
                burn_in=burn_in, thin=thin
            ),
            "cell_clone_coassignment": self.posterior_cell_clone_coassignment(
                burn_in=burn_in, thin=thin
            ),
        }
        if self.clone_mixture.enabled:
            summary.update(
                {
                    "hypercluster_clone_proportions": self.posterior_hypercluster_clone_proportions(
                        burn_in=burn_in, thin=thin
                    ),
                    "dominant_clone_probabilities": self.posterior_dominant_clone_probabilities(
                        burn_in=burn_in, thin=thin
                    ),
                    "admixture_probabilities": self.posterior_admixture_probabilities(
                        burn_in=burn_in, thin=thin
                    ),
                    "effective_clone_counts": self.posterior_effective_clone_counts(
                        burn_in=burn_in, thin=thin
                    ),
                }
            )
        if include_partition_medoids:
            summary["hypercluster_partition_medoid"] = (
                self.posterior_hypercluster_partition_medoid(
                    burn_in=burn_in, thin=thin
                )
            )
            summary["clone_partition_medoid"] = self.posterior_clone_partition_medoid(
                burn_in=burn_in, thin=thin
            )
        return summary

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state without ephemeral views or proposal caches."""

        state = self.__dict__.copy()
        state.pop("state_", None)
        state["_split_merge_cache_"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._migrate_clone_mixture_state()
        self.state_ = StateView(self)
        self._split_merge_cache_ = None

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        include_traces: bool = True,
    ) -> Path:
        """Serialize the exact sampler state for deterministic continuation.

        Checkpoints use Python pickle inside gzip and must only be loaded from a
        trusted source. The model RNG state, adaptive scheduler, diagnostics,
        and subclass-specific state are preserved. Temporary sufficient-statistic
        caches are intentionally discarded and rebuilt after loading.
        """

        self._require_prefit()
        destination = Path(path)
        if destination.parent != Path(""):
            destination.parent.mkdir(parents=True, exist_ok=True)
        model = copy.deepcopy(self)
        model._split_merge_cache_ = None
        if not include_traces:
            seen: set[int] = set()
            for name, value in vars(model).items():
                if name.endswith("_trace_") and isinstance(value, list):
                    identity = id(value)
                    if identity not in seen:
                        value.clear()
                        seen.add(identity)
        payload = {
            "format": "cactri-checkpoint",
            "format_version": 1,
            "package_version": _CHECKPOINT_PACKAGE_VERSION,
            "model_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "model": model,
        }
        with gzip.open(destination, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return destination

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        accelerator: Backend | None = None,
        deterministic: bool | None = None,
    ) -> "Cactri":
        """Load a trusted checkpoint and optionally change its compute backend."""

        source = Path(path)
        with gzip.open(source, "rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict) or payload.get("format") != "cactri-checkpoint":
            raise ValueError("not a Cactri checkpoint.")
        if payload.get("format_version") != 1:
            raise ValueError("unsupported Cactri checkpoint format version.")
        package_version = str(payload.get("package_version", "0.1"))
        if package_version.startswith("0.3"):
            raise ValueError(
                "Cactri 0.3.x checkpoints belong to the discontinued experimental "
                "sampler branch and cannot be loaded by the 0.4 lineage. Resume or "
                "export the state with Cactri 0.3.x instead."
            )
        model = payload.get("model")
        if not isinstance(model, Cactri):
            raise TypeError("checkpoint does not contain a Cactri model.")
        if cls is not Cactri and not isinstance(model, cls):
            raise TypeError(
                f"checkpoint contains {type(model).__name__}, not {cls.__name__}."
            )
        backend = model.accelerator_name if accelerator is None else accelerator
        deterministic_value = (
            model.deterministic if deterministic is None else bool(deterministic)
        )
        model.accelerator = Accelerator(
            backend, deterministic=deterministic_value
        )
        model.accelerator_name = model.accelerator.backend
        model.deterministic = deterministic_value
        model._migrate_clone_mixture_state()
        model.state_ = StateView(model)
        model._split_merge_cache_ = None
        return model

    def log_likelihood_components(self) -> dict[str, float]:
        rows = np.arange(self.n_cells)
        bcr = float(self._cell_hypercluster_bcr_loglikelihood()[rows, self.assignments_].sum())
        if self.clone_mixture.enabled:
            mutation = float(
                self._cell_clone_mut_loglikelihood()[rows, self.cell_clone_assignments_].sum()
            )
        else:
            mutation = float(
                self._cell_hypercluster_mut_loglikelihood()[rows, self.assignments_].sum()
            )
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
        if not self.clone_mixture.enabled:
            return float(np.sum(self.log_clone_prior[self.hypercluster_to_clone_]))
        return self._clone_mixture_log_prior()

    def log_posterior(self) -> float:
        return float(
            self.log_likelihood()
            + self.crp_assignment_log_prior()
            + self.clone_assignment_log_prior()
            + self._genotype_log_prior()
            + self._clone_mixture_hyperprior_log_prior()
            + self._alpha_log_prior()
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
        cell_clone = self._cell_clone_mut_loglikelihood()
        if not self.clone_mixture.enabled:
            return cell_clone[:, self.hypercluster_to_clone_]
        out = np.empty((self.n_cells, self.n_hyperclusters), dtype=float)
        log_pi = self._safe_log_probabilities(self.hypercluster_clone_proportions_)
        for h in range(self.n_hyperclusters):
            out[:, h] = self._logsumexp(cell_clone + log_pi[h][None, :], axis=1)
        return out

    def _new_hypercluster_mutation_loglikelihood(self) -> np.ndarray:
        prior = (
            self._new_hypercluster_clone_prior()
            if self.clone_mixture.enabled
            else self.clone_prior
        )
        return self._logsumexp(
            self._cell_clone_mut_loglikelihood()
            + self._safe_log_probabilities(prior)[None, :],
            axis=1,
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

    def _resample_alpha(self) -> None:
        """Escobar-West auxiliary update for the CRP concentration."""

        shape_prior, rate_prior = self.alpha_prior
        eta = float(self.rng.beta(self.alpha_ + 1.0, self.n_cells))
        rate = float(rate_prior - math.log(max(eta, self.eps)))
        numerator = shape_prior + self.n_hyperclusters - 1.0
        mixture_probability = numerator / (
            self.n_cells * rate + numerator
        )
        shape = (
            shape_prior + self.n_hyperclusters
            if self.rng.random() < mixture_probability
            else shape_prior + self.n_hyperclusters - 1.0
        )
        self.alpha_ = float(self.rng.gamma(shape=shape, scale=1.0 / rate))

    def _alpha_log_prior(self) -> float:
        if not self.sample_alpha:
            return 0.0
        shape, rate = self.alpha_prior
        return float((shape - 1.0) * math.log(self.alpha_) - rate * self.alpha_)

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
            "dominant_clone_trace_",
            "hypercluster_clone_proportions_trace_",
            "admixture_mass_trace_",
            "mixture_active_trace_",
            "effective_clone_count_trace_",
            "admixture_entropy_trace_",
            "dominant_fraction_trace_",
        ):
            getattr(self, name).clear()
        self.split_merge_attempts_ = 0
        self.split_merge_accepts_ = 0
        self.split_attempts_ = 0
        self.split_accepts_ = 0
        self.merge_attempts_ = 0
        self.merge_accepts_ = 0
        self.split_merge_cache_builds_ = 0
        self.split_merge_cache_reuses_ = 0
        self.adaptive_split_probability_ = 0.5
        self._adaptive_initialized_ = False
        self._adaptive_window_split_attempts = 0
        self._adaptive_window_split_accepts = 0
        self._adaptive_window_merge_attempts = 0
        self._adaptive_window_merge_accepts = 0
        self._split_merge_cache_ = None
        self._iterations_completed_ = 0
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
                components["total"]
                + crp
                + clone
                + genotype
                + self._clone_mixture_hyperprior_log_prior()
                + self._alpha_log_prior()
                + self._extra_log_prior()
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
        if config.dominant_clones:
            self.dominant_clone_trace_.append(self.dominant_clones_.copy())
        if config.clone_proportions:
            self.hypercluster_clone_proportions_trace_.append(
                self.hypercluster_clone_proportions_.copy()
            )
        if config.admixture_mass:
            self.admixture_mass_trace_.append(self.admixture_mass_.copy())
        if config.mixture_active:
            self.mixture_active_trace_.append(self.mixture_active_.copy())
        if config.effective_clone_counts:
            self.effective_clone_count_trace_.append(
                self.current_effective_clone_counts().copy()
            )
        if config.admixture_entropy:
            self.admixture_entropy_trace_.append(self.current_admixture_entropy().copy())
        if config.dominant_fraction:
            self.dominant_fraction_trace_.append(self.current_dominant_fractions().copy())
        self._append_subclass_tracking(config)

    def to_dict(self) -> dict[str, Any]:
        self._require_prefit()
        genotype = self._genotype_matrix().copy()
        out: dict[str, Any] = {
            "assignments": self.assignments_.copy(),
            "bcr_profiles": self.bcr_profiles_.copy(),
            "hypercluster_to_clone": self.hypercluster_to_clone_.copy(),
            "dominant_clones": self.dominant_clones_.copy(),
            "hypercluster_clone_proportions": self.hypercluster_clone_proportions_.copy(),
            "admixture_mass": self.admixture_mass_.copy(),
            "residual_clone_proportions": self.residual_clone_proportions_.copy(),
            "mixture_active": self.mixture_active_.copy(),
            "mixture_presence_rate": float(self.mixture_presence_rate_),
            "clone_mixture_config": self.clone_mixture,
            "cell_clone_assignment": self.current_cell_clone_assignment().copy(),
            "cell_clone_assignments": self.current_cell_clone_assignment().copy(),
            "genotype_matrix": genotype,
            "mutation_profile": genotype.copy(),
            "p_obs_by_mutation": self.p_obs_by_mutation_.copy(),
            "p_unobs": float(self.p_unobs_),
            "alpha": float(self.alpha_),
            "alpha_prior": self.alpha_prior,
            "sample_alpha": self.sample_alpha,
            "n_hyperclusters": self.n_hyperclusters,
            "n_clones": self.n_clones,
            "iterations_completed": self._iterations_completed_,
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
            "dominant_clone_trace": np.asarray(self.dominant_clone_trace_, dtype=object),
            "hypercluster_clone_proportions_trace": np.asarray(
                self.hypercluster_clone_proportions_trace_, dtype=object
            ),
            "admixture_mass_trace": np.asarray(self.admixture_mass_trace_, dtype=object),
            "mixture_active_trace": np.asarray(self.mixture_active_trace_, dtype=object),
            "effective_clone_count_trace": np.asarray(
                self.effective_clone_count_trace_, dtype=object
            ),
            "admixture_entropy_trace": np.asarray(
                self.admixture_entropy_trace_, dtype=object
            ),
            "dominant_fraction_trace": np.asarray(
                self.dominant_fraction_trace_, dtype=object
            ),
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
        # Dense cell-by-cell co-assignment matrices are intentionally not
        # materialized by to_dict()/fit().  For realistic datasets these are
        # O(N^2) in both time and memory.  Request them explicitly through
        # posterior_hypercluster_coassignment() or
        # posterior_cell_clone_coassignment().
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
