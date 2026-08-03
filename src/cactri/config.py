from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ResidualCloneBase = Literal["uniform", "tree_distance"]


@dataclass(frozen=True, slots=True)
class CloneMixtureConfig:
    """Dominant-clone sparse-admixture model within each BCR hypercluster.

    Mixtures are opt-in. When disabled, Cactri follows the exact v0.2.2
    one-clone-per-hypercluster model. When enabled, each cell receives its own
    latent mutation-clone label, shrunk toward one dominant clone per BCR
    hypercluster.
    """

    enabled: bool = False
    admixture_mass_prior: tuple[float, float] = (1.0, 19.0)
    residual_concentration: float = 0.2
    residual_base: ResidualCloneBase = "uniform"
    tree_distance_decay: float = 1.0
    allow_pure_hyperclusters: bool = True
    mixture_presence_prior: tuple[float, float] = (1.0, 9.0)

    def __post_init__(self) -> None:
        if len(self.admixture_mass_prior) != 2 or min(self.admixture_mass_prior) <= 0:
            raise ValueError("admixture_mass_prior must contain two positive values.")
        if self.residual_concentration <= 0:
            raise ValueError("residual_concentration must be positive.")
        if self.residual_base not in {"uniform", "tree_distance"}:
            raise ValueError("residual_base must be 'uniform' or 'tree_distance'.")
        if self.tree_distance_decay <= 0:
            raise ValueError("tree_distance_decay must be positive.")
        if len(self.mixture_presence_prior) != 2 or min(self.mixture_presence_prior) <= 0:
            raise ValueError("mixture_presence_prior must contain two positive values.")


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Control which sampler quantities are retained.

    Large traces are disabled by default. Mixture diagnostics are independently
    selectable. :meth:`posterior` tracks the coherent cell-clone, genotype, and
    BCR-partition states needed for posterior cell summaries.
    """

    log_likelihood: bool = True
    log_posterior: bool = True
    assignments: bool = False
    cell_clone_assignments: bool = True
    genotype_state: bool = False
    observation_probabilities: bool = False
    dominant_clones: bool = False
    clone_proportions: bool = False
    admixture_mass: bool = False
    mixture_active: bool = False
    effective_clone_counts: bool = False
    admixture_entropy: bool = False
    dominant_fraction: bool = False
    every: int = 1

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ValueError("TrackingConfig.every must be at least 1.")

    @classmethod
    def posterior(
        cls,
        *,
        every: int = 1,
        observation_probabilities: bool = False,
        mixture_diagnostics: bool = False,
    ) -> "TrackingConfig":
        """Return a configuration sufficient for coherent posterior summaries."""

        return cls(
            log_likelihood=True,
            log_posterior=True,
            assignments=True,
            cell_clone_assignments=True,
            genotype_state=True,
            observation_probabilities=observation_probabilities,
            dominant_clones=mixture_diagnostics,
            clone_proportions=mixture_diagnostics,
            admixture_mass=mixture_diagnostics,
            mixture_active=mixture_diagnostics,
            effective_clone_counts=mixture_diagnostics,
            admixture_entropy=mixture_diagnostics,
            dominant_fraction=mixture_diagnostics,
            every=every,
        )


LocalAssignmentSampler = Literal["none", "approximate", "sequential"]
AnchorStrategy = Literal["uniform_pair", "adaptive", "mutation_discordance"]


@dataclass(frozen=True, slots=True)
class SplitMergeConfig:
    """Configuration for the retained v0.2 split/merge transition.

    The extra v0.3-era fields are retained only so trusted v0.3 checkpoints can
    be decoded and rejected with a clear lineage error. The v0.4 runtime does
    not implement mutation-informed or hybrid partition kernels.
    """

    proposals_per_sweep: int = 1
    local_sampler: LocalAssignmentSampler = "approximate"
    cache_sufficient_statistics: bool = True
    anchor_strategy: AnchorStrategy = "uniform_pair"
    initial_split_probability: float = 0.5
    adaptation_interval: int = 25
    adaptation_step: float = 0.25
    min_split_probability: float = 0.1
    max_split_probability: float = 0.9
    adapt_until: int | None = None
    max_restricted_cells: int | None = None
    discordance_size_power: float = 0.5
    discordance_epsilon: float = 1e-8
    merge_similarity_strength: float = 2.0

    def __post_init__(self) -> None:
        if self.proposals_per_sweep < 1:
            raise ValueError("SplitMergeConfig.proposals_per_sweep must be at least 1.")
        if self.local_sampler not in {"none", "approximate", "sequential"}:
            raise ValueError(
                "SplitMergeConfig.local_sampler must be 'none', 'approximate', or 'sequential'."
            )
        if self.anchor_strategy not in {"uniform_pair", "adaptive"}:
            raise ValueError(
                "SplitMergeConfig.anchor_strategy must be 'uniform_pair' or 'adaptive'."
            )
        for name, value in (
            ("initial_split_probability", self.initial_split_probability),
            ("min_split_probability", self.min_split_probability),
            ("max_split_probability", self.max_split_probability),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"SplitMergeConfig.{name} must lie in (0, 1).")
        if self.min_split_probability >= self.max_split_probability:
            raise ValueError("min_split_probability must be below max_split_probability.")
        if not self.min_split_probability <= self.initial_split_probability <= self.max_split_probability:
            raise ValueError("initial_split_probability must lie between the configured bounds.")
        if self.adaptation_interval < 1:
            raise ValueError("adaptation_interval must be at least 1.")
        if self.adaptation_step <= 0:
            raise ValueError("adaptation_step must be positive.")
        if self.adapt_until is not None and self.adapt_until < 1:
            raise ValueError("adapt_until must be positive or None.")
        if self.max_restricted_cells is not None:
            if self.max_restricted_cells < 2:
                raise ValueError("max_restricted_cells must be at least 2 or None.")
            if self.anchor_strategy != "uniform_pair":
                raise ValueError(
                    "max_restricted_cells currently requires anchor_strategy='uniform_pair'."
                )
        if self.discordance_size_power < 0:
            raise ValueError("discordance_size_power must be nonnegative.")
        if self.discordance_epsilon <= 0:
            raise ValueError("discordance_epsilon must be positive.")
        if self.merge_similarity_strength < 0:
            raise ValueError("merge_similarity_strength must be nonnegative.")


# Decode-only v0.3 checkpoint classes. They are intentionally not exported.
@dataclass(frozen=True, slots=True)
class BlockMoveConfig:
    proposals_per_sweep: int = 2
    subset_strategy: str = "restricted_gibbs"
    max_block_cells: int = 200
    min_block_cells: int = 2
    initial_split_off_probability: float = 0.5
    adaptation_interval: int = 25
    adaptation_step: float = 0.25
    min_split_off_probability: float = 0.1
    max_split_off_probability: float = 0.9
    adapt_until: int | None = None
    source_size_power: float = 0.5
    discordance_epsilon: float = 1e-8
    destination_similarity_strength: float = 2.0
    bernoulli_intercept: float = -1.5
    bernoulli_temperature: float = 1.0
    block_size_geometric_p: float = 0.35
    restricted_gibbs_temperature: float = 1.0


@dataclass(frozen=True, slots=True)
class HybridSamplerConfig:
    initial_approximate_fraction: float = 0.30
    adaptation_fraction: float = 0.75
    warmup_approximate_every_initial: int = 1
    warmup_approximate_every_late: int = 5
    retention_approximate_every: int = 5
    warmup_split_merge_every: int = 5
    retention_split_merge_every: int = 10
