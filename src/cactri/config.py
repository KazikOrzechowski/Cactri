from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Control which sampler quantities are retained.

    Large cell-by-mutation or genotype traces are disabled by default. ``every``
    counts completed Gibbs iterations and must be positive.

    Use :meth:`posterior` when coherent posterior cell-genotype and partition
    summaries will be required after fitting.
    """

    log_likelihood: bool = True
    log_posterior: bool = True
    assignments: bool = False
    cell_clone_assignments: bool = True
    genotype_state: bool = False
    observation_probabilities: bool = False
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
    ) -> "TrackingConfig":
        """Return a configuration sufficient for coherent posterior summaries."""

        return cls(
            log_likelihood=True,
            log_posterior=True,
            assignments=True,
            cell_clone_assignments=True,
            genotype_state=True,
            observation_probabilities=observation_probabilities,
            every=every,
        )


LocalAssignmentSampler = Literal["none", "approximate", "sequential"]


@dataclass(frozen=True, slots=True)
class SplitMergeConfig:
    """Configuration for the collapsed restricted-Gibbs split/merge kernel.

    ``local_sampler`` specifies the local assignment transition run before the
    global proposals. The vectorized approximate sweep is the default because a
    complete sequential scan is expensive on large single-cell datasets.
    """

    proposals_per_sweep: int = 1
    local_sampler: LocalAssignmentSampler = "approximate"

    def __post_init__(self) -> None:
        if self.proposals_per_sweep < 1:
            raise ValueError("SplitMergeConfig.proposals_per_sweep must be at least 1.")
        if self.local_sampler not in {"none", "approximate", "sequential"}:
            raise ValueError(
                "SplitMergeConfig.local_sampler must be 'none', 'approximate', or 'sequential'."
            )
