from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Control which sampler quantities are retained.

    Large cell-by-mutation or genotype traces are disabled by default. ``every``
    counts completed Gibbs iterations and must be positive.
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
