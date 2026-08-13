from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import Cactri


class StateView:
    """Small compatibility view over a model's canonical ordinary attributes.

    New code should prefer attributes such as ``assignments_`` and
    ``p_obs_by_mutation_``. The view preserves the legacy ``state_`` access
    pattern without duplicating mutable state.
    """

    __slots__ = ("_model",)

    _ALIASES = {
        "assignments": "assignments_",
        "bcr_profiles": "bcr_profiles_",
        "hypercluster_to_clone": "hypercluster_to_clone_",
        "dominant_clones": "dominant_clones_",
        "hypercluster_clone_proportions": "hypercluster_clone_proportions_",
        "cell_clone_assignments": "cell_clone_assignments_",
        "admixture_mass": "admixture_mass_",
        "residual_clone_proportions": "residual_clone_proportions_",
        "mixture_active": "mixture_active_",
        "mixture_presence_rate": "mixture_presence_rate_",
        "p_obs_by_mutation": "p_obs_by_mutation_",
        "p_unobs": "p_unobs_",
        "alpha": "alpha_",
        "genotype_matrix": "genotype_matrix_",
        "mutation_profile": "mutation_profile_",
        "mutation_tree_assignment": "mutation_tree_assignment_",
        "relax_rate": "relax_rate_",
        "edge_error_rate": "edge_error_rate_",
    }

    def __init__(self, model: Cactri) -> None:
        object.__setattr__(self, "_model", model)

    def __getattr__(self, name: str) -> Any:
        target = self._ALIASES.get(name, name)
        try:
            return getattr(self._model, target)
        except AttributeError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        target = self._ALIASES.get(name, name)
        setattr(self._model, target, value)
