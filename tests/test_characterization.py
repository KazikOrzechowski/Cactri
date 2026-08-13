"""Stage-1 characterization tests for stable public behavior."""

import numpy as np

from cactri import CactriOmega, CactriTree


def test_result_keys_and_shapes_are_stable():
    rng = np.random.default_rng(30)
    seq = rng.integers(0, 4, size=(8, 3))
    total = rng.integers(1, 5, size=(8, 4))
    alt = rng.binomial(total, 0.25)

    for model in (
        CactriOmega(n_clones=3, accelerator="numpy", random_state=31),
        CactriTree(n_levels=1, accelerator="numpy", random_state=31),
    ):
        model.prefit(seq, alt, total, init=[5, 5, 5, 5, 8, 8, 8, 8])
        result = model.fit(0)
        required = {
            "assignments",
            "bcr_profiles",
            "hypercluster_to_clone",
            "cell_clone_assignment",
            "cell_clone_assignments",
            "genotype_matrix",
            "mutation_profile",
            "p_obs_by_mutation",
            "p_unobs",
            "alpha",
            "n_hyperclusters",
            "n_clones",
            "log_likelihood",
            "log_posterior",
        }
        assert required <= set(result)
        assert result["assignments"].shape == (8,)
        assert result["p_obs_by_mutation"].shape == (4,)
