import numpy as np

from cactri import BCRInitializer


def test_single_and_consensus_fit():
    rng = np.random.default_rng(20)
    seq = np.vstack(
        [
            np.tile([0, 1, 2, 3], (12, 1)),
            np.tile([3, 2, 1, 0], (12, 1)),
        ]
    )
    # Add a small amount of sequence noise.
    mask = rng.random(seq.shape) < 0.05
    seq[mask] = rng.integers(0, 4, size=int(mask.sum()))

    model = BCRInitializer(accelerator="numpy", random_state=21)
    result = model.fit(seq, n_iter=5, init="random", random_init_clusters=4)
    assert result["assignments"].shape == (24,)

    consensus = model.consensus_fit(
        seq,
        n_chains=2,
        n_iter=8,
        burn_in=4,
        thin=2,
        random_init_clusters=4,
    )
    assert consensus["consensus_assignments"].shape == (24,)
    assert consensus["coassignment_matrix"].shape == (24, 24)
    assert model.diagnostics_["n_retained_partitions"] == 4
