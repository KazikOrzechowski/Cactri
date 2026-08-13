import numpy as np
import pytest

from cactri import CactriOmega, CactriTree


@pytest.fixture
def data():
    rng = np.random.default_rng(3)
    seq = rng.integers(0, 4, size=(12, 5))
    total = rng.integers(1, 7, size=(12, 4))
    alt = rng.binomial(total, 0.15)
    return seq, alt, total


@pytest.mark.parametrize("model", ["omega", "tree"])
def test_vector_init_reindexed(model, data):
    seq, alt, total = data
    obj = (
        CactriOmega(n_clones=3, accelerator="numpy", random_state=1)
        if model == "omega"
        else CactriTree(n_levels=1, accelerator="numpy", random_state=1)
    )
    obj.prefit(seq, alt, total, init=[9] * 4 + [2] * 4 + [7] * 4)
    assert set(obj.assignments_.tolist()) == {0, 1, 2}
    assert obj.assignments_.shape == (12,)


@pytest.mark.parametrize(
    "bad",
    [
        [0, 1],
        [[0] * 12],
        [-1] + [0] * 11,
        [0.5] + [0] * 11,
        [np.nan] + [0] * 11,
    ],
)
def test_bad_vector_init_rejected(bad, data):
    seq, alt, total = data
    model = CactriOmega(n_clones=3, accelerator="numpy", random_state=1)
    with pytest.raises((ValueError, TypeError)):
        model.prefit(seq, alt, total, init=bad)


def test_internal_bcr_consensus_init(data):
    seq, alt, total = data
    model = CactriOmega(n_clones=3, accelerator="numpy", random_state=4)
    model.prefit(
        seq,
        alt,
        total,
        init="bcr_consensus",
        bcr_initializer_config={
            "n_chains": 2,
            "n_iter": 6,
            "burn_in": 3,
            "thin": 1,
            "random_init_clusters": 3,
            "accelerator": "numpy",
        },
    )
    assert model.assignments_.shape == (12,)
    assert hasattr(model, "bcr_initializer_")
