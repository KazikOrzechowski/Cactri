import numpy as np
import pytest

from cactri import CactriOmega, CactriTree, TrackingConfig
from cactri._numba_accelerator import Accelerator, NUMBA_AVAILABLE


def test_kernel_algorithmic_identity():
    rng = np.random.default_rng(4)
    log_probs = rng.normal(size=(20, 7))
    uniforms = rng.random(20)
    numpy_acc = Accelerator("numpy")
    expected = numpy_acc.sample_rows(log_probs, uniforms)
    if not NUMBA_AVAILABLE:
        return
    numba_acc = Accelerator("numba")
    actual = numba_acc.sample_rows(log_probs, uniforms)
    np.testing.assert_array_equal(actual, expected)

    labels = rng.integers(0, 4, size=30)
    alt = rng.integers(0, 5, size=(30, 8)).astype(float)
    total = alt + rng.integers(0, 5, size=(30, 8))
    np_alt, np_ref = numpy_acc.aggregate_counts(labels, alt, total, 4)
    nb_alt, nb_ref = numba_acc.aggregate_counts(labels, alt, total, 4)
    np.testing.assert_array_equal(nb_alt, np_alt)
    np.testing.assert_array_equal(nb_ref, np_ref)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is not installed")
@pytest.mark.parametrize("kind", ["omega", "tree"])
def test_seeded_chain_algorithmic_identity(kind):
    rng = np.random.default_rng(5)
    seq = rng.integers(0, 4, size=(18, 5))
    total = rng.integers(1, 8, size=(18, 6))
    alt = rng.binomial(total, 0.2)
    init = np.repeat([0, 1, 2], 6)
    kwargs = {"n_clones": 4} if kind == "omega" else {"n_levels": 2}
    numpy_model = (
        CactriOmega(**kwargs, accelerator="numpy", random_state=9)
        if kind == "omega"
        else CactriTree(**kwargs, accelerator="numpy", random_state=9)
    )
    numba_model = (
        CactriOmega(**kwargs, accelerator="numba", random_state=9)
        if kind == "omega"
        else CactriTree(**kwargs, accelerator="numba", random_state=9)
    )
    numpy_model.prefit(seq, alt, total, init=init)
    numba_model.prefit(seq, alt, total, init=init)
    tracking = TrackingConfig(genotype_state=True, assignments=True)
    numpy_model.fit(4, tracking=tracking)
    numba_model.fit(4, tracking=tracking)

    np.testing.assert_array_equal(numba_model.assignments_, numpy_model.assignments_)
    np.testing.assert_array_equal(
        numba_model.current_cell_clone_assignment(),
        numpy_model.current_cell_clone_assignment(),
    )
    np.testing.assert_array_equal(
        numba_model._genotype_matrix(), numpy_model._genotype_matrix()
    )
    np.testing.assert_allclose(
        numba_model.p_obs_by_mutation_, numpy_model.p_obs_by_mutation_, rtol=0, atol=0
    )
    assert numba_model.p_unobs_ == numpy_model.p_unobs_
