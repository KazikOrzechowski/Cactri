import inspect

import numpy as np
import pytest

from cactri import BCRInitializer, Cactri, CactriOmega, CactriTree, SplitMergeConfig, TrackingConfig


def test_public_imports_and_abstract_base():
    assert inspect.isabstract(Cactri)
    with pytest.raises(TypeError):
        Cactri(n_clones=2)
    assert TrackingConfig().every == 1
    assert BCRInitializer is not None
    assert CactriTree is not None
    assert CactriOmega is not None


def test_legacy_result_aliases():
    rng = np.random.default_rng(1)
    seq = rng.integers(0, 4, size=(10, 4))
    total = rng.integers(1, 5, size=(10, 3))
    alt = rng.binomial(total, 0.2)
    model = CactriOmega(n_clones=3, accelerator="numpy", random_state=2)
    model.prefit(seq, alt, total, init=np.repeat([10, 4], 5))
    result = model.fit(1)
    assert np.array_equal(result["genotype_matrix"], result["mutation_profile"])
    assert np.array_equal(result["cell_clone_assignment"], result["cell_clone_assignments"])
    assert np.array_equal(model.state_.assignments, model.assignments_)


def test_stage2_public_config():
    assert SplitMergeConfig().proposals_per_sweep == 1
    assert TrackingConfig.posterior().genotype_state
