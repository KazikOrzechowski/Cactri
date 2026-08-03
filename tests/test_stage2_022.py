from __future__ import annotations

import numpy as np
import pytest

from cactri import CactriOmega, CactriTree, SplitMergeConfig, TrackingConfig
from cactri._numba_accelerator import NUMBA_AVAILABLE


def make_data(seed: int = 522):
    rng = np.random.default_rng(seed)
    n_cells = 40
    n_snv = 9
    seq = rng.integers(0, 4, size=(n_cells, 6))
    total = rng.integers(2, 10, size=(n_cells, n_snv))
    alt = rng.binomial(total, 0.2)
    init = np.repeat(np.arange(4), 10)
    return seq, alt, total, init


def observed_probabilities(n_snv: int, n_vertices: int) -> np.ndarray:
    values = np.arange(1, n_snv * n_vertices + 1, dtype=float).reshape(
        n_snv, n_vertices
    )
    return values / values.sum(axis=1, keepdims=True)


def test_probabilistic_observed_edges_fit_and_serialize():
    seq, alt, total, init = make_data()
    probs = observed_probabilities(alt.shape[1], 7)
    model = CactriTree(
        n_levels=2,
        observed_edge_probabilities=probs,
        accelerator="numpy",
        random_state=31,
    )
    model.prefit(seq, alt, total, init=init)
    np.testing.assert_allclose(model.observed_edge_probabilities_.sum(axis=1), 1.0)
    model.fit(5, tracking=TrackingConfig(genotype_state=True))
    result = model.to_dict()
    assert 0 < model.edge_error_rate_ < 1
    assert result["observed_edge_probabilities"].shape == (9, 7)
    assert result["edge_error_component_trace"].ndim == 1


def test_probabilistic_and_point_observations_are_mutually_exclusive():
    probs = observed_probabilities(9, 7)
    with pytest.raises(ValueError, match="mutually exclusive"):
        CactriTree(
            n_levels=2,
            observed_edge_assignment=np.zeros(9, dtype=int),
            observed_edge_probabilities=probs,
        )


def fit_probabilistic(backend: str) -> CactriTree:
    seq, alt, total, init = make_data(523)
    probs = observed_probabilities(alt.shape[1], 7)
    model = CactriTree(
        n_levels=2,
        observed_edge_probabilities=probs,
        accelerator=backend,
        random_state=32,
    )
    model.prefit(seq, alt, total, init=init)
    model.fit(6, tracking=TrackingConfig.posterior())
    return model


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is not installed")
def test_probabilistic_edge_numpy_numba_algorithmic_identity():
    numpy_model = fit_probabilistic("numpy")
    numba_model = fit_probabilistic("numba")
    np.testing.assert_array_equal(
        numba_model.mutation_tree_assignment_, numpy_model.mutation_tree_assignment_
    )
    np.testing.assert_array_equal(numba_model.assignments_, numpy_model.assignments_)
    np.testing.assert_allclose(
        numba_model.p_obs_by_mutation_, numpy_model.p_obs_by_mutation_, rtol=0, atol=0
    )
    assert numba_model.edge_error_rate_ == numpy_model.edge_error_rate_
    assert (
        numba_model.edge_error_component_trace_
        == numpy_model.edge_error_component_trace_
    )


def test_bounded_split_merge_preserves_seeded_chain_and_limits_pairs():
    seq, alt, total, init = make_data(524)
    config = SplitMergeConfig(
        proposals_per_sweep=2,
        local_sampler="none",
        max_restricted_cells=25,
    )
    first = CactriOmega(n_clones=4, accelerator="numpy", random_state=33)
    second = CactriOmega(n_clones=4, accelerator="numpy", random_state=33)
    for model in (first, second):
        model.prefit(seq, alt, total, init=init)
        model.fit(
            5,
            assignment_sampler="split_merge",
            split_merge_config=config,
            tracking=TrackingConfig.posterior(),
        )
    np.testing.assert_array_equal(first.assignments_, second.assignments_)
    np.testing.assert_array_equal(first.genotype_matrix_, second.genotype_matrix_)
    assert first.split_merge_diagnostics() == second.split_merge_diagnostics()


def test_bounded_split_merge_rejects_adaptive_combination():
    with pytest.raises(ValueError, match="uniform_pair"):
        SplitMergeConfig(anchor_strategy="adaptive", max_restricted_cells=20)
