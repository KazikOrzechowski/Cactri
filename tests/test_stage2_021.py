from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cactri import Cactri, CactriOmega, SplitMergeConfig, TrackingConfig
from cactri._numba_accelerator import NUMBA_AVAILABLE
from cactri.utils.posterior import partition_medoid_from_trace


def make_data(seed: int = 404):
    rng = np.random.default_rng(seed)
    n_cells = 36
    seq = rng.integers(0, 4, size=(n_cells, 6))
    total = rng.integers(3, 10, size=(n_cells, 8))
    alt = rng.binomial(total, 0.2)
    init = np.repeat([0, 1, 2], 12)
    return seq, alt, total, init


def fit_with_cache(cache: bool) -> CactriOmega:
    seq, alt, total, init = make_data()
    model = CactriOmega(n_clones=4, accelerator="numpy", random_state=19)
    model.prefit(seq, alt, total, init=init)
    model.fit(
        6,
        assignment_sampler="split_merge",
        split_merge_config=SplitMergeConfig(
            proposals_per_sweep=3,
            local_sampler="none",
            cache_sufficient_statistics=cache,
        ),
        tracking=TrackingConfig.posterior(),
    )
    return model


def test_cached_sufficient_statistics_preserve_chain():
    uncached = fit_with_cache(False)
    cached = fit_with_cache(True)
    np.testing.assert_array_equal(cached.assignments_, uncached.assignments_)
    np.testing.assert_array_equal(cached.genotype_matrix_, uncached.genotype_matrix_)
    np.testing.assert_array_equal(
        np.asarray(cached.assignment_trace_), np.asarray(uncached.assignment_trace_)
    )
    np.testing.assert_allclose(cached.p_obs_by_mutation_, uncached.p_obs_by_mutation_)
    assert cached.split_merge_diagnostics()["cache_reuses"] > 0
    assert cached.split_merge_diagnostics()["cache_builds"] < uncached.split_merge_diagnostics()["cache_builds"]


def test_adaptive_anchor_scheduler_updates_and_is_bounded():
    seq, alt, total, init = make_data()
    config = SplitMergeConfig(
        proposals_per_sweep=4,
        local_sampler="none",
        anchor_strategy="adaptive",
        adaptation_interval=4,
        adaptation_step=0.5,
        min_split_probability=0.2,
        max_split_probability=0.8,
    )
    model = CactriOmega(n_clones=4, accelerator="numpy", random_state=20)
    model.prefit(seq, alt, total, init=init)
    model.fit(8, assignment_sampler="split_merge", split_merge_config=config)
    diagnostics = model.split_merge_diagnostics()
    assert diagnostics["attempts"] == 32
    assert 0.2 <= diagnostics["adaptive_split_probability"] <= 0.8
    assert diagnostics["split_attempts"] > 0
    assert diagnostics["merge_attempts"] > 0


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is not installed")
def test_adaptive_scheduler_numpy_numba_algorithmic_identity():
    seq, alt, total, init = make_data(405)
    config = SplitMergeConfig(
        proposals_per_sweep=3,
        local_sampler="none",
        anchor_strategy="adaptive",
        adaptation_interval=3,
    )
    models = []
    for backend in ("numpy", "numba"):
        model = CactriOmega(n_clones=4, accelerator=backend, random_state=21)
        model.prefit(seq, alt, total, init=init)
        model.fit(
            6,
            assignment_sampler="split_merge",
            split_merge_config=config,
            tracking=TrackingConfig.posterior(),
        )
        models.append(model)
    numpy_model, numba_model = models
    np.testing.assert_array_equal(numba_model.assignments_, numpy_model.assignments_)
    np.testing.assert_array_equal(numba_model.genotype_matrix_, numpy_model.genotype_matrix_)
    np.testing.assert_array_equal(
        np.asarray(numba_model.assignment_trace_),
        np.asarray(numpy_model.assignment_trace_),
    )
    assert numba_model.split_merge_diagnostics() == numpy_model.split_merge_diagnostics()


def test_checkpoint_resume_is_exact_and_preserves_tracking_cadence(tmp_path: Path):
    seq, alt, total, init = make_data(406)
    config = SplitMergeConfig(
        proposals_per_sweep=2,
        local_sampler="none",
        anchor_strategy="adaptive",
        adaptation_interval=4,
    )
    tracking = TrackingConfig.posterior(every=2)

    uninterrupted = CactriOmega(n_clones=4, accelerator="numpy", random_state=22)
    uninterrupted.prefit(seq, alt, total, init=init)
    uninterrupted.fit(
        10,
        assignment_sampler="split_merge",
        split_merge_config=config,
        tracking=tracking,
    )

    partial = CactriOmega(n_clones=4, accelerator="numpy", random_state=22)
    partial.prefit(seq, alt, total, init=init)
    partial.fit(
        5,
        assignment_sampler="split_merge",
        split_merge_config=config,
        tracking=tracking,
    )
    checkpoint = partial.save_checkpoint(tmp_path / "chain.cactri.gz")
    resumed = Cactri.load_checkpoint(checkpoint)
    resumed.fit(
        5,
        assignment_sampler="split_merge",
        split_merge_config=config,
        tracking=tracking,
    )

    assert isinstance(resumed, CactriOmega)
    np.testing.assert_array_equal(resumed.assignments_, uninterrupted.assignments_)
    np.testing.assert_array_equal(resumed.genotype_matrix_, uninterrupted.genotype_matrix_)
    np.testing.assert_allclose(resumed.p_obs_by_mutation_, uninterrupted.p_obs_by_mutation_)
    np.testing.assert_array_equal(
        np.asarray(resumed.assignment_trace_),
        np.asarray(uninterrupted.assignment_trace_),
    )
    assert resumed._iterations_completed_ == 10
    assert len(resumed.assignment_trace_) == 5
    assert resumed.split_merge_diagnostics() == uninterrupted.split_merge_diagnostics()


def test_checkpoint_can_omit_large_traces(tmp_path: Path):
    seq, alt, total, init = make_data(407)
    model = CactriOmega(n_clones=4, accelerator="numpy", random_state=23)
    model.prefit(seq, alt, total, init=init)
    model.fit(4, tracking=TrackingConfig.posterior())
    checkpoint = model.save_checkpoint(
        tmp_path / "no_traces.cactri.gz", include_traces=False
    )
    loaded = CactriOmega.load_checkpoint(checkpoint)
    assert loaded._iterations_completed_ == 4
    assert loaded.assignment_trace_ == []
    assert loaded.cell_clone_assignment_trace_ == []
    assert loaded.genotype_state_trace_ == []
    np.testing.assert_array_equal(loaded.assignments_, model.assignments_)


def test_partition_medoid_helpers_are_label_invariant():
    trace = [
        np.array([0, 0, 1, 1]),
        np.array([4, 4, 9, 9]),
        np.array([0, 1, 1, 1]),
    ]
    assignment, index, losses = partition_medoid_from_trace(trace)
    assert index in {0, 1}
    assert np.array_equal(assignment[:, None] == assignment[None, :], trace[0][:, None] == trace[0][None, :])
    assert losses.shape == (3,)

    seq, alt, total, init = make_data(408)
    model = CactriOmega(n_clones=4, accelerator="numpy", random_state=24)
    model.prefit(seq, alt, total, init=init)
    model.fit(5, tracking=TrackingConfig.posterior())
    summary = model.posterior_summary(include_partition_medoids=True)
    assert summary["hypercluster_partition_medoid"]["assignment"].shape == (36,)
    assert summary["clone_partition_medoid"]["assignment"].shape == (36,)
