import numpy as np
import pytest

from cactri import (
    CactriOmega,
    CactriTree,
    SplitMergeConfig,
    TrackingConfig,
)
from cactri._numba_accelerator import NUMBA_AVAILABLE


def make_data(seed: int = 101):
    rng = np.random.default_rng(seed)
    n_cells = 30
    seq = np.empty((n_cells, 5), dtype=np.int64)
    seq[:15] = rng.choice([0, 1], size=(15, 5), p=[0.9, 0.1])
    seq[15:] = rng.choice([2, 3], size=(15, 5), p=[0.1, 0.9])
    total = rng.integers(4, 10, size=(n_cells, 6))
    truth = np.repeat([0, 1], 15)
    genotype = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=np.int8,
    )
    clone = truth + 1
    p = np.where(genotype[clone] == 1, 0.5, 0.001)
    alt = rng.binomial(total, p)
    return seq, alt, total


def test_stage2_default_absent_prior():
    model = CactriOmega(n_clones=3, accelerator="numpy")
    assert model.p_unobs_beta_prior == (1.0, 999.0)


@pytest.mark.parametrize("kind", ["omega", "tree"])
def test_collapsed_split_merge_is_valid_and_tracks_diagnostics(kind):
    seq, alt, total = make_data()
    model = (
        CactriOmega(n_clones=3, accelerator="numpy", random_state=7)
        if kind == "omega"
        else CactriTree(n_levels=1, accelerator="numpy", random_state=7)
    )
    model.prefit(seq, alt, total, init=np.zeros(seq.shape[0], dtype=int))
    model.fit(
        8,
        assignment_sampler="split_merge",
        split_merge_config=SplitMergeConfig(proposals_per_sweep=3, local_sampler="none"),
        update_bcr_profiles=False,
        update_hypercluster_clones=False,
        update_genotypes=False,
        update_observation_probabilities=False,
    )
    diagnostics = model.split_merge_diagnostics()
    assert diagnostics["attempts"] == 24
    assert diagnostics["split_attempts"] + diagnostics["merge_attempts"] == 24
    assert np.array_equal(np.unique(model.assignments_), np.arange(model.n_hyperclusters))
    assert model.bcr_profiles_.shape[0] == model.n_hyperclusters
    assert model.hypercluster_to_clone_.shape == (model.n_hyperclusters,)


def test_coherent_posterior_summary_shapes_and_identities():
    seq, alt, total = make_data()
    model = CactriOmega(n_clones=3, accelerator="numpy", random_state=11)
    model.prefit(seq, alt, total, init=np.repeat([0, 1], 15))
    model.fit(6, tracking=TrackingConfig.posterior())

    summary = model.posterior_summary(burn_in=0.5)
    assert summary["cell_clone_probabilities"].shape == (30, 3)
    assert summary["cell_genotype_probabilities"].shape == (30, 6)
    assert summary["hypercluster_coassignment"].shape == (30, 30)
    np.testing.assert_allclose(summary["cell_clone_probabilities"].sum(axis=1), 1.0)
    np.testing.assert_allclose(np.diag(summary["hypercluster_coassignment"]), 1.0)
    assert np.all((summary["cell_genotype_probabilities"] >= 0) & (summary["cell_genotype_probabilities"] <= 1))

    expected = []
    for clones, genotype in zip(
        model.cell_clone_assignment_trace_[3:], model.genotype_state_trace_[3:]
    ):
        expected.append(genotype[clones])
    np.testing.assert_allclose(
        summary["cell_genotype_probabilities"], np.mean(expected, axis=0)
    )


def test_trace_requirement_is_explicit():
    seq, alt, total = make_data()
    model = CactriOmega(n_clones=3, accelerator="numpy", random_state=12)
    model.prefit(seq, alt, total, init=np.repeat([0, 1], 15))
    with pytest.raises(RuntimeError):
        model.posterior_cell_genotype_probabilities(use_trace=True)
    assert model.posterior_cell_genotype_probabilities(use_trace=False).shape == (30, 6)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is not installed")
@pytest.mark.parametrize("kind", ["omega", "tree"])
def test_split_merge_chain_algorithmic_identity(kind):
    seq, alt, total = make_data(202)
    kwargs = {"n_clones": 3} if kind == "omega" else {"n_levels": 1}
    cls = CactriOmega if kind == "omega" else CactriTree
    numpy_model = cls(**kwargs, accelerator="numpy", random_state=23)
    numba_model = cls(**kwargs, accelerator="numba", random_state=23)
    init = np.repeat([0, 1], 15)
    numpy_model.prefit(seq, alt, total, init=init)
    numba_model.prefit(seq, alt, total, init=init)
    config = SplitMergeConfig(proposals_per_sweep=2, local_sampler="sequential")
    tracking = TrackingConfig.posterior()
    numpy_model.fit(5, assignment_sampler="split_merge", split_merge_config=config, tracking=tracking)
    numba_model.fit(5, assignment_sampler="split_merge", split_merge_config=config, tracking=tracking)

    np.testing.assert_array_equal(numba_model.assignments_, numpy_model.assignments_)
    np.testing.assert_array_equal(
        numba_model.current_cell_clone_assignment(),
        numpy_model.current_cell_clone_assignment(),
    )
    np.testing.assert_array_equal(numba_model._genotype_matrix(), numpy_model._genotype_matrix())
    np.testing.assert_array_equal(
        np.asarray(numba_model.assignment_trace_), np.asarray(numpy_model.assignment_trace_)
    )
    assert numba_model.split_merge_diagnostics() == numpy_model.split_merge_diagnostics()
