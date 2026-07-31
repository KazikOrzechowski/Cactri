import numpy as np
import pytest

from cactri import CactriOmega, CactriTree, TrackingConfig


def make_data(seed=10):
    rng = np.random.default_rng(seed)
    seq = rng.integers(0, 4, size=(24, 6))
    total = rng.integers(1, 9, size=(24, 7))
    alt = rng.binomial(total, 0.2)
    return seq, alt, total


@pytest.mark.parametrize("sampler", ["approximate", "sequential", "split_merge"])
def test_omega_assignment_samplers(sampler):
    seq, alt, total = make_data()
    model = CactriOmega(n_clones=4, accelerator="numpy", random_state=11)
    model.prefit(seq, alt, total, init=np.repeat([0, 1, 2, 3], 6))
    result = model.fit(2, assignment_sampler=sampler)
    assert result["genotype_matrix"].shape == (4, 7)
    assert model.posterior_cell_genotype_probabilities().shape == (24, 7)


@pytest.mark.parametrize("tree_prior", ["uniform", "level_inverse"])
def test_tree_priors(tree_prior):
    seq, alt, total = make_data()
    model = CactriTree(
        n_levels=2,
        tree_prior=tree_prior,
        accelerator="numpy",
        random_state=12,
    )
    model.prefit(seq, alt, total, init=np.repeat([0, 1, 2, 3], 6))
    result = model.fit(3, tracking=TrackingConfig(genotype_state=True))
    assert result["mutation_tree_assignment"].shape == (7,)
    assert result["mutation_profile"].shape == (5, 7)
    assert model.posterior_cell_genotype_probabilities().shape == (24, 7)


def test_tree_learned_error_rate():
    seq, alt, total = make_data()
    observed = np.array([0, 1, 2, 3, 4, 5, 6])
    model = CactriTree(
        n_levels=2,
        observed_edge_assignment=observed,
        accelerator="numpy",
        random_state=13,
    )
    model.prefit(seq, alt, total, init=np.repeat([0, 1, 2, 3], 6))
    assert model.learn_edge_error_rate
    model.fit(3)
    assert 0 < model.edge_error_rate_ < 1
    assert len(model.edge_error_rate_trace_) == 3


def test_omega_reference_flag():
    seq, alt, total = make_data()
    fixed = CactriOmega(
        n_clones=4, fix_reference_clone=True, accelerator="numpy", random_state=14
    )
    fixed.prefit(seq, alt, total, init=np.repeat([0, 1, 2, 3], 6))
    fixed.fit(2)
    assert not fixed.genotype_matrix_[0].any()

    sampled = CactriOmega(
        n_clones=4, fix_reference_clone=False, accelerator="numpy", random_state=14
    )
    sampled.prefit(seq, alt, total, init=np.repeat([0, 1, 2, 3], 6))
    sampled.fit(2)
    assert sampled.genotype_matrix_.shape == (4, 7)


def test_fixed_mutation_edge_prior_vector_and_matrix():
    seq, alt, total = make_data()
    vector = np.arange(1, 8, dtype=float)
    vector_model = CactriTree(
        n_levels=2,
        mutation_edge_prior=vector,
        accelerator="numpy",
        random_state=15,
    )
    vector_model.prefit(seq, alt, total, init=np.repeat([0, 1, 2, 3], 6))
    assert vector_model.mutation_tree_assignment_.shape == (7,)

    matrix = np.broadcast_to(vector, (7, 7)).copy()
    matrix_model = CactriTree(
        n_levels=2,
        mutation_edge_prior=matrix,
        accelerator="numpy",
        random_state=15,
    )
    matrix_model.prefit(seq, alt, total, init=np.repeat([0, 1, 2, 3], 6))
    assert matrix_model.mutation_tree_assignment_.shape == (7,)
