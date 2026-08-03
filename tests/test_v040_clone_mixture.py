from __future__ import annotations

import gzip
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

from cactri import (
    Cactri,
    CactriOmega,
    CactriTree,
    CloneMixtureConfig,
    TrackingConfig,
)


def data(seed: int = 17, n: int = 36, m: int = 7, length: int = 6):
    rng = np.random.default_rng(seed)
    seq = rng.integers(0, 4, size=(n, length), dtype=np.int64)
    total = rng.poisson(4.0, size=(n, m)).astype(np.int64)
    latent = rng.integers(0, 3, size=n)
    genotype = rng.integers(0, 2, size=(3, m))
    probability = np.where(genotype[latent] == 1, 0.45, 0.01)
    alt = rng.binomial(total, probability).astype(np.int64)
    return seq, alt, total


def make_omega(backend: str = "numpy", seed: int = 91) -> CactriOmega:
    seq, alt, total = data()
    model = CactriOmega(
        n_clones=4,
        clone_mixture=CloneMixtureConfig(enabled=True),
        accelerator=backend,
        random_state=seed,
    )
    model.prefit(seq, alt, total, init="random", random_init_clusters=4)
    return model


def make_tree(backend: str = "numpy", seed: int = 92) -> CactriTree:
    seq, alt, total = data(seed=18)
    model = CactriTree(
        n_levels=2,
        clone_mixture=CloneMixtureConfig(enabled=True),
        accelerator=backend,
        random_state=seed,
    )
    model.prefit(seq, alt, total, init="random", random_init_clusters=4)
    return model


def test_clone_mixture_config_validation():
    assert not CloneMixtureConfig().enabled
    assert CloneMixtureConfig(enabled=True).admixture_mass_prior == (1.0, 19.0)
    with pytest.raises(ValueError):
        CloneMixtureConfig(residual_concentration=0.0)
    with pytest.raises(ValueError):
        CloneMixtureConfig(admixture_mass_prior=(1.0, 0.0))
    with pytest.raises(ValueError):
        CloneMixtureConfig(residual_base="bad")  # type: ignore[arg-type]


def test_disabled_mode_has_exact_v02_state_representation():
    seq, alt, total = data()
    model = CactriOmega(n_clones=4, random_state=4, accelerator="numpy")
    model.prefit(seq, alt, total, init="random", random_init_clusters=4)
    expected = model.hypercluster_to_clone_[model.assignments_]
    np.testing.assert_array_equal(model.cell_clone_assignments_, expected)
    np.testing.assert_array_equal(model.current_cell_clone_assignment(), expected)
    np.testing.assert_allclose(model.hypercluster_clone_proportions_.sum(axis=1), 1.0)
    assert not model.mixture_active_.any()
    assert model.mixture_presence_rate_ == 0.0


def test_exact_zero_pure_formulation_is_finite_and_normalized():
    model = make_omega()
    assert not model.mixture_active_.any()
    proportions = model.hypercluster_clone_proportions_
    assert np.all((proportions == 0.0) | (proportions == 1.0))
    np.testing.assert_allclose(proportions.sum(axis=1), 1.0)
    assert np.isfinite(model.log_likelihood())
    assert np.isfinite(model.log_posterior())
    probability = model.current_state_cell_clone_probabilities()
    assert np.isfinite(probability).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)


def test_collapsed_active_mass_matches_manual_beta_dirichlet_integral():
    model = make_omega()
    counts = np.array([8, 2, 1, 0], dtype=np.int64)
    dominant = 0
    actual = model._collapsed_active_mixture_log_mass(counts, dominant)
    a, b = model.clone_mixture.admixture_mass_prior
    n_dom = counts[dominant]
    n_other = counts.sum() - n_dom
    expected = model._log_beta_function(a + n_other, b + n_dom)
    expected -= model._log_beta_function(a, b)
    mask = np.arange(model.n_clones) != dominant
    alpha = (
        model.clone_mixture.residual_concentration
        * model._residual_base_weights(dominant)[mask]
    )
    residual = counts[mask]
    expected += math.lgamma(alpha.sum()) - math.lgamma(
        alpha.sum() + residual.sum()
    )
    expected += sum(
        math.lgamma(x + n) - math.lgamma(x)
        for x, n in zip(alpha, residual, strict=True)
    )
    assert actual == pytest.approx(expected)


def test_mixture_cell_update_and_marginal_assignment_scores_are_normalized():
    model = make_omega()
    model.mixture_active_[:] = True
    model.admixture_mass_[:] = 0.1
    model.hypercluster_clone_proportions_[:] = 0.1 / (model.n_clones - 1)
    model.hypercluster_clone_proportions_[
        np.arange(model.n_hyperclusters), model.dominant_clones_
    ] = 0.9
    model._sample_cell_clone_assignments()
    assert model.cell_clone_assignments_.shape == (model.n_cells,)
    marginal = model._cell_hypercluster_mut_loglikelihood()
    assert marginal.shape == (model.n_cells, model.n_hyperclusters)
    assert np.isfinite(marginal).all()


def test_mixture_tracking_and_posterior_summaries_include_clone_coclustering():
    model = make_tree()
    tracking = TrackingConfig.posterior(every=1, mixture_diagnostics=True)
    model.fit(6, tracking=tracking)
    assert len(model.dominant_clone_trace_) == 6
    assert len(model.hypercluster_clone_proportions_trace_) == 6
    cell_co = model.posterior_cell_clone_coassignment()
    np.testing.assert_allclose(cell_co, cell_co.T)
    np.testing.assert_allclose(np.diag(cell_co), 1.0)
    assert model.posterior_hypercluster_clone_proportions().shape == (
        model.n_hyperclusters,
        model.n_clones,
    )
    assert model.posterior_dominant_clone_probabilities().shape == (
        model.n_hyperclusters,
        model.n_clones,
    )
    assert model.posterior_admixture_probabilities().shape == (
        model.n_hyperclusters,
    )
    assert model.posterior_effective_clone_counts().shape == (
        model.n_hyperclusters,
    )


def test_tree_distance_residual_base_prefers_nearby_leaves():
    seq, alt, total = data(n=20)
    model = CactriTree(
        n_levels=2,
        clone_mixture=CloneMixtureConfig(
            enabled=True,
            residual_base="tree_distance",
            tree_distance_decay=1.0,
        ),
        random_state=1,
        accelerator="numpy",
    )
    model.prefit(seq, alt, total, init="random", random_init_clusters=3)
    weights = model._residual_base_weights(1)
    assert weights[1] == 0.0
    assert weights[2] > weights[3]
    assert weights[2] > weights[0]
    np.testing.assert_allclose(weights.sum(), 1.0)


def test_tree_distance_residual_base_is_rejected_for_omega():
    seq, alt, total = data(n=20)
    model = CactriOmega(
        n_clones=4,
        clone_mixture=CloneMixtureConfig(
            enabled=True, residual_base="tree_distance"
        ),
        random_state=1,
    )
    with pytest.raises(ValueError, match="CactriTree"):
        model.prefit(seq, alt, total)


def test_mixture_checkpoint_resume_is_exact(tmp_path: Path):
    tracking = TrackingConfig.posterior(mixture_diagnostics=True)
    uninterrupted = make_omega(seed=120)
    uninterrupted.fit(8, tracking=tracking)

    partial = make_omega(seed=120)
    partial.fit(3, tracking=tracking)
    path = partial.save_checkpoint(tmp_path / "mixture.cactri.gz")
    resumed = CactriOmega.load_checkpoint(path)
    resumed.fit(5, tracking=tracking)

    for name in (
        "assignments_",
        "cell_clone_assignments_",
        "dominant_clones_",
        "hypercluster_clone_proportions_",
        "mixture_active_",
        "genotype_matrix_",
        "p_obs_by_mutation_",
    ):
        np.testing.assert_array_equal(getattr(resumed, name), getattr(uninterrupted, name))
    assert resumed.p_unobs_ == uninterrupted.p_unobs_
    assert resumed.mixture_presence_rate_ == uninterrupted.mixture_presence_rate_
    np.testing.assert_array_equal(
        resumed.cell_clone_assignment_trace_, uninterrupted.cell_clone_assignment_trace_
    )


def test_v02_checkpoint_state_migrates_to_exact_pure_mixture(tmp_path: Path):
    model = make_omega()
    model.clone_mixture = CloneMixtureConfig(enabled=False)
    for name in (
        "cell_clone_assignments_",
        "hypercluster_clone_proportions_",
        "admixture_mass_",
        "residual_clone_proportions_",
        "mixture_active_",
        "mixture_presence_rate_",
    ):
        delattr(model, name)
    payload = {
        "format": "cactri-checkpoint",
        "format_version": 1,
        "package_version": "0.2.2",
        "model": model,
    }
    path = tmp_path / "old.cactri.gz"
    with gzip.open(path, "wb") as handle:
        pickle.dump(payload, handle)
    loaded = Cactri.load_checkpoint(path)
    assert not loaded.clone_mixture.enabled
    np.testing.assert_array_equal(
        loaded.cell_clone_assignments_,
        loaded.hypercluster_to_clone_[loaded.assignments_],
    )
    assert not loaded.mixture_active_.any()
    np.testing.assert_allclose(
        loaded.hypercluster_clone_proportions_.sum(axis=1), 1.0
    )


def test_v03_checkpoint_is_rejected_with_lineage_error(tmp_path: Path):
    model = make_omega()
    payload = {
        "format": "cactri-checkpoint",
        "format_version": 1,
        "package_version": "0.3.1",
        "model": model,
    }
    path = tmp_path / "v03.cactri.gz"
    with gzip.open(path, "wb") as handle:
        pickle.dump(payload, handle)
    with pytest.raises(ValueError, match="discontinued experimental"):
        Cactri.load_checkpoint(path)


@pytest.mark.parametrize("factory", [make_omega, make_tree])
def test_complete_mixture_chain_numpy_numba_identity(factory):
    numpy_model = factory("numpy", seed=222)
    numba_model = factory("numba", seed=222)
    tracking = TrackingConfig.posterior(mixture_diagnostics=True)
    numpy_model.fit(7, tracking=tracking)
    numba_model.fit(7, tracking=tracking)
    for name in (
        "assignments_",
        "cell_clone_assignments_",
        "dominant_clones_",
        "hypercluster_clone_proportions_",
        "admixture_mass_",
        "residual_clone_proportions_",
        "mixture_active_",
        "p_obs_by_mutation_",
        "cell_clone_assignment_trace_",
        "genotype_state_trace_",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(numpy_model, name), dtype=object),
            np.asarray(getattr(numba_model, name), dtype=object),
        )
    assert numpy_model.p_unobs_ == numba_model.p_unobs_
    assert numpy_model.mixture_presence_rate_ == numba_model.mixture_presence_rate_


def test_pure_and_admixed_structure_transitions_are_numerically_safe():
    model = make_omega(seed=331)
    model.mixture_presence_rate_ = 1.0 - model.eps
    model._sample_clone_mixture_state()
    assert model.mixture_active_.all()
    np.testing.assert_allclose(model.hypercluster_clone_proportions_.sum(axis=1), 1.0)
    assert np.all(model.hypercluster_clone_proportions_ > 0.0)

    model.cell_clone_assignments_ = model.dominant_clones_[model.assignments_].copy()
    model.mixture_presence_rate_ = model.eps
    model._sample_clone_mixture_state()
    assert not model.mixture_active_.any()
    assert np.all(
        (model.hypercluster_clone_proportions_ == 0.0)
        | (model.hypercluster_clone_proportions_ == 1.0)
    )
    assert np.isfinite(model.log_posterior())


def test_sequential_assignment_keeps_mixture_state_shapes_consistent():
    model = make_omega(seed=442)
    model.fit(4, assignment_sampler="sequential")
    assert model.dominant_clones_.shape == (model.n_hyperclusters,)
    assert model.hypercluster_clone_proportions_.shape == (
        model.n_hyperclusters,
        model.n_clones,
    )
    assert model.admixture_mass_.shape == (model.n_hyperclusters,)
    np.testing.assert_allclose(
        model.hypercluster_clone_proportions_.sum(axis=1), 1.0
    )


def test_split_merge_is_explicitly_unavailable_for_clone_mixtures():
    model = make_omega()
    with pytest.raises(ValueError, match="not defined for clone mixtures"):
        model.fit(1, assignment_sampler="split_merge")


def test_alpha_sampling_defaults_follow_mixture_opt_in():
    seq, alt, total = data()
    legacy = CactriOmega(n_clones=4, random_state=3)
    legacy.prefit(seq, alt, total)
    assert not legacy.sample_alpha
    original = legacy.alpha_
    legacy.fit(3)
    assert legacy.alpha_ == original

    mixture = CactriOmega(
        n_clones=4,
        clone_mixture=CloneMixtureConfig(enabled=True),
        random_state=3,
    )
    mixture.prefit(seq, alt, total)
    assert mixture.sample_alpha
    original = mixture.alpha_
    mixture.fit(3)
    assert mixture.alpha_ != original
    assert mixture.alpha_ > 0.0
