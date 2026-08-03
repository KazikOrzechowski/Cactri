import numpy as np
from cactri import CactriTree, CloneMixtureConfig, TrackingConfig

def test_fit_does_not_eagerly_materialize_dense_coassignment():
    rng=np.random.default_rng(3)
    n=30; m=4
    seq=rng.integers(0,4,size=(n,5))
    total=np.full((n,m),10,dtype=int)
    alt=rng.binomial(total,0.1)
    model=CactriTree(n_levels=1, clone_mixture=CloneMixtureConfig(enabled=True), random_state=4)
    model.prefit(seq,alt,total,init='random')
    result=model.fit(3,tracking=TrackingConfig.posterior(every=1))
    assert 'posterior_hypercluster_coassignment' not in result
    assert 'posterior_cell_clone_coassignment' not in result
    assert model.posterior_hypercluster_coassignment().shape==(n,n)
    assert model.posterior_cell_clone_coassignment().shape==(n,n)

def test_active_residual_simplex_is_strictly_positive_and_logprior_finite():
    rng=np.random.default_rng(8)
    n=40; m=5
    seq=rng.integers(0,4,size=(n,6))
    total=np.full((n,m),12,dtype=int)
    alt=rng.binomial(total,0.15)
    model=CactriTree(
        n_levels=2,
        clone_mixture=CloneMixtureConfig(
            enabled=True,
            allow_pure_hyperclusters=False,
            residual_concentration=0.05,
        ),
        random_state=9,
    )
    model.prefit(seq,alt,total,init='random')
    model.fit(10,tracking=TrackingConfig(every=20))
    for h,d in enumerate(model.dominant_clones_):
        mask=np.arange(model.n_clones)!=d
        assert np.all(model.residual_clone_proportions_[h,mask] > 0)
    assert np.isfinite(model._clone_mixture_log_prior())
