import numpy as np
from sklearn.linear_model import Ridge
from volrisk.stream_ridge import StreamRidge,batches,mask_statistics


def test_stream_ridge_matches_regularized_intercept_solution():
    rng=np.random.default_rng(42);x=rng.normal(size=(100,3));y=rng.normal(size=100);idx=np.arange(3,95)
    xx=np.concatenate(list(batches(x,idx,3)))
    for alpha in [1.,100.,10000.]:
        expected=Ridge(alpha=alpha).fit(xx,y[idx])
        actual=StreamRidge(alpha).fit_statistics(StreamRidge.statistics(x,idx,y,3))
        np.testing.assert_allclose(actual.predict(x,idx,3),expected.predict(xx),atol=1e-11)


def test_static_information_and_masked_controls_match_explicit_designs():
    rng=np.random.default_rng(7)
    x=rng.normal(size=(120,3));static=rng.normal(size=(120,4));y=rng.normal(size=120);ids=np.arange(4,110)
    statistics=StreamRidge.statistics(x,ids,y,lookback=4,static=static)
    for active in [np.array([0,0,0,0]),np.array([0,0,1,1]),np.ones(4)]:
        keep=np.r_[np.ones(12),active]
        masked=static*active
        design=np.concatenate(list(batches(x,ids,4,static=masked)))
        reference=Ridge(alpha=10.).fit(design,y[ids])
        model=StreamRidge(10.).fit_statistics(mask_statistics(statistics,keep))
        np.testing.assert_allclose(model.predict(x,ids,4,static=masked),reference.predict(design),atol=1e-11)
