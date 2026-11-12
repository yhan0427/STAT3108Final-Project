import numpy as np
from sklearn.linear_model import Ridge
from volrisk.stream_ridge import StreamRidge,batches


def test_stream_ridge_matches_regularized_intercept_solution():
    rng=np.random.default_rng(42);x=rng.normal(size=(100,3));y=rng.normal(size=100);idx=np.arange(3,95)
    xx=np.concatenate(list(batches(x,idx,3)))
    for alpha in [1.,100.,10000.]:
        expected=Ridge(alpha=alpha).fit(xx,y[idx])
        actual=StreamRidge(alpha).fit_statistics(StreamRidge.statistics(x,idx,y,3))
        np.testing.assert_allclose(actual.predict(x,idx,3),expected.predict(xx),atol=1e-11)
