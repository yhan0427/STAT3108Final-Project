"""Exact Ridge sufficient statistics in batches, avoiding a multi-GB lag matrix."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve


def batches(values,indices,lookback=60,batch_size=4096):
    for start in range(0,len(indices),batch_size):
        ids=indices[start:start+batch_size]
        yield np.stack([values[i-lookback+1:i+1].reshape(-1) for i in ids]).astype(np.float64)


class StreamRidge:
    def __init__(self,alpha=1.):self.alpha=alpha
    @staticmethod
    def statistics(values,indices,target,lookback=60):
        dimension=values.shape[1]*lookback
        xx=np.zeros((dimension,dimension));xy=np.zeros(dimension);sx=np.zeros(dimension);sy=0.;n=0
        for x in batches(values,indices,lookback):
            y=np.asarray(target[indices[n:n+len(x)]],dtype=float)
            if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite training input')
            xx+=x.T@x;xy+=x.T@y;sx+=x.sum(axis=0);sy+=y.sum();n+=len(x)
        mean=sx/n;ym=sy/n
        return xx-n*np.outer(mean,mean),xy-n*mean*ym,mean,ym
    def fit_statistics(self,statistics):
        xx,xy,mean,ym=statistics
        self.coef_=cho_solve(cho_factor(xx+self.alpha*np.eye(len(xy)),lower=True),xy)
        self.intercept_=float(ym-mean@self.coef_)
        return self
    def predict(self,values,indices,lookback=60):
        return np.concatenate([x@self.coef_+self.intercept_ for x in batches(values,indices,lookback)])
