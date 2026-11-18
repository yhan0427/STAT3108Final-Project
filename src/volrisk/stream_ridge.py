"""Exact Ridge sufficient statistics in batches, avoiding a multi-GB lag matrix."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve


def mask_statistics(statistics,keep):
    """Exact sufficient statistics for setting selected input coordinates to zero."""
    xx,xy,mean,ym=statistics
    keep=np.asarray(keep,dtype=float)
    if keep.shape!=mean.shape or not np.isin(keep,[0.,1.]).all():
        raise ValueError('A binary feature-coordinate mask is required')
    return xx*keep[:,None]*keep[None,:],xy*keep,mean*keep,ym


def batches(values,indices,lookback=60,batch_size=4096,static=None):
    for start in range(0,len(indices),batch_size):
        ids=indices[start:start+batch_size]
        x=np.stack([values[i-lookback+1:i+1].reshape(-1) for i in ids]).astype(np.float64)
        if static is not None:
            x=np.concatenate([x,np.asarray(static[ids],dtype=np.float64)],axis=1)
        yield x


class StreamRidge:
    def __init__(self,alpha=1.):self.alpha=alpha
    @staticmethod
    def statistics(values,indices,target,lookback=60,static=None):
        dimension=values.shape[1]*lookback+(static.shape[1] if static is not None else 0)
        xx=np.zeros((dimension,dimension));xy=np.zeros(dimension);sx=np.zeros(dimension);sy=0.;n=0
        for x in batches(values,indices,lookback,static=static):
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
    def predict(self,values,indices,lookback=60,static=None):
        return np.concatenate([x@self.coef_+self.intercept_ for x in batches(values,indices,lookback,static=static)])
