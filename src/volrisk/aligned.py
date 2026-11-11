"""Execution-aligned target and validation-only calibration for versioned research.

This module does not mutate the original forecasting panel or demo evidence.
"""
import numpy as np
import pandas as pd


def execution_target(returns, horizon=20, execution_lag=1, annualization=252):
    """At close t, execute at t+lag; earn r[t+lag+1:t+lag+horizon]."""
    if horizon < 2 or execution_lag < 0:
        raise ValueError('Require horizon >= 2 and nonnegative execution lag')
    end = horizon + execution_lag
    return returns.rolling(horizon, min_periods=horizon).std(ddof=0).shift(-end) * np.sqrt(annualization)


def align_panel(panel, horizon=20, execution_lag=1, annualization=252):
    result = panel.sort_values(['ticker', 'date']).reset_index(drop=True).copy()
    result['original_target'] = result['target']
    for _, group in result.groupby('ticker', sort=False):
        idx = group.index
        result.loc[idx, 'target'] = execution_target(group['return'], horizon, execution_lag, annualization)
        result.loc[idx, 'target_end'] = group.date.shift(-(horizon + execution_lag))
        result.loc[idx, 'target_start'] = group.date.shift(-(execution_lag + 1))
        rv = group.rv20.clip(lower=1e-8)
        result.loc[idx, 'loghar_day'] = np.log(rv)
        result.loc[idx, 'loghar_week'] = np.log(rv.rolling(5).mean())
        result.loc[idx, 'loghar_month'] = np.log(rv.rolling(22).mean())
        result.loc[idx, 'context_ready'] = group.rv20.notna().rolling(512, min_periods=512).sum().eq(512)
    return result


def context_matrix(panel, indices, length=512):
    """Each context ends at its origin; reject cross-company or missing windows."""
    indices = np.asarray(indices, dtype=int)
    if not len(indices) or (indices < length-1).any():
        raise ValueError('Insufficient context')
    tickers = panel.ticker.to_numpy()
    if not np.array_equal(tickers[indices], tickers[indices-length+1]):
        raise ValueError('Context crosses company boundary')
    values = panel.rv20.to_numpy(dtype=float)
    result = np.stack([values[i-length+1:i+1] for i in indices])
    if not np.isfinite(result).all():
        raise ValueError('Context contains missing values')
    return result


def variance_calibration(actual, predicted):
    """Closed-form multiplicative QLIKE optimum, to be fitted on validation only."""
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if a.shape != p.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(p).all() or (a <= 0).any() or (p <= 0).any():
        raise ValueError('Calibration requires matched finite positive observations')
    return float(np.sqrt(np.mean((a / p) ** 2)))


def restore_ttm_prediction(output, means, scales, step=21, floor=1e-6):
    output = np.asarray(output)
    if output.ndim != 3 or output.shape[1] < step or output.shape[2] != 1:
        raise ValueError('Unexpected TTM output shape')
    raw = output[:, step-1, 0] * np.asarray(scales).reshape(-1) + np.asarray(means).reshape(-1)
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite TTM forecast')
    return np.maximum(raw, floor), int((raw < floor).sum())
