"""Exact content-token coverage and document pooling, independent of model fitting."""
import numpy as np


def chunk_token_ids(ids, content_limit=510):
    if content_limit < 1 or not len(ids):
        raise ValueError("Non-empty content and a positive token limit required")
    return [ids[start:start + content_limit] for start in range(0, len(ids), content_limit)]


def weighted_document_mean(chunk_means, token_counts):
    values = np.asarray(chunk_means, dtype=np.float64)
    counts = np.asarray(token_counts, dtype=np.float64)
    if values.ndim != 2 or counts.shape != (len(values),):
        raise ValueError("One positive content count is required per chunk")
    if not np.isfinite(values).all() or not np.isfinite(counts).all() or not (counts > 0).all():
        raise ValueError("Invalid chunk values/counts")
    return np.average(values, axis=0, weights=counts).astype(np.float32)
