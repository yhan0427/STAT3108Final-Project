"""Past-only annual-document joins and training-document-only reduction."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def align_documents(dates, documents, original_filing_dates=None, stale_days=550, entity_start=None):
    """One issuer at a time; document_id is a preassigned global matrix coordinate."""
    left = pd.DataFrame({"date": pd.to_datetime(dates)}).sort_values("date").reset_index(drop=True)
    fields = ["filing_date", "report_period", "accession", "document_id", "content_tokens"]
    if len(documents):
        docs = documents[fields].copy()
        docs["filing_date"] = pd.to_datetime(docs.filing_date)
        if docs.filing_date.max() >= pd.Timestamp("2026-01-01"):
            raise ValueError("Later disclosures cannot enter this sealed-stage document join")
        if entity_start is not None:
            docs = docs.loc[docs.filing_date.ge(pd.Timestamp(entity_start))]
        docs = docs.sort_values(["filing_date", "report_period", "accession"])
        joined = pd.merge_asof(left, docs, left_on="date", right_on="filing_date",
                               direction="backward", allow_exact_matches=False)
    else:
        joined = left.copy()
        joined["filing_date"] = pd.NaT
        for key in fields[1:]:
            joined[key] = np.nan
    joined["text_age_days"] = (joined.date - joined.filing_date).dt.days.astype(float)
    valid = joined.document_id.notna() & joined.text_age_days.le(stale_days)
    joined["source_document_id"] = joined.document_id.fillna(-1).astype(int)
    joined["text_document_id"] = joined.source_document_id.where(valid, -1)
    joined["text_available"] = valid.astype(float)
    joined["text_log_tokens"] = np.log1p(joined.content_tokens)
    joined["newer_original_unavailable"] = 0.
    if original_filing_dates is not None and len(original_filing_dates):
        all_dates = pd.DataFrame({"latest_original_filing_date": pd.to_datetime(original_filing_dates)}).drop_duplicates().sort_values("latest_original_filing_date")
        if all_dates.latest_original_filing_date.max() >= pd.Timestamp("2026-01-01"):
            raise ValueError("Later original filings in join metadata")
        if entity_start is not None:
            all_dates = all_dates.loc[all_dates.latest_original_filing_date.ge(pd.Timestamp(entity_start))]
        latest = pd.merge_asof(left, all_dates, left_on="date", right_on="latest_original_filing_date",
                               direction="backward", allow_exact_matches=False)
        joined["newer_original_unavailable"] = (
            latest.latest_original_filing_date.notna()
            & (joined.filing_date.isna() | latest.latest_original_filing_date.gt(joined.filing_date))).astype(float)
    return joined


def fit_training_document_pca(embeddings, documents, origins, document_ids, training_mask, width=32):
    """Return a fitted reducer and zero-padded matrix; repeated daily joins do not reweight PCA."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    ids = np.asarray(document_ids, dtype=int)
    training = np.asarray(training_mask, dtype=bool)
    if ids.shape != (len(origins),) or training.shape != ids.shape or matrix.ndim != 2 or len(documents) != len(matrix):
        raise ValueError("Inconsistent document/origin dimensions")
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite document embedding")
    if not origins.loc[training, "role"].eq("training_company").all() or pd.to_datetime(origins.loc[training, "date"]).max() >= pd.Timestamp("2022-01-01"):
        raise ValueError("Reduction may use only training-role, pre-2022 origins")
    used_rows = np.flatnonzero(training & (ids >= 0))
    used = np.unique(ids[used_rows])
    if len(used) < 2:
        raise ValueError("Insufficient unique training documents for reduction")
    if ids[used_rows].max() >= len(matrix):
        raise ValueError("Invalid document matrix coordinate")
    selected = documents.iloc[used]
    if not selected.role.eq("training_company").all():
        raise ValueError("Held-company document entered reduction")
    if "entity_start" in selected:
        boundary = pd.to_datetime(selected.entity_start, errors="coerce")
        if (boundary.notna() & pd.to_datetime(selected.filing_date).lt(boundary)).any():
            raise ValueError("Pre-spell document entered training reduction")
    disclosed = pd.to_datetime(documents.iloc[ids[used_rows]].filing_date).to_numpy()
    observed = pd.to_datetime(origins.iloc[used_rows].date).to_numpy()
    if not (disclosed < observed).all():
        raise ValueError("Same-day/future document entered training reduction")
    if not np.array_equal(documents.iloc[ids[used_rows]].issuer_id.to_numpy(), origins.iloc[used_rows].issuer_id.to_numpy()):
        raise ValueError("Cross-issuer document join")
    reducer = PCA(n_components=min(width, len(used) - 1, matrix.shape[1]), svd_solver="full", whiten=False)
    reducer.fit(matrix[used])
    transformed = np.zeros((len(matrix), width), dtype=np.float32)
    transformed[:, :reducer.n_components_] = reducer.transform(matrix).astype(np.float32)
    return reducer, transformed, used
