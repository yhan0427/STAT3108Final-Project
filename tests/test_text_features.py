import numpy as np
import pandas as pd
import pytest

from volrisk.text_features import align_documents, fit_training_document_pca


def test_filing_date_strictness_staleness_and_failed_new_document_are_explicit():
    docs = pd.DataFrame({"filing_date": ["2021-01-04"], "report_period": ["2020-12-31"],
                         "accession": ["a"], "document_id": [0], "content_tokens": [100]})
    dates = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-02-02", "2022-08-01"])
    joined = align_documents(dates, docs, ["2021-01-04", "2021-02-01"])
    assert joined.text_document_id.tolist() == [-1, 0, 0, -1]
    assert joined.newer_original_unavailable.tolist() == [0., 0., 1., 1.]
    assert joined.text_available.tolist() == [0., 1., 1., 0.]
    assert joined.source_document_id.iloc[-1] == 0
    guarded = align_documents(dates, docs, ["2021-01-04"], entity_start="2021-02-01")
    assert guarded.text_document_id.eq(-1).all()


def test_reduction_uses_unique_training_documents_and_ignores_future_held_vectors():
    embeddings = np.array([[1., 2., 3.], [3., 4., 2.], [5., 1., 6.], [100., 100., 100.]])
    documents = pd.DataFrame({"role": ["training_company"] * 3 + ["held_company"],
                              "issuer_id": ["one"] * 3 + ["two"],
                              "filing_date": ["2019-01-01", "2020-01-01", "2021-01-01", "2024-01-01"]})
    origins = pd.DataFrame({"role": ["training_company"] * 5 + ["held_company"],
                            "issuer_id": ["one"] * 5 + ["two"],
                            "date": pd.to_datetime(["2019-02-01", "2019-03-01", "2019-04-01", "2020-02-01", "2021-02-01", "2024-02-01"])})
    ids = np.array([0, 0, 0, 1, 2, 3]); mask = np.array([True] * 5 + [False])
    reducer, reduced, used = fit_training_document_pca(embeddings, documents, origins, ids, mask, width=4)
    np.testing.assert_allclose(reducer.mean_, embeddings[:3].mean(axis=0))
    assert used.tolist() == [0, 1, 2] and reduced.shape == (4, 4)
    changed = embeddings.copy(); changed[3] *= 1000
    other, _, _ = fit_training_document_pca(changed, documents, origins, ids, mask, width=4)
    np.testing.assert_array_equal(reducer.components_, other.components_)
    with pytest.raises(ValueError, match="training-role"):
        fit_training_document_pca(embeddings, documents, origins, ids, np.ones(6, dtype=bool))
    invalid_dates = origins.copy(); invalid_dates.loc[0, "date"] = pd.Timestamp("2019-01-01")
    with pytest.raises(ValueError, match="Same-day/future"):
        fit_training_document_pca(embeddings, documents, invalid_dates, ids, mask)
    before_spell = documents.copy(); before_spell["entity_start"] = "2019-06-01"
    with pytest.raises(ValueError, match="Pre-spell"):
        fit_training_document_pca(embeddings, before_spell, origins, ids, mask)
