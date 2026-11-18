import numpy as np
import pytest

from volrisk.text_embeddings import chunk_token_ids, weighted_document_mean


def test_no_truncation_or_duplicate_tokens_and_partial_chunk_weight():
    ids = list(range(1021))
    chunks = chunk_token_ids(ids)
    assert [len(c) for c in chunks] == [510, 510, 1]
    assert sum(chunks, []) == ids
    result = weighted_document_mean([[1., 2.], [3., 4.], [10., 20.]], [510, 510, 1])
    expected = np.array([2050, 3080]) / 1021
    np.testing.assert_allclose(result, expected, rtol=1e-7)
    with pytest.raises(ValueError):
        chunk_token_ids([])
    with pytest.raises(ValueError):
        weighted_document_mean([[1., 2.]], [0])
