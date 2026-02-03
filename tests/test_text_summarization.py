"""
tests/test_text_summarization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pandas as pd
import pytest

from himalayas.text import summarize_clusters, summarize_terms


@pytest.mark.api
def test_summarize_terms_weights_length_mismatch():
    """
    Ensures weights length mismatch raises a ValueError.

    Raises:
        ValueError: If weights length does not match terms length.
    """
    with pytest.raises(ValueError):
        summarize_terms(["a", "b"], weights=[1.0])


@pytest.mark.api
def test_summarize_terms_empty_returns_na():
    """
    Ensures empty or whitespace-only inputs return N/A.
    """
    assert summarize_terms([]) == "N/A"
    assert summarize_terms(["", "   "]) == "N/A"


@pytest.mark.api
def test_summarize_clusters_invalid_label_mode():
    """
    Ensures invalid label_mode raises a ValueError.

    Raises:
        ValueError: If label_mode is not supported.
    """
    df = pd.DataFrame({"term": ["t1"], "cluster": [1], "pval": [0.1]})
    with pytest.raises(ValueError):
        summarize_clusters(df, label_mode="bad")


@pytest.mark.api
def test_summarize_clusters_missing_required_columns():
    """
    Ensures missing required columns raise a KeyError.

    Raises:
        KeyError: If required columns are missing.
    """
    df = pd.DataFrame({"cluster": [1], "pval": [0.1]})
    with pytest.raises(KeyError):
        summarize_clusters(df)


@pytest.mark.api
def test_summarize_clusters_top_term_requires_pval():
    """
    Ensures top_term mode requires a p-value column.

    Raises:
        KeyError: If the p-value column is missing.
    """
    df = pd.DataFrame({"term": ["t1"], "cluster": [1]})
    with pytest.raises(KeyError):
        summarize_clusters(df, label_mode="top_term")
