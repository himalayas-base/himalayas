"""
tests/test_results
~~~~~~~~~~~~~~~~~~
"""

import numpy as np
import pandas as pd
import pytest

from himalayas import Matrix, Results, cluster


@pytest.mark.api
def test_results_with_qvalues_adds_column():
    """
    Ensures q-values are added to results.
    """
    # Compute q-values on a small p-value table
    df = pd.DataFrame({"pval": [0.01, 0.2, 0.05]})
    res = Results(df, method="test")
    out = res.with_qvalues()

    assert "qval" in out.df.columns
    assert np.all(np.isfinite(out.df["qval"]))


@pytest.mark.api
def test_results_subset_requires_matrix_and_clusters():
    """
    Ensures subsetting requires matrix and clusters.

    Raises:
        ValueError: If subsetting is attempted without attachments.
    """
    # Subsetting without attachments should fail
    df = pd.DataFrame({"pval": [0.1]})
    res = Results(df, method="test")
    with pytest.raises(ValueError):
        res.subset(cluster=1)


@pytest.mark.api
def test_results_subset_returns_new_matrix():
    """
    Ensures subsetting returns a new matrix and clears clusters.
    """
    # Build a tiny matrix and subset the first cluster
    df = pd.DataFrame(
        [[0.0], [1.0], [2.0]],
        index=["a", "b", "c"],
        columns=["x"],
    )
    matrix = Matrix(df)
    clusters = cluster(matrix, linkage_threshold=100.0)
    res = Results(pd.DataFrame(), method="test", matrix=matrix, clusters=clusters)
    sub = res.subset(cluster=int(clusters.unique_clusters[0]))

    assert sub.matrix is not None
    assert sub.clusters is None


@pytest.mark.api
def test_results_with_qvalues_rejects_invalid_pvals():
    """
    Ensures invalid p-values raise a ValueError.

    Raises:
        ValueError: If p-values contain invalid values.
    """
    df = pd.DataFrame({"pval": [0.2, -0.1, 1.2]})
    res = Results(df, method="test")
    with pytest.raises(ValueError):
        res.with_qvalues()


@pytest.mark.api
def test_results_with_qvalues_preserves_nan():
    """
    Ensures NaN p-values remain NaN after q-value computation.
    """
    df = pd.DataFrame({"pval": [0.01, np.nan, 0.2]})
    res = Results(df, method="test")
    out = res.with_qvalues()

    assert np.isnan(out.df.loc[1, "qval"])


@pytest.mark.api
def test_results_subset_invalid_cluster_raises(toy_results):
    """
    Ensures subsetting with an unknown cluster id raises.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        KeyError: If the requested cluster id does not exist.
    """
    with pytest.raises(KeyError):
        toy_results.subset(cluster=999)


@pytest.mark.api
def test_results_filter_preserves_parent_and_method(toy_results):
    """
    Ensures filter() keeps method and parent linkage.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    filtered = toy_results.filter("pval >= 0")

    assert filtered.parent is toy_results
    assert filtered.method == toy_results.method
    assert filtered.matrix is toy_results.matrix


@pytest.mark.api
def test_results_cluster_labels_top_term_defaults():
    """
    Ensures cluster_labels() returns top-term labels and canonical columns.
    """
    df = pd.DataFrame(
        {
            "cluster": [1, 1, 2],
            "term": ["t_a", "t_b", "t_c"],
            "term_name": ["Term A", None, "Term C"],
            "pval": [0.2, 0.01, 0.4],
            "n": [7, 7, 3],
        }
    )
    res = Results(df, method="test")
    out = res.cluster_labels()

    assert list(out.columns) == ["cluster", "label", "pval", "n", "term"]
    c1 = out.loc[out["cluster"] == 1].iloc[0]
    c2 = out.loc[out["cluster"] == 2].iloc[0]

    # Cluster 1: term_name is missing for best row; fallback to stable term id.
    assert c1["label"] == "t_b"
    assert c1["term"] == "t_b"
    assert c1["pval"] == pytest.approx(0.01)
    assert c1["n"] == 7
    assert c2["label"] == "Term C"
    assert c2["term"] == "t_c"


@pytest.mark.api
def test_results_cluster_labels_compressed_works_without_pvals():
    """
    Ensures compressed labels work without a p-value column.
    """
    df = pd.DataFrame(
        {
            "cluster": [1, 1],
            "term": ["Alpha Beta", "Alpha Gamma"],
        }
    )
    res = Results(df, method="test")
    out = res.cluster_labels(label_mode="compressed", max_words=1)

    assert out.loc[0, "cluster"] == 1
    assert out.loc[0, "label"] == "alpha"
    assert out.loc[0, "pval"] is None


@pytest.mark.api
def test_results_cluster_labels_invalid_label_mode_raises():
    """
    Ensures unsupported label modes raise a ValueError.

    Raises:
        ValueError: If label_mode is not supported.
    """
    df = pd.DataFrame({"cluster": [1], "term": ["t1"], "pval": [0.1]})
    res = Results(df, method="test")
    with pytest.raises(ValueError):
        res.cluster_labels(label_mode="bad")


@pytest.mark.api
def test_results_cluster_labels_missing_required_columns_raises():
    """
    Ensures missing required columns raise a KeyError.

    Raises:
        KeyError: If required input columns are missing.
    """
    df = pd.DataFrame({"cluster": [1], "pval": [0.1]})
    res = Results(df, method="test")
    with pytest.raises(KeyError):
        res.cluster_labels()


@pytest.mark.api
def test_results_cluster_labels_top_term_requires_pval():
    """
    Ensures top_term mode requires a p-value column.

    Raises:
        KeyError: If top_term mode is requested without a p-value column.
    """
    df = pd.DataFrame({"cluster": [1], "term": ["t1"]})
    res = Results(df, method="test")
    with pytest.raises(KeyError):
        res.cluster_labels(label_mode="top_term")
