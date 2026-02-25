"""
tests/test_results
~~~~~~~~~~~~~~~~~~
"""

import numpy as np
import pandas as pd
import pytest

from himalayas import Matrix, Results
from himalayas.core.clustering import cluster
from himalayas.core.results import _resolve_rank_spec


@pytest.mark.api
def test_results_with_qvalues_adds_column():
    """
    Ensures q-values are added to results.
    """
    # Compute q-values on a small p-value table
    df = pd.DataFrame({"pval": [0.01, 0.2, 0.05]})
    res = Results(df)
    out = res.with_qvalues()

    assert "qval" in out.df.columns
    assert np.all(np.isfinite(out.df["qval"]))


@pytest.mark.api
def test_results_with_effect_sizes_adds_column():
    """
    Ensures fold enrichment is added with expected values.
    """
    df = pd.DataFrame(
        {
            "k": [2, 1],
            "K": [4, 2],
            "n": [5, 5],
            "N": [20, 20],
        }
    )
    res = Results(df)
    out = res.with_effect_sizes()

    assert "fe" in out.df.columns
    assert out.df["fe"].to_numpy(dtype=float) == pytest.approx([2.0, 2.0])


@pytest.mark.api
def test_results_subset_requires_matrix_and_clusters():
    """
    Ensures subsetting requires matrix and clusters.

    Raises:
        ValueError: If subsetting is attempted without attachments.
    """
    # Subsetting without attachments should fail
    df = pd.DataFrame({"pval": [0.1]})
    res = Results(df)
    with pytest.raises(ValueError):
        res.subset(cluster=1)


@pytest.mark.api
def test_results_subset_single_cluster_returns_cluster_rows():
    """
    Ensures subsetting a single cluster returns only that cluster in matrix row order.
    """
    # Build a tiny matrix and subset the first cluster
    df = pd.DataFrame(
        [[0.0, 0.1], [0.0, 0.2], [5.0, 5.1], [5.0, 5.2]],
        index=["a", "b", "c", "d"],
        columns=["f1", "f2"],
    )
    matrix = Matrix(df)
    clusters = cluster(matrix, linkage_threshold=1.0)
    res = Results(pd.DataFrame(), matrix=matrix, clusters=clusters)
    cid = int(clusters.unique_clusters[0])
    expected_labels = [lab for lab in matrix.df.index if lab in clusters.cluster_to_labels[cid]]
    sub = res.subset(cluster=cid)

    assert sub.matrix is not None
    assert list(sub.matrix.df.index) == expected_labels
    assert sub.clusters is None


@pytest.mark.api
def test_results_subset_clusters_returns_union_in_matrix_order(toy_results):
    """
    Ensures multi-cluster subsetting returns the label union in original matrix row order.

    Args:
        toy_results (Results): Results fixture with clusters and matrix attached.
    """
    # Select two real clusters from the fixture to validate multi-cluster subsetting.
    cluster_ids = [int(c) for c in toy_results.clusters.unique_clusters]
    assert len(cluster_ids) >= 2
    # Build expected labels as the selected-cluster union in original matrix row order.
    selected = cluster_ids[:2]
    cluster_to_labels = toy_results.clusters.cluster_to_labels
    expected_set = set().union(*(cluster_to_labels[cid] for cid in selected))
    expected_labels = [lab for lab in toy_results.matrix.df.index if lab in expected_set]
    sub = toy_results.subset_clusters(selected)

    assert sub.matrix is not None
    assert list(sub.matrix.df.index) == expected_labels
    assert sub.clusters is None


@pytest.mark.api
def test_results_with_qvalues_rejects_invalid_pvals():
    """
    Ensures invalid p-values raise a ValueError.

    Raises:
        ValueError: If p-values contain invalid values.
    """
    df = pd.DataFrame({"pval": [0.2, -0.1, 1.2]})
    res = Results(df)
    with pytest.raises(ValueError):
        res.with_qvalues()


@pytest.mark.api
def test_results_with_qvalues_preserves_nan():
    """
    Ensures NaN p-values remain NaN after q-value computation.
    """
    df = pd.DataFrame({"pval": [0.01, np.nan, 0.2]})
    res = Results(df)
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
def test_results_filter_preserves_parent_and_matrix(toy_results):
    """
    Ensures filter() keeps parent linkage and matrix attachment.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    filtered = toy_results.filter("pval >= 0")

    assert filtered.parent is toy_results
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
            "fe": [1.2, 2.5, 0.8],
            "n": [7, 7, 3],
        }
    )
    res = Results(df)
    out = res.cluster_labels()

    assert list(out.columns) == ["cluster", "label", "pval", "qval", "score", "n", "term", "fe"]
    c1 = out.loc[out["cluster"] == 1].iloc[0]
    c2 = out.loc[out["cluster"] == 2].iloc[0]

    # Cluster 1: term_name is missing for best row; fallback to stable term id.
    assert c1["label"] == "t_b"
    assert c1["term"] == "t_b"
    assert c1["pval"] == pytest.approx(0.01)
    assert c1["qval"] is None
    assert c1["score"] == pytest.approx(0.01)
    assert c1["fe"] == pytest.approx(2.5)
    assert c1["n"] == 7
    assert c2["label"] == "Term C"
    assert c2["term"] == "t_c"
    assert c2["qval"] is None
    assert c2["score"] == pytest.approx(0.4)
    assert c2["fe"] == pytest.approx(0.8)


@pytest.mark.api
def test_results_cluster_labels_compressed_requires_rank_column():
    """
    Ensures compressed labels require the selected ranking column.

    Raises:
        KeyError: If the selected ranking column is missing.
    """
    df = pd.DataFrame(
        {
            "cluster": [1, 1],
            "term": ["Alpha Beta", "Alpha Gamma"],
        }
    )
    res = Results(df)
    with pytest.raises(KeyError):
        res.cluster_labels(label_mode="compressed", max_words=1)


@pytest.mark.api
def test_results_cluster_labels_compressed_rank_by_q_requires_qval():
    """
    Ensures compressed labels with rank_by='q' require a q-value column.

    Raises:
        KeyError: If q-values are missing when rank_by='q'.
    """
    df = pd.DataFrame(
        {
            "cluster": [1, 1],
            "term": ["Alpha Beta", "Alpha Gamma"],
            "pval": [0.2, 0.05],
        }
    )
    res = Results(df)
    with pytest.raises(KeyError):
        res.cluster_labels(label_mode="compressed", rank_by="q", max_words=1)


@pytest.mark.api
def test_results_cluster_labels_rank_by_q_changes_top_term_and_score():
    """
    Ensures rank_by='q' uses q-values for representative-term ranking and score output.
    """
    # Use conflicting p/q ranks so rank_by changes the selected representative term.
    df = pd.DataFrame(
        {
            "cluster": [1, 1],
            "term": ["t_a", "t_b"],
            "term_name": ["Term A", "Term B"],
            "pval": [0.01, 0.02],
            "qval": [0.20, 0.05],
            "fe": [1.1, 3.2],
            "n": [5, 5],
        }
    )
    res = Results(df)
    out_p = res.cluster_labels(rank_by="p")
    out_q = res.cluster_labels(rank_by="q")
    row_p = out_p.iloc[0]
    row_q = out_q.iloc[0]

    assert row_p["term"] == "t_a"
    assert row_p["score"] == pytest.approx(0.01)
    assert row_q["term"] == "t_b"
    assert row_q["score"] == pytest.approx(0.05)
    # Display stats remain semantically named regardless of ranking signal.
    assert row_q["pval"] == pytest.approx(0.02)
    assert row_q["qval"] == pytest.approx(0.05)
    assert row_q["fe"] == pytest.approx(3.2)


@pytest.mark.api
def test_results_cluster_labels_rank_by_q_ties_break_by_p_then_term():
    """
    Ensures q-rank ties are broken deterministically by p-value, then term id.
    """
    df = pd.DataFrame(
        {
            "cluster": [1, 1, 1],
            "term": ["t_b", "t_a", "t_c"],
            "pval": [0.01, 0.01, 0.02],
            "qval": [0.05, 0.05, 0.05],
        }
    )
    res = Results(df)
    out = res.cluster_labels(rank_by="q")
    row = out.iloc[0]

    assert row["term"] == "t_a"
    assert row["pval"] == pytest.approx(0.01)
    assert row["qval"] == pytest.approx(0.05)
    assert row["score"] == pytest.approx(0.05)


@pytest.mark.api
def test_results_cluster_labels_invalid_label_mode_raises():
    """
    Ensures unsupported label modes raise a ValueError.

    Raises:
        ValueError: If label_mode is not supported.
    """
    df = pd.DataFrame({"cluster": [1], "term": ["t1"], "pval": [0.1]})
    res = Results(df)
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
    res = Results(df)
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
    res = Results(df)
    with pytest.raises(KeyError):
        res.cluster_labels(label_mode="top_term")


@pytest.mark.unit
def test_resolve_rank_spec_supports_rank_by_values():
    """
    Ensures rank specification resolves cleanly for supported semantic selectors.

    Raises:
        ValueError: If rank_by is outside the supported set.
    """
    assert _resolve_rank_spec(rank_by="p") == "pval"
    assert _resolve_rank_spec(rank_by="q") == "qval"

    with pytest.raises(ValueError):
        _resolve_rank_spec(rank_by="bad")


@pytest.mark.api
def test_results_cluster_labels_rejects_unknown_kwargs():
    """
    Ensures unknown keyword arguments are rejected.

    Raises:
        TypeError: If unknown keyword arguments are provided.
    """
    df = pd.DataFrame({"cluster": [1], "term": ["t1"], "pval": [0.1]})
    res = Results(df)
    with pytest.raises(TypeError, match="unknown_kwarg"):
        res.cluster_labels(unknown_kwarg=True)
