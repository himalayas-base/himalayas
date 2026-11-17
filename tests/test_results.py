"""
tests/test_results
~~~~~~~~~~~~~~~~~~
"""

import numpy as np
import pandas as pd
import pytest

from himalayas import Annotations, Matrix, Results
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
    # Subsetting without attachments should fail.
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


@pytest.mark.api
def test_results_with_qvalues_per_cluster_less_conservative_than_global():
    """
    Ensures per_cluster FDR yields q-values <= global FDR, and that the two methods
    differ when correction pools differ across clusters.
    """
    df = pd.DataFrame(
        {
            "cluster": [1, 1, 1, 2, 2, 2],
            "pval": [0.01, 0.02, 0.05, 0.03, 0.04, 0.06],
        }
    )
    res = Results(df)
    global_q = res.with_qvalues(method="global").df["qval"].to_numpy()
    per_cluster_q = res.with_qvalues(method="per_cluster").df["qval"].to_numpy()

    assert np.all(per_cluster_q <= global_q + 1e-12)
    assert not np.allclose(per_cluster_q, global_q)


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


def _redundancy_fixture():
    """
    Builds a Results table and Annotations with overlapping terms in two clusters.

    Returns:
        tuple[Results, Annotations]: Results with a matrix attached, and matching annotations.
    """
    labels = list("abcdefghij")
    matrix = Matrix(pd.DataFrame(np.eye(10), index=labels, columns=labels))
    terms = {
        "big": list("abcd"),
        "big_copy": list("abcd"),
        "big_part": list("abc"),
        "other": list("ef"),
        "far": list("ghij"),
    }
    annotations = Annotations(terms, matrix)
    rows = [
        (1, "big", 1e-9),
        (1, "big_copy", 1e-8),
        (1, "big_part", 1e-7),
        (1, "other", 1e-3),
        (2, "big_copy", 1e-6),
        (2, "far", 1e-5),
        (2, "big", 1e-4),
    ]
    df = pd.DataFrame(rows, columns=["cluster", "term", "pval"])
    return Results(df, matrix=matrix), annotations


def _brute_force_distinct(df, terms, threshold):
    """
    Computes the distinct (cluster, term) pairs by comparing every pair of rows directly.

    Args:
        df (pd.DataFrame): Results rows.
        terms (dict[str, set]): Term to member labels.
        threshold (float): Jaccard index from which a term repeats a more significant one.

    Returns:
        set[tuple[int, str]]: Kept (cluster, term) pairs.
    """
    kept = set()
    for cid, term, _p in sorted(df.itertuples(index=False), key=lambda r: (r.pval, r.term)):
        repeats = False
        for kept_cid, kept_term in kept:
            if kept_cid != cid:
                continue
            shared = len(terms[term] & terms[kept_term])
            union = len(terms[term] | terms[kept_term])
            if shared > 0 and shared / union >= threshold:
                repeats = True
        if not repeats:
            kept.add((cid, term))
    return kept


@pytest.mark.api
@pytest.mark.parametrize("threshold", [0.5, 0.75, 1.0])
def test_results_drop_redundant_matches_brute_force(threshold):
    """
    Ensures drop_redundant() keeps the same rows as a direct pairwise comparison, per cluster.

    Args:
        threshold (float): Jaccard threshold under test.
    """
    results, annotations = _redundancy_fixture()
    out = results.drop_redundant(annotations, jaccard=threshold)

    expected = _brute_force_distinct(results.df, annotations.term_to_labels, threshold)
    assert set(zip(out.df["cluster"], out.df["term"])) == expected
    assert out.parent is results
    assert out.matrix is results.matrix
    assert list(out.df.index) == sorted(out.df.index)


@pytest.mark.api
def test_results_drop_redundant_keeps_most_significant_of_a_pair():
    """
    Ensures the more significant term of two near-duplicates survives, and that a term that
    repeats one in another cluster is kept there.
    """
    results, annotations = _redundancy_fixture()
    out = results.drop_redundant(annotations, jaccard=0.75)

    assert set(zip(out.df["cluster"], out.df["term"])) == {
        (1, "big"),
        (1, "other"),
        (2, "big_copy"),
        (2, "far"),
    }


@pytest.mark.api
@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"jaccard": 1.5}, ValueError),
        ({"jaccard": -0.1}, ValueError),
        ({"rank_by": "q"}, KeyError),
        ({"rank_by": "x"}, ValueError),
    ],
)
def test_results_drop_redundant_rejects_invalid_arguments(kwargs, error):
    """
    Ensures drop_redundant() rejects an out-of-range jaccard, an unknown rank, and a missing qval.

    Args:
        kwargs (dict): Keyword arguments for drop_redundant().
        error (type): Expected exception type.
    """
    results, annotations = _redundancy_fixture()
    with pytest.raises(error):
        results.drop_redundant(annotations, **kwargs)


@pytest.mark.api
def test_results_drop_redundant_requires_matrix_and_known_terms():
    """
    Ensures drop_redundant() raises without a matrix, and for a term missing from annotations.

    Raises:
        ValueError: If the matrix is not attached or a term is unknown.
    """
    results, annotations = _redundancy_fixture()
    with pytest.raises(ValueError):
        Results(results.df).drop_redundant(annotations)
    unknown = Results(
        pd.DataFrame({"cluster": [1], "term": ["missing"], "pval": [0.1]}),
        matrix=results.matrix,
    )
    with pytest.raises(ValueError):
        unknown.drop_redundant(annotations)


@pytest.mark.api
def test_results_drop_redundant_handles_terms_dropped_from_annotations():
    """
    Ensures drop_redundant() compares terms by their labels in the matrix even when a term was
    dropped from the annotations' retained terms (as in zoomed analyses with a background).
    """
    results, _ = _redundancy_fixture()
    matrix = results.matrix
    # "outside" has one label in the matrix, so Annotations drops it, but keeps its source labels.
    terms = {"big": list("abcd"), "other": list("ef"), "outside": ["a", "x1", "x2"]}
    with pytest.warns(RuntimeWarning):
        annotations = Annotations(terms, matrix, min_term_size=2)
    assert "outside" not in annotations.term_to_labels

    df = pd.DataFrame(
        {"cluster": [1, 1, 1], "term": ["outside", "big", "other"], "pval": [1e-9, 1e-5, 1e-3]}
    )
    zoom = Results(df, matrix=matrix)

    # Jaccard(outside, big) = 1/4: kept at 0.5, dropped at 0.2.
    assert set(zoom.drop_redundant(annotations, jaccard=0.5).df["term"]) == {
        "outside",
        "big",
        "other",
    }
    assert set(zoom.drop_redundant(annotations, jaccard=0.2).df["term"]) == {"outside", "other"}
