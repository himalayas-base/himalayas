"""
tests/test_analysis_workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import numpy as np
import pandas as pd
import pytest

from himalayas import Analysis, Annotations, Matrix
from himalayas.core import analysis as analysis_module
from himalayas.core import clustering as clustering_module


@pytest.mark.api
def test_analysis_requires_cluster_before_enrich(toy_matrix, toy_annotations):
    """
    Ensures enrich() requires clustering first.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
        toy_annotations (Annotations): Toy annotations fixture.

    Raises:
        RuntimeError: If enrich() is called before clustering.
    """
    analysis = Analysis(toy_matrix, toy_annotations)
    with pytest.raises(RuntimeError):
        analysis.enrich()


@pytest.mark.api
def test_analysis_requires_cluster_and_enrich_before_finalize(toy_matrix, toy_annotations):
    """
    Ensures finalize() requires both clustering and enrichment.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
        toy_annotations (Annotations): Toy annotations fixture.

    Raises:
        RuntimeError: If finalize() is called before clustering or enrichment.
    """
    analysis = Analysis(toy_matrix, toy_annotations)
    with pytest.raises(RuntimeError):
        analysis.finalize()


@pytest.mark.api
def test_finalize_attaches_layout_and_qvalues(toy_matrix, toy_annotations):
    """
    Ensures finalize() attaches layout, effect sizes, and q-values.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
        toy_annotations (Annotations): Toy annotations fixture.
    """
    analysis = (
        Analysis(toy_matrix, toy_annotations)
        .cluster(linkage_threshold=1.0)
        .enrich()
        .finalize(col_cluster=True)
    )
    results = analysis.results

    assert results is not None
    assert "fe" in results.df.columns
    assert "qval" in results.df.columns
    layout = results.cluster_layout()
    assert layout.col_order is not None


@pytest.mark.api
def test_finalize_attaches_qvalues_without_col_clustering(toy_matrix, toy_annotations):
    """
    Ensures finalize() adds effect sizes and q-values when column clustering is disabled.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
        toy_annotations (Annotations): Toy annotations fixture.
    """
    analysis = (
        Analysis(toy_matrix, toy_annotations)
        .cluster(linkage_threshold=1.0)
        .enrich()
        .finalize(col_cluster=False)
    )
    results = analysis.results

    assert "fe" in results.df.columns
    assert "qval" in results.df.columns


@pytest.mark.api
def test_recluster_invalidates_downstream_state(toy_matrix, toy_annotations):
    """
    Ensures repeated clustering invalidates stale downstream state.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
        toy_annotations (Annotations): Toy annotations fixture.
    """
    analysis = (
        Analysis(toy_matrix, toy_annotations)
        .cluster(linkage_threshold=1.0)
        .enrich()
        .finalize(col_cluster=False)
    )
    first_clusters = analysis.clusters
    analysis.cluster(linkage_threshold=1.0)

    assert analysis.clusters is not None
    assert analysis.clusters is not first_clusters
    assert analysis.results is None
    assert analysis.layout is None
    with pytest.raises(RuntimeError):
        analysis.finalize(col_cluster=False)


@pytest.mark.api
def test_finalize_col_cluster_uses_cluster_linkage_kwargs(monkeypatch, toy_matrix, toy_annotations):
    """
    Ensures finalize(col_cluster=True) uses linkage settings from cluster().

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing module call targets.
        toy_matrix (Matrix): Toy matrix fixture.
        toy_annotations (Annotations): Toy annotations fixture.
    """
    seen = {}

    def _capture_col_order(matrix, **kwargs):
        seen["kwargs"] = dict(kwargs)
        return np.arange(matrix.df.shape[1], dtype=int)

    monkeypatch.setattr(analysis_module, "compute_col_order", _capture_col_order)

    (
        Analysis(toy_matrix, toy_annotations)
        .cluster(
            linkage_method="average",
            linkage_metric="cosine",
            linkage_threshold=1.0,
            optimal_ordering=True,
        )
        .enrich()
        .finalize(col_cluster=True)
    )

    assert seen["kwargs"]["linkage_method"] == "average"
    assert seen["kwargs"]["linkage_metric"] == "cosine"
    assert seen["kwargs"]["optimal_ordering"] is True


@pytest.mark.api
def test_finalize_col_cluster_caches_col_order_for_same_linkage(
    monkeypatch, toy_matrix, toy_annotations
):
    """
    Ensures repeated finalize(col_cluster=True) reuses cached column order for the same linkage.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing module call targets.
        toy_matrix (Matrix): Toy matrix fixture.
        toy_annotations (Annotations): Toy annotations fixture.
    """
    seen = {"calls": 0}

    def _capture_col_order(matrix, **kwargs):
        seen["calls"] += 1
        return np.arange(matrix.df.shape[1], dtype=int)

    monkeypatch.setattr(analysis_module, "compute_col_order", _capture_col_order)

    analysis = (
        Analysis(toy_matrix, toy_annotations)
        .cluster(
            linkage_method="average",
            linkage_metric="cosine",
            linkage_threshold=1.0,
            optimal_ordering=False,
        )
        .enrich()
        .finalize(col_cluster=True)
    )
    analysis.finalize(col_cluster=True)

    assert seen["calls"] == 1


@pytest.mark.api
def test_finalize_col_cluster_recomputes_col_order_when_linkage_changes(
    monkeypatch, toy_matrix, toy_annotations
):
    """
    Ensures changing linkage settings causes a new column-order computation.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing module call targets.
        toy_matrix (Matrix): Toy matrix fixture.
        toy_annotations (Annotations): Toy annotations fixture.
    """
    seen = {"kwargs": []}

    def _capture_col_order(matrix, **kwargs):
        seen["kwargs"].append(dict(kwargs))
        return np.arange(matrix.df.shape[1], dtype=int)

    monkeypatch.setattr(analysis_module, "compute_col_order", _capture_col_order)

    analysis = (
        Analysis(toy_matrix, toy_annotations)
        .cluster(
            linkage_method="average",
            linkage_metric="cosine",
            linkage_threshold=1.0,
            optimal_ordering=False,
        )
        .enrich()
        .finalize(col_cluster=True)
    )
    (
        analysis.cluster(
            linkage_method="ward",
            linkage_metric="euclidean",
            linkage_threshold=1.0,
            optimal_ordering=False,
        )
        .enrich()
        .finalize(col_cluster=True)
    )

    assert len(seen["kwargs"]) == 2
    assert seen["kwargs"][0]["linkage_method"] == "average"
    assert seen["kwargs"][0]["linkage_metric"] == "cosine"
    assert seen["kwargs"][0]["optimal_ordering"] is False
    assert seen["kwargs"][1]["linkage_method"] == "ward"
    assert seen["kwargs"][1]["linkage_metric"] == "euclidean"
    assert seen["kwargs"][1]["optimal_ordering"] is False


@pytest.mark.api
def test_finalize_col_cluster_recomputes_col_order_when_optimal_ordering_changes(
    monkeypatch, toy_matrix, toy_annotations
):
    """
    Ensures changing optimal_ordering causes a new column-order computation.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing module call targets.
        toy_matrix (Matrix): Toy matrix fixture.
        toy_annotations (Annotations): Toy annotations fixture.
    """
    seen = {"kwargs": []}

    def _capture_col_order(matrix, **kwargs):
        seen["kwargs"].append(dict(kwargs))
        return np.arange(matrix.df.shape[1], dtype=int)

    monkeypatch.setattr(analysis_module, "compute_col_order", _capture_col_order)

    analysis = (
        Analysis(toy_matrix, toy_annotations)
        .cluster(
            linkage_method="average",
            linkage_metric="cosine",
            linkage_threshold=1.0,
            optimal_ordering=False,
        )
        .enrich()
        .finalize(col_cluster=True)
    )
    (
        analysis.cluster(
            linkage_method="average",
            linkage_metric="cosine",
            linkage_threshold=1.0,
            optimal_ordering=True,
        )
        .enrich()
        .finalize(col_cluster=True)
    )

    assert len(seen["kwargs"]) == 2
    assert seen["kwargs"][0]["optimal_ordering"] is False
    assert seen["kwargs"][1]["optimal_ordering"] is True


@pytest.mark.api
def test_cluster_reuses_cached_row_linkage_for_same_linkage_settings(
    monkeypatch, toy_matrix, toy_annotations
):
    """
    Ensures repeated cluster() calls reuse cached row linkage for the same linkage settings.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing module call targets.
        toy_matrix (Matrix): Toy matrix fixture.
        toy_annotations (Annotations): Toy annotations fixture.
    """
    seen = {"calls": 0}
    orig_compute_linkage = clustering_module.compute_linkage

    def _capture_compute_linkage(matrix, **kwargs):
        seen["calls"] += 1
        return orig_compute_linkage(matrix, **kwargs)

    monkeypatch.setattr(analysis_module, "compute_linkage", _capture_compute_linkage)

    analysis = Analysis(toy_matrix, toy_annotations).cluster(
        linkage_method="average",
        linkage_metric="cosine",
        linkage_threshold=0.5,
        optimal_ordering=False,
    )
    analysis.cluster(
        linkage_method="average",
        linkage_metric="cosine",
        linkage_threshold=1.0,
        optimal_ordering=False,
    )

    assert seen["calls"] == 1


@pytest.mark.api
def test_cluster_recomputes_row_linkage_when_linkage_settings_change(
    monkeypatch, toy_matrix, toy_annotations
):
    """
    Ensures changing linkage settings causes a new row-linkage computation.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing module call targets.
        toy_matrix (Matrix): Toy matrix fixture.
        toy_annotations (Annotations): Toy annotations fixture.
    """
    seen = {"kwargs": []}
    orig_compute_linkage = clustering_module.compute_linkage

    def _capture_compute_linkage(matrix, **kwargs):
        seen["kwargs"].append(dict(kwargs))
        return orig_compute_linkage(matrix, **kwargs)

    monkeypatch.setattr(analysis_module, "compute_linkage", _capture_compute_linkage)

    analysis = Analysis(toy_matrix, toy_annotations).cluster(
        linkage_method="average",
        linkage_metric="cosine",
        linkage_threshold=1.0,
        optimal_ordering=False,
    )
    analysis.cluster(
        linkage_method="ward",
        linkage_metric="euclidean",
        linkage_threshold=1.0,
        optimal_ordering=False,
    )

    assert len(seen["kwargs"]) == 2
    assert seen["kwargs"][0]["linkage_method"] == "average"
    assert seen["kwargs"][0]["linkage_metric"] == "cosine"
    assert seen["kwargs"][0]["optimal_ordering"] is False
    assert seen["kwargs"][1]["linkage_method"] == "ward"
    assert seen["kwargs"][1]["linkage_metric"] == "euclidean"
    assert seen["kwargs"][1]["optimal_ordering"] is False


@pytest.mark.api
def test_cluster_recomputes_row_linkage_when_optimal_ordering_changes(
    monkeypatch, toy_matrix, toy_annotations
):
    """
    Ensures changing optimal_ordering causes a new row-linkage computation.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing module call targets.
        toy_matrix (Matrix): Toy matrix fixture.
        toy_annotations (Annotations): Toy annotations fixture.
    """
    seen = {"kwargs": []}
    orig_compute_linkage = clustering_module.compute_linkage

    def _capture_compute_linkage(matrix, **kwargs):
        seen["kwargs"].append(dict(kwargs))
        return orig_compute_linkage(matrix, **kwargs)

    monkeypatch.setattr(analysis_module, "compute_linkage", _capture_compute_linkage)

    analysis = Analysis(toy_matrix, toy_annotations).cluster(
        linkage_method="average",
        linkage_metric="cosine",
        linkage_threshold=1.0,
        optimal_ordering=False,
    )
    analysis.cluster(
        linkage_method="average",
        linkage_metric="cosine",
        linkage_threshold=1.0,
        optimal_ordering=True,
    )

    assert len(seen["kwargs"]) == 2
    assert seen["kwargs"][0]["optimal_ordering"] is False
    assert seen["kwargs"][1]["optimal_ordering"] is True


@pytest.mark.api
def test_end_to_end_smoke(toy_df):
    """
    Ensures the basic analysis pipeline produces usable results.

    Args:
        toy_df (pd.DataFrame): Toy input DataFrame.
    """
    matrix = Matrix(toy_df)
    annotations = Annotations({"t1": ["a", "b"], "t2": ["c", "d"]}, matrix)
    analysis = (
        Analysis(matrix, annotations)
        .cluster(linkage_threshold=1.0)
        .enrich()
        .finalize(col_cluster=False)
    )
    results = analysis.results

    assert results is not None
    assert results.matrix is matrix
    assert results.clusters is not None
    assert "pval" in results.df.columns
    assert "fe" in results.df.columns
    assert "qval" in results.df.columns
    assert results.cluster_layout().cluster_spans


@pytest.mark.api
def test_analysis_cluster_propagates_merge_small_clusters():
    """
    Ensures Analysis.cluster() propagates merge_small_clusters to the underlying Clusters
    object, preserving small dendrogram-cut clusters and excluding them from enrichment
    when set to False.
    """
    df = pd.DataFrame(
        [[0.0], [0.1], [5.0], [5.1], [10.0]],
        index=["a", "b", "c", "d", "e"],
        columns=["x"],
    )
    matrix = Matrix(df)
    annotations = Annotations({"t1": ["a", "b"], "t2": ["c", "d"], "t3": ["e"]}, matrix)
    analysis = Analysis(matrix, annotations).cluster(
        linkage_threshold=0.5,
        min_cluster_size=2,
        merge_small_clusters=False,
    )

    assert analysis.clusters.merge_small_clusters is False
    assert analysis.clusters.min_cluster_size == 2
    singleton_cid = analysis.clusters.label_to_cluster["e"]
    assert analysis.clusters.cluster_sizes[singleton_cid] == 1

    analysis = analysis.enrich()
    assert singleton_cid not in set(analysis.results.df["cluster"].tolist())


@pytest.mark.api
def test_analysis_cluster_auto_threshold_is_finite_numeric(toy_matrix, toy_annotations):
    """
    Ensures Analysis.cluster(linkage_threshold="auto") resolves to a finite numeric
    Clusters.threshold.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
        toy_annotations (Annotations): Toy annotations fixture.
    """
    analysis = Analysis(toy_matrix, toy_annotations).cluster(linkage_threshold="auto")

    assert isinstance(analysis.clusters.threshold, float)
    assert np.isfinite(analysis.clusters.threshold)


@pytest.mark.api
def test_analysis_cluster_auto_threshold_uses_requested_method_and_metric(toy_matrix, toy_annotations):
    """
    Ensures linkage_threshold="auto" actually selects using the requested linkage method
    and metric, not just stores them: the resolved threshold must match an independent
    silhouette-times-diversity argmax computed over linkage built with those same settings.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
        toy_annotations (Annotations): Toy annotations fixture.
    """
    from scipy.cluster.hierarchy import fcluster
    from scipy.spatial.distance import pdist, squareform
    from sklearn.metrics import silhouette_score

    linkage_matrix = clustering_module.compute_linkage(
        toy_matrix, linkage_method="average", linkage_metric="cityblock"
    )
    distance_matrix = squareform(pdist(toy_matrix.values, metric="cityblock"))
    n = toy_matrix.values.shape[0]
    expected_threshold, expected_score = None, -np.inf
    for threshold in np.unique(linkage_matrix[:, 2]):
        labels = fcluster(linkage_matrix, threshold, criterion="distance")
        n_clusters = len(np.unique(labels))
        if n_clusters < 2 or n_clusters >= n:
            continue
        silhouette = silhouette_score(distance_matrix, labels, metric="precomputed")
        score = silhouette
        if silhouette > 0:
            _, counts = np.unique(labels, return_counts=True)
            proportions = counts / counts.sum()
            diversity = 1.0 - np.sum(proportions**2)
            score *= diversity
        if score > expected_score:
            expected_score = score
            expected_threshold = float(threshold)

    analysis = Analysis(toy_matrix, toy_annotations).cluster(
        linkage_method="average",
        linkage_metric="cityblock",
        linkage_threshold="auto",
    )

    assert analysis.clusters.threshold == pytest.approx(expected_threshold)


@pytest.mark.api
def test_analysis_cluster_auto_threshold_passes_min_cluster_size_to_rescue():
    """
    Ensures Analysis.cluster() forwards min_cluster_size and merge_small_clusters into auto
    threshold resolution, engaging the reportability rescue when the committed winner is
    under-reportable. The fixture's committed silhouette-diversity winner has only 1
    reportable cluster under min_cluster_size=2.
    """
    df = pd.DataFrame(
        [[-8.5], [23.0], [21.0], [43.0], [-38.0], [23.0]],
        index=["a", "b", "c", "d", "e", "f"],
        columns=["x"],
    )
    matrix = Matrix(df)
    annotations = Annotations({"t1": ["a", "b"], "t2": ["c", "d"]}, matrix)

    committed = Analysis(matrix, annotations).cluster(linkage_threshold="auto")
    rescued = Analysis(matrix, annotations).cluster(
        linkage_threshold="auto", min_cluster_size=2, merge_small_clusters=False
    )

    assert rescued.clusters.threshold != committed.clusters.threshold


@pytest.mark.api
@pytest.mark.parametrize("bad_threshold", ["bad", True, False])
def test_analysis_cluster_invalid_threshold_raises(toy_matrix, toy_annotations, bad_threshold):
    """
    Ensures Analysis.cluster() rejects invalid string and boolean linkage_threshold values.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
        toy_annotations (Annotations): Toy annotations fixture.
        bad_threshold: Invalid linkage_threshold value.
    """
    with pytest.raises(ValueError):
        Analysis(toy_matrix, toy_annotations).cluster(linkage_threshold=bad_threshold)
