"""
tests/test_analysis_workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import numpy as np
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
    Ensures finalize() attaches layout and q-values.

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
    assert "qval" in results.df.columns
    layout = results.cluster_layout()
    assert layout.col_order is not None


@pytest.mark.api
def test_finalize_attaches_qvalues_without_col_clustering(toy_matrix, toy_annotations):
    """
    Ensures finalize() adds q-values even when column clustering is disabled.

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
def test_finalize_col_cluster_uses_cluster_linkage_kwargs(
    monkeypatch, toy_matrix, toy_annotations
):
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
    assert "qval" in results.df.columns
    assert results.cluster_layout().cluster_spans
