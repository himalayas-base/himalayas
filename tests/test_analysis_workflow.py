"""
tests/test_analysis_workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import inspect
import numpy as np
import pytest

from himalayas import Analysis
from himalayas.core import analysis as analysis_module


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
    Ensures finalize() attaches layout and q-values when requested.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
        toy_annotations (Annotations): Toy annotations fixture.
    """
    analysis = (
        Analysis(toy_matrix, toy_annotations)
        .cluster(linkage_threshold=1.0)
        .enrich()
        .finalize(add_qvalues=True, col_cluster=True)
    )
    results = analysis.results

    assert results is not None
    assert "qval" in results.df.columns
    layout = results.cluster_layout()
    assert layout.col_order is not None


@pytest.mark.api
def test_finalize_without_qvalues(toy_matrix, toy_annotations):
    """
    Ensures finalize(add_qvalues=False) leaves q-values absent.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
        toy_annotations (Annotations): Toy annotations fixture.
    """
    analysis = (
        Analysis(toy_matrix, toy_annotations)
        .cluster(linkage_threshold=1.0)
        .enrich()
        .finalize(add_qvalues=False, col_cluster=False)
    )
    results = analysis.results

    assert "qval" not in results.df.columns


@pytest.mark.api
def test_cluster_can_be_called_twice(toy_matrix, toy_annotations):
    """
    Ensures repeated clustering updates the analysis state without error.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
        toy_annotations (Annotations): Toy annotations fixture.
    """
    analysis = Analysis(toy_matrix, toy_annotations).cluster(linkage_threshold=1.0)
    first_clusters = analysis.clusters
    analysis.cluster(linkage_threshold=1.0)

    assert analysis.clusters is not None
    assert analysis.clusters is not first_clusters


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
        .cluster(linkage_method="average", linkage_metric="cosine", linkage_threshold=1.0)
        .enrich()
        .finalize(add_qvalues=False, col_cluster=True)
    )

    assert seen["kwargs"]["linkage_method"] == "average"
    assert seen["kwargs"]["linkage_metric"] == "cosine"
