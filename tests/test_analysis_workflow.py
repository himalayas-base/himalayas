"""
tests/test_analysis_workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pytest

from himalayas import Analysis


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
    """
    analysis = Analysis(toy_matrix, toy_annotations).cluster(linkage_threshold=1.0)
    first_clusters = analysis.clusters
    analysis.cluster(linkage_threshold=1.0)

    assert analysis.clusters is not None
    assert analysis.clusters is not first_clusters
