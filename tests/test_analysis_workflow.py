import pytest

from himalayas import Analysis


@pytest.mark.api
def test_analysis_requires_cluster_before_enrich(toy_matrix, toy_annotations):
    analysis = Analysis(toy_matrix, toy_annotations)
    with pytest.raises(RuntimeError):
        analysis.enrich()


@pytest.mark.api
def test_analysis_requires_cluster_and_enrich_before_finalize(toy_matrix, toy_annotations):
    analysis = Analysis(toy_matrix, toy_annotations)
    with pytest.raises(RuntimeError):
        analysis.finalize()


@pytest.mark.api
def test_finalize_attaches_layout_and_qvalues(toy_matrix, toy_annotations):
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
