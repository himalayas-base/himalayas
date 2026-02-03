import pytest

from himalayas import Analysis, Annotations, Matrix, Results, cluster


@pytest.mark.api
def test_public_imports():
    assert Matrix is not None
    assert Annotations is not None
    assert Analysis is not None
    assert Results is not None
    assert cluster is not None


@pytest.mark.api
def test_end_to_end_smoke(toy_df):
    matrix = Matrix(toy_df)
    annotations = Annotations({"t1": ["a", "b"], "t2": ["c", "d"]}, matrix)

    analysis = (
        Analysis(matrix, annotations)
        .cluster(linkage_threshold=1.0)
        .enrich()
        .finalize(add_qvalues=False, col_cluster=False)
    )
    results = analysis.results

    assert results is not None
    assert results.matrix is matrix
    assert results.clusters is not None
    assert "pval" in results.df.columns
    assert results.cluster_layout().cluster_spans
