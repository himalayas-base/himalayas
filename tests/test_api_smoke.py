"""
tests/test_api_smoke
~~~~~~~~~~~~~~~~~~~~
"""

import pytest

from himalayas import Analysis, Annotations, Matrix


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
