"""
tests/test_enrichment
~~~~~~~~~~~~~~~~~~~~~
"""

import pandas as pd
import pytest

from himalayas import Annotations, Matrix, cluster
from himalayas.core import run_cluster_hypergeom


@pytest.mark.api
def test_run_cluster_hypergeom_basic():
    """
    Ensures hypergeometric enrichment returns expected counts.
    """
    # Create a minimal matrix/annotation setup
    df = pd.DataFrame(
        [[0.0], [1.0], [2.0]],
        index=["a", "b", "c"],
        columns=["x"],
    )
    matrix = Matrix(df)
    clusters = cluster(matrix, linkage_threshold=100.0)
    annotations = Annotations({"t1": ["a", "b"], "t2": ["c"]}, matrix)
    results = run_cluster_hypergeom(matrix, clusters, annotations)

    assert set(results.df["term"].tolist()) == {"t1", "t2"}
    row_t1 = results.df[results.df["term"] == "t1"].iloc[0]
    assert row_t1["K"] == 2
    assert row_t1["k"] == 2
    assert row_t1["n"] == 3
    assert row_t1["N"] == 3


@pytest.mark.api
def test_run_cluster_hypergeom_empty_rows_schema(toy_matrix):
    """
    Ensures empty enrichments return the expected schema.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
    """
    clusters = cluster(toy_matrix, linkage_threshold=1.0)
    annotations = Annotations({"t1": ["a", "b", "c", "d"]}, toy_matrix)
    results = run_cluster_hypergeom(
        toy_matrix,
        clusters,
        annotations,
        min_overlap=3,
    )

    assert results.df.empty
    assert results.df.columns.tolist() == ["cluster", "term", "k", "K", "n", "N", "pval"]
