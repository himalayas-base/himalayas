"""
tests/test_results
~~~~~~~~~~~~~~~~~~
"""

import numpy as np
import pandas as pd
import pytest

from himalayas import Matrix, Results, cluster


@pytest.mark.api
def test_results_with_qvalues_adds_column():
    """
    Ensures q-values are added to results.
    """
    # Compute q-values on a small p-value table
    df = pd.DataFrame({"pval": [0.01, 0.2, 0.05]})
    res = Results(df, method="test")
    out = res.with_qvalues()

    assert "qval" in out.df.columns
    assert np.all(np.isfinite(out.df["qval"]))


@pytest.mark.api
def test_results_subset_requires_matrix_and_clusters():
    """
    Ensures subsetting requires matrix and clusters.
    """
    # Subsetting without attachments should fail
    df = pd.DataFrame({"pval": [0.1]})
    res = Results(df, method="test")
    with pytest.raises(ValueError):
        res.subset(cluster=1)


@pytest.mark.api
def test_results_subset_returns_new_matrix():
    """
    Ensures subsetting returns a new matrix and clears clusters.
    """
    # Build a tiny matrix and subset the first cluster
    df = pd.DataFrame(
        [[0.0], [1.0], [2.0]],
        index=["a", "b", "c"],
        columns=["x"],
    )
    matrix = Matrix(df)
    clusters = cluster(matrix, linkage_threshold=100.0)
    res = Results(pd.DataFrame(), method="test", matrix=matrix, clusters=clusters)
    sub = res.subset(cluster=int(clusters.unique_clusters[0]))

    assert sub.matrix is not None
    assert sub.clusters is None
