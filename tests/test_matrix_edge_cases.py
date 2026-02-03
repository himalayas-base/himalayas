"""
tests/test_matrix_edge_cases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pandas as pd
import pytest

from himalayas import Matrix, cluster


@pytest.mark.api
def test_cluster_rejects_non_numeric_matrix():
    """
    Ensures clustering fails on non-numeric data.
    """
    df = pd.DataFrame([["x"], ["y"]], index=["a", "b"], columns=["v"])
    matrix = Matrix(df)

    with pytest.raises(Exception):
        cluster(matrix)
