"""
tests/test_layout_utils
~~~~~~~~~~~~~~~~~~~~~~~
"""

import pandas as pd
import pytest

from himalayas import Matrix
from himalayas.core.layout import compute_col_order


@pytest.mark.api
def test_compute_col_order_length_matches_columns():
    """
    Ensures column order length matches matrix column count.
    """
    df = pd.DataFrame([[0.0, 1.0], [2.0, 3.0]], index=["a", "b"], columns=["x", "y"])
    matrix = Matrix(df)
    order = compute_col_order(matrix)

    assert len(order) == df.shape[1]


@pytest.mark.api
def test_compute_col_order_single_column():
    """
    Ensures column order works for a single-column matrix.
    """
    df = pd.DataFrame([[1.0], [2.0]], index=["a", "b"], columns=["x"])
    matrix = Matrix(df)
    order = compute_col_order(matrix)

    assert order.tolist() == [0]
