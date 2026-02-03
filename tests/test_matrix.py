"""
tests/test_matrix
~~~~~~~~~~~~~~~~~
"""

import pandas as pd
import pytest

from himalayas.core import Matrix


def test_matrix_labels_unique():
    """
    Ensures matrix labels are preserved when unique.
    """
    # Build a simple square matrix with unique labels
    df = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["a", "b"],
        columns=["a", "b"],
    )
    matrix = Matrix(df)
    assert matrix.labels.tolist() == ["a", "b"]


def test_matrix_duplicate_labels_raise():
    """
    Ensures duplicate matrix labels raise a ValueError.
    """
    # Construct a matrix with duplicate row labels
    df = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["a", "a"],
        columns=["x", "y"],
    )
    with pytest.raises(ValueError):
        Matrix(df)
