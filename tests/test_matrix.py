import pandas as pd
import pytest

from himalayas.core import Matrix


def test_matrix_labels_unique():
    df = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["a", "b"],
        columns=["a", "b"],
    )
    matrix = Matrix(df)
    assert matrix.labels.tolist() == ["a", "b"]


def test_matrix_duplicate_labels_raise():
    df = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["a", "a"],
        columns=["x", "y"],
    )
    with pytest.raises(ValueError):
        Matrix(df)
