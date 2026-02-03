"""
tests/test_matrix_edge_cases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pandas as pd
import pytest

from himalayas import Matrix


@pytest.mark.api
def test_matrix_rejects_non_numeric_values():
    """
    Ensures Matrix rejects non-numeric values.
    """
    df = pd.DataFrame([["x"], ["y"]], index=["a", "b"], columns=["v"])
    with pytest.raises(ValueError):
        Matrix(df)


@pytest.mark.api
def test_matrix_rejects_empty():
    """
    Ensures empty matrices are rejected.
    """
    df = pd.DataFrame([], index=[], columns=[])
    with pytest.raises(ValueError):
        Matrix(df)
