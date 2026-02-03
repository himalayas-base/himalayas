"""
tests/test_annotations
~~~~~~~~~~~~~~~~~~~~~~
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from himalayas.core import Annotations, Matrix


def test_annotations_filtering_warns():
    """
    Ensures annotations warn when terms are dropped after filtering.
    """
    df = pd.DataFrame(np.eye(3), index=["a", "b", "c"], columns=["a", "b", "c"])
    matrix = Matrix(df)
    term_to_labels = {"t1": ["a", "b"], "t2": ["x"]}
    # Capture warnings while constructing annotations
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        annotations = Annotations(term_to_labels, matrix)

    assert set(annotations.term_to_labels.keys()) == {"t1"}
    assert any("Dropped" in str(w.message) for w in caught)


def test_annotations_all_dropped_raises():
    """
    Ensures annotations raise when all terms are dropped.
    """
    df = pd.DataFrame(np.eye(2), index=["a", "b"], columns=["a", "b"])
    matrix = Matrix(df)
    with pytest.raises(ValueError):
        Annotations({"t1": ["x"]}, matrix)
