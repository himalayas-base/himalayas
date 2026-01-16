import warnings

import numpy as np
import pandas as pd
import pytest

from himalayas.core import Annotations, Matrix


def test_annotations_filtering_warns():
    df = pd.DataFrame(np.eye(3), index=["a", "b", "c"], columns=["a", "b", "c"])
    matrix = Matrix(df)
    term_to_labels = {"t1": ["a", "b"], "t2": ["x"]}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        annotations = Annotations(term_to_labels, matrix)

    assert set(annotations.term_to_labels.keys()) == {"t1"}
    assert any("Dropped" in str(w.message) for w in caught)


def test_annotations_all_dropped_raises():
    df = pd.DataFrame(np.eye(2), index=["a", "b"], columns=["a", "b"])
    matrix = Matrix(df)
    with pytest.raises(ValueError):
        Annotations({"t1": ["x"]}, matrix)
