"""
tests/test_annotations
~~~~~~~~~~~~~~~~~~~~~~
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from himalayas import Annotations, Matrix


@pytest.mark.api
def test_annotations_filtering_warns():
    """
    Ensures annotations warn when terms are dropped after filtering.
    """
    df = pd.DataFrame(np.eye(3), index=["a", "b", "c"], columns=["a", "b", "c"])
    matrix = Matrix(df)
    # "t2" has K=1 (one valid matrix label) — dropped by default min_term_size=2.
    term_to_labels = {"t1": ["a", "b"], "t2": ["a"]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        annotations = Annotations(term_to_labels, matrix)

    assert set(annotations.term_to_labels.keys()) == {"t1"}
    assert annotations.terms == ["t1"]
    assert any("Dropped" in str(w.message) for w in caught)


@pytest.mark.api
def test_annotations_all_dropped_raises():
    """
    Ensures annotations raise when all terms are dropped.

    Raises:
        ValueError: If no terms overlap matrix labels.
    """
    df = pd.DataFrame(np.eye(2), index=["a", "b"], columns=["a", "b"])
    matrix = Matrix(df)
    with pytest.raises(ValueError):
        Annotations({"t1": ["x"]}, matrix)


@pytest.mark.api
def test_annotations_rejects_string_labels(toy_matrix):
    """
    Ensures string labels are rejected for term mappings.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.

    Raises:
        TypeError: If term labels are not iterable collections.
    """
    with pytest.raises(TypeError):
        Annotations({"t1": "abc"}, toy_matrix)


@pytest.mark.api
def test_annotations_rejects_non_iterable_labels(toy_matrix):
    """
    Ensures non-iterable labels are rejected.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.

    Raises:
        TypeError: If a term maps to a non-iterable label value.
    """
    with pytest.raises(TypeError):
        Annotations({"t1": 123}, toy_matrix)


@pytest.mark.api
def test_annotations_term_size_bounds():
    """
    Ensures min_term_size and max_term_size filter on post-intersection K.
    """
    df = pd.DataFrame(np.eye(4), index=["a", "b", "c", "d"], columns=range(4))
    matrix = Matrix(df)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        annotations = Annotations(
            {"small": ["a"], "mid": ["a", "b"], "large": ["a", "b", "c", "d"]},
            matrix,
            min_term_size=2,
            max_term_size=3,
        )
    assert set(annotations.term_to_labels.keys()) == {"mid"}


@pytest.mark.api
def test_annotations_empty_input_raises(toy_matrix):
    """
    Ensures empty annotations raise a ValueError.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.

    Raises:
        ValueError: If the annotations mapping is empty.
    """
    with pytest.raises(ValueError):
        Annotations({}, toy_matrix)


@pytest.mark.api
def test_annotations_gmt_round_trip(toy_matrix, tmp_path):
    """
    Ensures to_gmt() output is read back unchanged by from_gmt().

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
        tmp_path (Path): Temporary directory fixture.
    """
    original = Annotations({"t1": ["a", "b"], "t2": ["c", "d", "a"]}, toy_matrix)
    path = tmp_path / "terms.gmt"
    original.to_gmt(path)
    loaded = Annotations.from_gmt(path, toy_matrix)
    assert loaded.term_to_labels == original.term_to_labels
    assert path.read_text(encoding="utf-8").splitlines()[0] == "t1\t-\ta\tb"


@pytest.mark.api
def test_annotations_from_gmt_ignores_description_and_blank_lines(toy_matrix, tmp_path):
    """
    Ensures from_gmt() ignores the description field and blank lines, and applies size filters.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
        tmp_path (Path): Temporary directory fixture.
    """
    path = tmp_path / "terms.gmt"
    path.write_text("t1\tsome text\ta\tb\tz\n\nt2\tx\tc\td\n", encoding="utf-8")
    annotations = Annotations.from_gmt(path, toy_matrix)
    assert annotations.term_to_labels == {"t1": {"a", "b"}, "t2": {"c", "d"}}
    assert annotations.labels_for_universe(["z"]) == {"t1": {"z"}}


@pytest.mark.api
@pytest.mark.parametrize(
    "text",
    ["t1\ta\tb\n", "t1\tdesc\n", "t1\tx\ta\tb\nt1\ty\tc\td\n", "\n"],
)
def test_annotations_from_gmt_rejects_malformed_files(toy_matrix, tmp_path, text):
    """
    Ensures from_gmt() rejects a missing description, no labels, a repeated term, and no terms.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
        tmp_path (Path): Temporary directory fixture.
        text (str): GMT file content.

    Raises:
        ValueError: If the file is malformed.
    """
    path = tmp_path / "bad.gmt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        Annotations.from_gmt(path, toy_matrix)


@pytest.mark.api
def test_annotations_to_gmt_rejects_tabs_and_line_breaks(tmp_path):
    """
    Ensures to_gmt() refuses terms or labels that would corrupt the file.

    Args:
        tmp_path (Path): Temporary directory fixture.

    Raises:
        ValueError: If a label contains a tab.
    """
    matrix = Matrix(pd.DataFrame(np.eye(2), index=["a\tb", "c"], columns=["x", "y"]))
    annotations = Annotations({"t1": ["a\tb", "c"]}, matrix)
    with pytest.raises(ValueError):
        annotations.to_gmt(tmp_path / "bad.gmt")
