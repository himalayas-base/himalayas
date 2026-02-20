"""
tests/test_layout_utils
~~~~~~~~~~~~~~~~~~~~~~~
"""

import pandas as pd
import pytest
from scipy.cluster.hierarchy import linkage as scipy_linkage

from himalayas import Matrix
from himalayas.core import layout as layout_module
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


@pytest.mark.api
def test_compute_col_order_prefers_fastcluster_when_available(monkeypatch):
    """
    Ensures compute_col_order() uses fastcluster when available and optimal_ordering is disabled.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing module call targets.
    """
    df = pd.DataFrame([[0.0, 1.0], [2.0, 3.0]], index=["a", "b"], columns=["x", "y"])
    matrix = Matrix(df)
    seen = {"fast_calls": 0}

    def _fake_fastcluster_linkage(values, method, metric):
        seen["fast_calls"] += 1
        return scipy_linkage(values, method=method, metric=metric, optimal_ordering=False)

    def _unexpected_scipy(*_args, **_kwargs):
        raise AssertionError("SciPy linkage should not be used when fastcluster is available")

    monkeypatch.setattr(
        layout_module,
        "_resolve_fastcluster_linkage",
        lambda: _fake_fastcluster_linkage,
    )
    monkeypatch.setattr(layout_module, "linkage", _unexpected_scipy)

    order = compute_col_order(matrix, optimal_ordering=False)

    assert seen["fast_calls"] == 1
    assert len(order) == matrix.df.shape[1]


@pytest.mark.api
def test_compute_col_order_uses_scipy_when_optimal_ordering_enabled(monkeypatch):
    """
    Ensures compute_col_order() uses SciPy when optimal_ordering is enabled.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing module call targets.
    """
    df = pd.DataFrame([[0.0, 1.0], [2.0, 3.0]], index=["a", "b"], columns=["x", "y"])
    matrix = Matrix(df)
    seen = {"kwargs": None}

    def _unexpected_fastcluster():
        raise AssertionError("fastcluster should not be used when optimal_ordering=True")

    def _capture_scipy(values, method, metric, optimal_ordering):
        seen["kwargs"] = {
            "method": method,
            "metric": metric,
            "optimal_ordering": bool(optimal_ordering),
        }
        return scipy_linkage(
            values,
            method=method,
            metric=metric,
            optimal_ordering=optimal_ordering,
        )

    monkeypatch.setattr(layout_module, "_resolve_fastcluster_linkage", _unexpected_fastcluster)
    monkeypatch.setattr(layout_module, "linkage", _capture_scipy)

    compute_col_order(matrix, linkage_method="average", linkage_metric="cosine", optimal_ordering=True)

    assert seen["kwargs"] is not None
    assert seen["kwargs"]["method"] == "average"
    assert seen["kwargs"]["metric"] == "cosine"
    assert seen["kwargs"]["optimal_ordering"] is True
