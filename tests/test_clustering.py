"""
tests/test_clustering
~~~~~~~~~~~~~~~~~~~~~
"""

import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage as scipy_linkage

from himalayas.core import clustering as clustering_module
from himalayas.core.clustering import cluster, compute_linkage, cut_linkage


@pytest.mark.api
def test_cluster_layout_spans_cover_all_rows(toy_matrix):
    """
    Ensures cluster spans cover all rows in dendrogram order.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
    """
    clusters = cluster(toy_matrix, linkage_threshold=1.0)
    layout = clusters.layout()

    assert layout.ordered_labels.shape[0] == toy_matrix.df.shape[0]
    assert sum(e - s + 1 for _, s, e in layout.cluster_spans) == toy_matrix.df.shape[0]


@pytest.mark.api
def test_cluster_min_cluster_size_too_large_raises(toy_matrix):
    """
    Ensures min_cluster_size larger than N raises a ValueError.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.

    Raises:
        ValueError: If min_cluster_size exceeds the number of rows.
    """
    with pytest.raises(ValueError):
        cluster(toy_matrix, linkage_threshold=1.0, min_cluster_size=999)


@pytest.mark.api
def test_cluster_layout_reflects_new_col_order_after_none(toy_matrix):
    """
    Ensures layout cache respects a later explicit column order after an initial None.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
    """
    clusters = cluster(toy_matrix, linkage_threshold=1.0)
    layout_default = clusters.layout(col_order=None)
    desired = np.arange(toy_matrix.df.shape[1], dtype=int)[::-1]
    layout_custom = clusters.layout(col_order=desired)

    assert layout_default.col_order is None
    assert layout_custom.col_order is not None
    assert np.array_equal(layout_custom.col_order, desired)


@pytest.mark.api
def test_cluster_layout_reflects_none_after_custom_col_order(toy_matrix):
    """
    Ensures layout cache respects a later None column order after an initial explicit order.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
    """
    clusters = cluster(toy_matrix, linkage_threshold=1.0)
    desired = np.arange(toy_matrix.df.shape[1], dtype=int)[::-1]
    layout_custom = clusters.layout(col_order=desired)
    layout_default = clusters.layout(col_order=None)

    assert layout_custom.col_order is not None
    assert np.array_equal(layout_custom.col_order, desired)
    assert layout_default.col_order is None


@pytest.mark.api
def test_cluster_layout_reuses_cached_object_for_same_inputs(toy_matrix):
    """
    Ensures identical layout() inputs reuse the cached layout instance.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
    """
    clusters = cluster(toy_matrix, linkage_threshold=1.0)
    desired = np.arange(toy_matrix.df.shape[1], dtype=int)[::-1]
    first = clusters.layout(col_order=desired)
    second = clusters.layout(col_order=desired)

    assert first is second


@pytest.mark.api
def test_compute_and_cut_linkage_matches_cluster(toy_matrix):
    """
    Ensures compute_linkage()+cut_linkage() matches cluster() semantics.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
    """
    direct = cluster(toy_matrix, linkage_threshold=1.0)
    linkage_matrix = compute_linkage(toy_matrix)
    split = cut_linkage(
        linkage_matrix,
        toy_matrix.labels,
        linkage_threshold=1.0,
    )

    assert np.array_equal(direct.cluster_ids, split.cluster_ids)
    assert np.array_equal(direct.leaf_order, split.leaf_order)


@pytest.mark.api
def test_compute_linkage_prefers_fastcluster_when_available(monkeypatch, toy_matrix):
    """
    Ensures compute_linkage() uses fastcluster when available and optimal_ordering is disabled.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing module call targets.
        toy_matrix (Matrix): Toy matrix fixture.
    """
    seen = {"fast_calls": 0}

    def _fake_fastcluster_linkage(values, method, metric):
        seen["fast_calls"] += 1
        return scipy_linkage(values, method=method, metric=metric, optimal_ordering=False)

    def _unexpected_scipy(*_args, **_kwargs):
        raise AssertionError("SciPy linkage should not be used when fastcluster is available")

    monkeypatch.setattr(
        clustering_module,
        "_resolve_fastcluster_linkage",
        lambda: _fake_fastcluster_linkage,
    )
    monkeypatch.setattr(clustering_module, "linkage", _unexpected_scipy)

    Z = compute_linkage(toy_matrix, optimal_ordering=False)

    assert seen["fast_calls"] == 1
    assert Z.shape[0] == toy_matrix.df.shape[0] - 1


@pytest.mark.api
def test_compute_linkage_falls_back_to_scipy_when_fastcluster_unavailable(
    monkeypatch, toy_matrix
):
    """
    Ensures compute_linkage() falls back to SciPy when fastcluster is unavailable.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing module call targets.
        toy_matrix (Matrix): Toy matrix fixture.
    """
    seen = {"kwargs": None}

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

    monkeypatch.setattr(clustering_module, "_resolve_fastcluster_linkage", lambda: None)
    monkeypatch.setattr(clustering_module, "linkage", _capture_scipy)

    compute_linkage(toy_matrix, linkage_method="average", linkage_metric="cosine")

    assert seen["kwargs"] is not None
    assert seen["kwargs"]["method"] == "average"
    assert seen["kwargs"]["metric"] == "cosine"
    assert seen["kwargs"]["optimal_ordering"] is False


@pytest.mark.api
def test_compute_linkage_uses_scipy_when_optimal_ordering_enabled(monkeypatch, toy_matrix):
    """
    Ensures compute_linkage() uses SciPy when optimal_ordering is enabled.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing module call targets.
        toy_matrix (Matrix): Toy matrix fixture.
    """
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

    monkeypatch.setattr(clustering_module, "_resolve_fastcluster_linkage", _unexpected_fastcluster)
    monkeypatch.setattr(clustering_module, "linkage", _capture_scipy)

    compute_linkage(toy_matrix, optimal_ordering=True)

    assert seen["kwargs"] is not None
    assert seen["kwargs"]["optimal_ordering"] is True
