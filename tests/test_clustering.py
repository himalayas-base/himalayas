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
def test_min_cluster_size_merges_singleton():
    """
    Ensures min_cluster_size=2 absorbs a singleton cluster into its parent,
    leaving no cluster smaller than the requested minimum.
    """
    import pandas as pd
    from himalayas import Matrix

    df = pd.DataFrame(
        [[0.0], [0.1], [5.0], [5.1], [10.0]],
        index=["a", "b", "c", "d", "e"],
        columns=["x"],
    )
    matrix = Matrix(df)
    clusters = cluster(matrix, linkage_threshold=0.5, min_cluster_size=2)

    assert all(sz >= 2 for sz in clusters.cluster_sizes.values())


@pytest.mark.api
def test_merge_small_clusters_defaults_to_true_and_matches_legacy_behavior():
    """
    Ensures merge_small_clusters defaults to True, so existing callers that omit it get
    identical results to explicitly passing merge_small_clusters=True.
    """
    import pandas as pd
    from himalayas import Matrix

    df = pd.DataFrame(
        [[0.0], [0.1], [5.0], [5.1], [10.0]],
        index=["a", "b", "c", "d", "e"],
        columns=["x"],
    )
    matrix = Matrix(df)
    default_clusters = cluster(matrix, linkage_threshold=0.5, min_cluster_size=2)
    explicit_clusters = cluster(
        matrix, linkage_threshold=0.5, min_cluster_size=2, merge_small_clusters=True
    )

    assert default_clusters.merge_small_clusters is True
    assert np.array_equal(default_clusters.cluster_ids, explicit_clusters.cluster_ids)
    assert all(sz >= 2 for sz in default_clusters.cluster_sizes.values())


@pytest.mark.api
def test_merge_small_clusters_false_preserves_small_dendrogram_cut_clusters():
    """
    Ensures merge_small_clusters=False preserves a singleton dendrogram-cut cluster
    structurally instead of merging it upward, even though it is smaller than
    min_cluster_size.
    """
    import pandas as pd
    from himalayas import Matrix

    df = pd.DataFrame(
        [[0.0], [0.1], [5.0], [5.1], [10.0]],
        index=["a", "b", "c", "d", "e"],
        columns=["x"],
    )
    matrix = Matrix(df)
    clusters = cluster(
        matrix,
        linkage_threshold=0.5,
        min_cluster_size=2,
        merge_small_clusters=False,
    )

    assert clusters.min_cluster_size == 2
    assert clusters.merge_small_clusters is False
    # The singleton "e" cluster is preserved rather than merged upward.
    assert any(sz < 2 for sz in clusters.cluster_sizes.values())
    assert clusters.cluster_to_labels[clusters.label_to_cluster["e"]] == {"e"}

    # Layout must still surface the small cluster as its own contiguous span.
    layout = clusters.layout()
    small_cid = clusters.label_to_cluster["e"]
    assert any(cid == small_cid for cid, _, _ in layout.cluster_spans)
    assert layout.cluster_sizes[small_cid] == 1


@pytest.mark.api
def test_merge_small_clusters_coerced_to_bool(toy_matrix):
    """
    Ensures merge_small_clusters is coerced to bool, consistent with how other boolean
    kwargs (e.g. optimal_ordering) are handled in this module.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
    """
    clusters = cluster(
        toy_matrix,
        linkage_threshold=1.0,
        min_cluster_size=1,
        merge_small_clusters=0,
    )
    assert clusters.merge_small_clusters is False


@pytest.mark.api
def test_merge_small_clusters_truthy_int_behaves_like_true():
    """
    Ensures a truthy non-bool value (e.g. 1) for merge_small_clusters is stored as True
    and actually triggers merge behavior, not just the stored attribute. Guards against
    the merge condition checking argument identity (`merge_small_clusters is True`)
    instead of the coerced `self.merge_small_clusters` attribute.
    """
    import pandas as pd
    from himalayas import Matrix

    df = pd.DataFrame(
        [[0.0], [0.1], [5.0], [5.1], [10.0]],
        index=["a", "b", "c", "d", "e"],
        columns=["x"],
    )
    matrix = Matrix(df)
    clusters = cluster(
        matrix,
        linkage_threshold=0.5,
        min_cluster_size=2,
        merge_small_clusters=1,
    )

    assert clusters.merge_small_clusters is True
    # Behavior must match merge_small_clusters=True: no cluster smaller than min_cluster_size.
    assert all(sz >= 2 for sz in clusters.cluster_sizes.values())


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
def test_cut_linkage_propagates_merge_small_clusters(toy_matrix):
    """
    Ensures cluster(), compute_linkage()+cut_linkage() agree on merge_small_clusters=False,
    confirming the flag is propagated consistently across the low-level API.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
    """
    direct = cluster(toy_matrix, linkage_threshold=1.0, merge_small_clusters=False)
    linkage_matrix = compute_linkage(toy_matrix)
    split = cut_linkage(
        linkage_matrix,
        toy_matrix.labels,
        linkage_threshold=1.0,
        merge_small_clusters=False,
    )

    assert direct.merge_small_clusters is False
    assert split.merge_small_clusters is False
    assert np.array_equal(direct.cluster_ids, split.cluster_ids)


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
def test_compute_linkage_falls_back_to_scipy_when_fastcluster_unavailable(monkeypatch, toy_matrix):
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


@pytest.mark.api
def test_compute_linkage_correlation_raises_early_on_zero_variance_row():
    """
    Regression: sparse binary matrix with an all-zero row must raise a HiMaLAYAS-owned
    ValueError before scipy/fastcluster is invoked, not the opaque downstream error
    "The condensed distance matrix must contain only finite values."
    """
    import pandas as pd
    from himalayas import Matrix

    df = pd.DataFrame(
        [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
        index=["row_a", "row_b", "row_c"],
        columns=["c1", "c2", "c3"],
    )
    matrix = Matrix(df)
    with pytest.raises(ValueError, match="Correlation distance is undefined for constant rows") as excinfo:
        compute_linkage(matrix, linkage_method="average", linkage_metric="correlation")

    assert "row_b" in str(excinfo.value)


@pytest.mark.api
def test_compute_linkage_cosine_raises_early_on_zero_norm_row():
    """
    Regression: matrix with an all-zero row must raise a HiMaLAYAS-owned ValueError for
    linkage_metric='cosine' before scipy/fastcluster is invoked, not the opaque downstream
    error "The condensed distance matrix must contain only finite values."
    """
    import pandas as pd
    from himalayas import Matrix

    df = pd.DataFrame(
        [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
        index=["row_a", "row_b", "row_c"],
        columns=["c1", "c2", "c3"],
    )
    matrix = Matrix(df)
    with pytest.raises(ValueError, match="Cosine distance is undefined for zero vectors") as excinfo:
        compute_linkage(matrix, linkage_method="average", linkage_metric="cosine")

    assert "row_b" in str(excinfo.value)
