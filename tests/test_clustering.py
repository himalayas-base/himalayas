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


@pytest.mark.api
def test_cluster_auto_threshold_is_finite_numeric(toy_matrix):
    """
    Ensures linkage_threshold="auto" resolves to a finite numeric Clusters.threshold.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
    """
    clusters = cluster(toy_matrix, linkage_threshold="auto")

    assert isinstance(clusters.threshold, float)
    assert np.isfinite(clusters.threshold)


@pytest.mark.api
def test_cluster_auto_threshold_matches_independent_silhouette_diversity_argmax():
    """
    Critical correctness check: independently enumerates candidate thresholds from the
    linkage matrix and computes silhouette-times-diversity scores by hand, then asserts
    "auto" returns the threshold with the highest score. Uses a fixture with both negative
    and positive matrix values whose highest raw silhouette belongs to a coarse 2-cluster
    cut, while the finer, more balanced 3-cluster cut wins once weighted by diversity,
    demonstrating the combined objective favors it over pure silhouette maximization.
    """
    import pandas as pd
    from scipy.cluster.hierarchy import fcluster as scipy_fcluster
    from scipy.spatial.distance import pdist, squareform
    from sklearn.metrics import silhouette_score
    from himalayas import Matrix

    df = pd.DataFrame(
        [
            [-8.0, -8.0],
            [-7.8, -8.0],
            [-0.3, 0.0],
            [-0.2, 0.9],
            [-0.4, -0.1],
            [2.5, 1.7],
            [2.3, 1.2],
            [2.9, 2.5],
        ],
        index=["a", "b", "c", "d", "e", "f", "g", "h"],
        columns=["x", "y"],
    )
    matrix = Matrix(df)
    linkage_matrix = compute_linkage(matrix, linkage_method="ward", linkage_metric="euclidean")

    distance_matrix = squareform(pdist(matrix.values, metric="euclidean"))
    n = matrix.values.shape[0]
    expected_threshold = None
    expected_score = -np.inf
    for threshold in np.unique(linkage_matrix[:, 2]):
        labels = scipy_fcluster(linkage_matrix, threshold, criterion="distance")
        n_clusters = len(np.unique(labels))
        if n_clusters < 2 or n_clusters >= n:
            continue
        silhouette = silhouette_score(distance_matrix, labels, metric="precomputed")
        score = silhouette
        if silhouette > 0:
            _, counts = np.unique(labels, return_counts=True)
            proportions = counts / counts.sum()
            diversity = 1.0 - np.sum(proportions**2)
            score *= diversity
        if score > expected_score:
            expected_score = score
            expected_threshold = float(threshold)

    clusters = cluster(matrix, linkage_threshold="auto", linkage_method="ward", linkage_metric="euclidean")

    assert clusters.threshold == pytest.approx(expected_threshold)


@pytest.mark.api
def test_cluster_auto_threshold_independent_of_merge_small_clusters():
    """
    Ensures the resolved "auto" threshold is identical regardless of merge_small_clusters,
    proving min_cluster_size/merge_small_clusters (post-cut cleanup) cannot influence
    auto-threshold selection. Final cluster assignments may differ; the threshold must not.
    """
    import pandas as pd
    from himalayas import Matrix

    df = pd.DataFrame(
        [[0.0], [0.2], [5.0], [5.2], [10.0], [10.2]],
        index=["a", "b", "c", "d", "e", "f"],
        columns=["x"],
    )
    matrix = Matrix(df)
    result_merged = cluster(
        matrix, linkage_threshold="auto", min_cluster_size=2, merge_small_clusters=True
    )
    result_raw = cluster(
        matrix, linkage_threshold="auto", min_cluster_size=2, merge_small_clusters=False
    )

    assert result_merged.threshold == result_raw.threshold


@pytest.mark.api
def test_resolve_auto_threshold_breaks_ties_with_smallest_threshold(monkeypatch):
    """
    Ensures _resolve_auto_threshold breaks exact ties in the combined silhouette-times-
    diversity objective by choosing the smallest candidate threshold (the finer partition),
    as documented in its docstring. For the k=3 and k=2 candidates, silhouette_score is
    monkeypatched to return 0.25 divided by that candidate's own Gini-Simpson diversity, so
    the production multiplication (`silhouette * diversity`) reconstructs 0.25 for both,
    producing a genuine floating-point tie rather than a near-tie. k=4 is given a clearly
    lower objective so it cannot win outright.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing module call targets.
    """
    import pandas as pd
    from himalayas import Matrix

    df = pd.DataFrame(
        [[-10.0], [-9.9], [-0.1], [0.1], [9.9], [10.0]],
        index=["a", "b", "c", "d", "e", "f"],
        columns=["x"],
    )
    matrix = Matrix(df)
    linkage_matrix = compute_linkage(matrix, linkage_method="ward", linkage_metric="euclidean")

    def fake_silhouette(distance_matrix, labels, metric="precomputed"):
        _, counts = np.unique(labels, return_counts=True)
        n_clusters = int(counts.shape[0])
        if n_clusters in {2, 3}:
            proportions = counts / counts.sum()
            diversity = 1.0 - np.sum(proportions**2)
            return 0.25 / diversity
        return 0.1

    monkeypatch.setattr(clustering_module, "silhouette_score", fake_silhouette)

    resolved = clustering_module._resolve_auto_threshold(linkage_matrix, matrix, "euclidean")

    candidates = sorted(np.unique(linkage_matrix[:, 2]).tolist())
    tied_candidates = candidates[1:3]  # the k=3 and k=2 thresholds, excluding k=4 and k=1
    assert resolved == pytest.approx(min(tied_candidates))


@pytest.mark.api
def test_resolve_auto_threshold_falls_back_to_raw_silhouette_when_non_positive(monkeypatch):
    """
    Ensures that when every candidate's silhouette score is zero or negative, selection
    falls back to the greatest raw silhouette rather than an ordering distorted by
    multiplying with diversity. The k=3 candidate is given the greatest (least negative)
    silhouette despite not having the highest diversity among the candidates (k=4 does),
    proving diversity is not applied in the non-positive branch.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing module call targets.
    """
    import pandas as pd
    from himalayas import Matrix

    df = pd.DataFrame(
        [[-10.0], [-9.9], [-0.1], [0.1], [9.9], [10.0]],
        index=["a", "b", "c", "d", "e", "f"],
        columns=["x"],
    )
    matrix = Matrix(df)
    linkage_matrix = compute_linkage(matrix, linkage_method="ward", linkage_metric="euclidean")

    def fake_silhouette(distance_matrix, labels, metric="precomputed"):
        n_clusters = len(np.unique(labels))
        return {2: -0.5, 3: -0.1, 4: -0.3}.get(n_clusters, -1.0)

    monkeypatch.setattr(clustering_module, "silhouette_score", fake_silhouette)

    resolved = clustering_module._resolve_auto_threshold(linkage_matrix, matrix, "euclidean")

    candidates = sorted(np.unique(linkage_matrix[:, 2]).tolist())
    assert resolved == pytest.approx(candidates[1])


@pytest.mark.api
def test_cluster_auto_threshold_no_valid_candidate_raises():
    """
    Ensures linkage_threshold="auto" raises ValueError when no candidate threshold yields
    a scoreable (2 to N-1 cluster) partition, rather than silently falling back to a default.
    """
    import pandas as pd
    from himalayas import Matrix

    df = pd.DataFrame(
        [[0.0], [10.0]],
        index=["a", "b"],
        columns=["x"],
    )
    matrix = Matrix(df)
    with pytest.raises(ValueError, match="auto"):
        cluster(matrix, linkage_threshold="auto")


@pytest.mark.api
@pytest.mark.parametrize("bad_threshold", ["bad", "AUTO", ""])
def test_cluster_invalid_threshold_string_raises(toy_matrix, bad_threshold):
    """
    Ensures a string linkage_threshold other than exactly "auto" raises ValueError.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
        bad_threshold (str): Invalid string threshold value.
    """
    with pytest.raises(ValueError):
        cluster(toy_matrix, linkage_threshold=bad_threshold)


@pytest.mark.api
@pytest.mark.parametrize("bad_threshold", [True, False])
def test_cluster_boolean_threshold_raises(toy_matrix, bad_threshold):
    """
    Ensures a boolean linkage_threshold raises ValueError instead of being silently
    coerced through bool's int subclassing.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
        bad_threshold (bool): Invalid boolean threshold value.
    """
    with pytest.raises(ValueError):
        cluster(toy_matrix, linkage_threshold=bad_threshold)
