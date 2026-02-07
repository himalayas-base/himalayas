"""
tests/test_clustering
~~~~~~~~~~~~~~~~~~~~~
"""

import pytest

from himalayas import cluster


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
