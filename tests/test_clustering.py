"""
tests/test_clustering
~~~~~~~~~~~~~~~~~~~~~
"""

import pandas as pd

from himalayas import Matrix, cluster


def test_cluster_layout_spans_cover_all_rows():
    df = pd.DataFrame(
        [[0.0, 0.1], [0.0, 0.2], [5.0, 5.1], [5.0, 5.2]],
        index=["a", "b", "c", "d"],
        columns=["f1", "f2"],
    )
    matrix = Matrix(df)
    clusters = cluster(matrix, linkage_threshold=1.0)
    layout = clusters.layout()

    assert layout.ordered_labels.shape[0] == df.shape[0]
    assert sum(e - s + 1 for _, s, e in layout.cluster_spans) == df.shape[0]
