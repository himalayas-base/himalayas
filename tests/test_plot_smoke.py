"""
tests/test_plot_smoke
~~~~~~~~~~~~~~~~~~~~~
"""

import pandas as pd
import pytest

from himalayas import Analysis, Matrix
from himalayas.core import Annotations


def test_plotter_smoke():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from himalayas.plot import Plotter

    df = pd.DataFrame(
        [[0.0], [1.0], [2.0]],
        index=["a", "b", "c"],
        columns=["x"],
    )
    matrix = Matrix(df)
    annotations = Annotations({"t1": ["a", "b"], "t2": ["c"]}, matrix)

    analysis = (
        Analysis(matrix, annotations)
        .cluster(linkage_threshold=100.0)
        .enrich()
        .finalize(
            add_qvalues=False,
            col_cluster=False,
        )
    )
    results = analysis.results

    cluster_labels = pd.DataFrame(
        {
            "cluster": [int(results.clusters.unique_clusters[0])],
            "label": ["Test"],
            "pval": [1.0],
        }
    )

    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        Plotter(results).plot_matrix().plot_cluster_labels(cluster_labels).show()
    finally:
        plt.show = plt_show
