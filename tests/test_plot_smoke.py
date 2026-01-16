import pandas as pd
import pytest

from himalayas import Matrix, run_first_pass
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
    matrix = Matrix(df, matrix_semantics="distance")
    annotations = Annotations({"t1": ["a", "b"], "t2": ["c"]}, matrix)

    results = run_first_pass(
        matrix,
        annotations,
        linkage_threshold=100.0,
        add_qvalues=False,
        col_cluster=False,
    )

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
