"""
tests/test_plot_smoke
~~~~~~~~~~~~~~~~~~~~~
"""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from himalayas import Analysis, Matrix
from himalayas.core import Annotations
from himalayas.plot import Plotter





def test_plotter_smoke():
    """
    Ensures Plotter can render a minimal plot without errors.
    """
    # Use a non-interactive backend for headless testing
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
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
    # Build a minimal cluster label table for the plotter
    cluster_labels = pd.DataFrame(
        {
            "cluster": [int(results.clusters.unique_clusters[0])],
            "label": ["Test"],
            "pval": [1.0],
        }
    )
    # Suppress GUI rendering during tests
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        Plotter(results).plot_matrix().plot_cluster_labels(cluster_labels).show()
    finally:
        plt.show = plt_show
