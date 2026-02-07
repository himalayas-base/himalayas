"""
tests/test_plot_smoke
~~~~~~~~~~~~~~~~~~~~~
"""

import matplotlib.pyplot as plt
import pytest

from himalayas.plot import Plotter


@pytest.mark.api
def test_plotter_smoke(toy_results):
    """
    Ensures Plotter can render a minimal plot without errors.

    Args:
        toy_results (Results): Results fixture for plotting.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    # Suppress GUI rendering during tests
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        Plotter(toy_results).plot_matrix().plot_cluster_labels().show()
    finally:
        plt.show = plt_show
