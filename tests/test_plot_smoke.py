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
        (
            Plotter(toy_results)
            .set_label_panel(
                axes=[0.70, 0.05, 0.29, 0.90],
                text_pad=0.01,
            )
            .plot_matrix()
            .plot_cluster_labels()
            .show()
        )
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plotter_stacked_defaults_smoke(toy_results):
    """
    Ensures a stacked default Plotter chain renders without explicit style kwargs.

    Args:
        toy_results (Results): Results fixture for plotting.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    # Suppress GUI rendering during tests
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        (
            Plotter(toy_results)
            .set_label_panel(
                axes=[0.70, 0.05, 0.29, 0.90],
                track_x=0.02,
                gutter_width=0.01,
                text_pad=0.01,
            )
            .plot_dendrogram()
            .plot_matrix()
            .plot_matrix_axis_labels()
            .plot_row_ticks()
            .plot_col_ticks()
            .plot_cluster_labels()
            .plot_cluster_bar(name="sigbar")
            .show()
        )
    finally:
        plt.show = plt_show
