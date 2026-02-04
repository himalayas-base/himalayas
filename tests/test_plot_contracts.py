"""
tests/test_plot_contracts
~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from himalayas import Results, cluster
from himalayas.plot import Plotter
from himalayas.plot.track_layout import TrackLayoutManager


def _use_agg_backend():
    """
    Configures Matplotlib to use the Agg backend for tests.

    Returns:
        Any: Matplotlib pyplot module with Agg backend active.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    return plt


@pytest.mark.api
def test_plotter_requires_layers(toy_results):
    """
    Ensures Plotter refuses to render with no declared layers.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        RuntimeError: If no plot layers are declared.
    """
    plt = _use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        with pytest.raises(RuntimeError):
            Plotter(toy_results).show()
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plotter_requires_layout(toy_matrix):
    """
    Ensures Plotter errors when Results has no attached layout.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.

    Raises:
        ValueError: If Results has no attached layout.
    """
    clusters = cluster(toy_matrix, linkage_threshold=1.0)
    results = Results(pd.DataFrame(), method="test", matrix=toy_matrix, clusters=clusters)

    plt = _use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        with pytest.raises(ValueError):
            Plotter(results).plot_matrix().show()
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_bar_rejects_invalid_values(toy_results):
    """
    Ensures plot_cluster_bar enforces supported input types.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        TypeError: If values are not a supported type.
    """
    with pytest.raises(TypeError):
        Plotter(toy_results).plot_cluster_bar(name="sig", values=[1, 2, 3])


@pytest.mark.api
def test_plot_gene_bar_requires_colors(toy_results, toy_cluster_labels):
    """
    Ensures categorical gene bars require an explicit colors mapping.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        TypeError: If categorical mode is missing a colors mapping.
    """
    plt = _use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        with pytest.raises(TypeError):
            (
                Plotter(toy_results)
                .plot_matrix()
                .plot_cluster_labels(toy_cluster_labels)
                .plot_gene_bar(values={"a": "hit"}, mode="categorical")
                .show()
            )
    finally:
        plt.show = plt_show


@pytest.mark.unit
def test_track_layout_duplicate_names_raise():
    """
    Ensures duplicate track names are rejected at layout time.

    Raises:
        ValueError: If duplicate track names are registered.
    """
    manager = TrackLayoutManager()
    manager.register_track(name="a", renderer=lambda *args, **kwargs: None, width=0.1)
    manager.register_track(name="a", renderer=lambda *args, **kwargs: None, width=0.1)

    with pytest.raises(ValueError):
        manager.compute_layout(base_x=0.0, gutter_width=0.0)


@pytest.mark.unit
def test_track_layout_unknown_order_raises():
    """
    Ensures unknown track names in ordering raise an error.

    Raises:
        ValueError: If ordering includes unknown tracks.
    """
    manager = TrackLayoutManager()
    manager.register_track(name="a", renderer=lambda *args, **kwargs: None, width=0.1)
    manager.set_order(["missing"])

    with pytest.raises(ValueError):
        manager.compute_layout(base_x=0.0, gutter_width=0.0)
