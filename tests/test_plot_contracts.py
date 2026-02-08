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
def test_plot_cluster_bar_requires_cluster_labels_layer(toy_results):
    """
    Ensures cluster bars require the cluster label layer in the same plot chain.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        ValueError: If plot_cluster_bar is used without plot_cluster_labels.
    """
    plt = _use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        with pytest.raises(ValueError, match="plot_cluster_labels"):
            Plotter(toy_results).plot_matrix().plot_cluster_bar(name="sig").show()
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_bar_uses_internal_cluster_labels(toy_results):
    """
    Ensures plot_cluster_bar renders from internally generated cluster labels.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = _use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        (
            Plotter(toy_results)
            .plot_matrix()
            .plot_cluster_labels()
            .plot_cluster_bar(name="sig")
            .show()
        )
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_label_bar_requires_colors(toy_results):
    """
    Ensures categorical label bars require an explicit colors mapping.

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
                .plot_cluster_labels()
                .plot_label_bar(values={"a": "hit"}, mode="categorical")
                .show()
            )
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_respect_np_order(toy_results):
    """
    Ensures cluster labels keep label first while respecting n/p order from label_fields
    and emitting a single stats block.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = _use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_cluster_labels(
                label_fields=("label", "p", "n"),
            )
        )
        plotter.show()
        fig = plotter._fig
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        label_texts = [t for t in texts if "$p$=" in t and "n=" in t]
        assert label_texts, "Expected cluster label text to be rendered."
        for txt in label_texts:
            assert " (" in txt
            assert txt.strip().endswith(")")
            assert "$p$=" in txt
            assert "n=" in txt
            assert txt.find("$p$=") < txt.find("n=")
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_accepts_override_mapper(toy_results):
    """
    Ensures cluster label overrides can replace generated labels by cluster id.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = _use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        cid = int(toy_results.cluster_layout().cluster_spans[0][0])
        custom_label = f"Custom-{cid}"
        plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_cluster_labels(
                overrides={cid: custom_label},
                label_fields=("label",),
            )
        )
        plotter.show()
        fig = plotter._fig
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        assert any(custom_label in t for t in texts)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_summary_max_words_controls_label_building(toy_results):
    """
    Ensures summary_max_words controls compressed label building.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = _use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_cluster_labels(
                label_mode="compressed",
                summary_max_words=1,
                label_fields=("label",),
                wrap_text=False,
            )
        )
        plotter.show()
        fig = plotter._fig
        texts = [t.get_text().strip() for ax in fig.axes for t in ax.texts]
        cluster_texts = [t for t in texts if t and t != "—"]
        assert cluster_texts, "Expected generated cluster labels to be rendered."
        for txt in cluster_texts:
            assert len(txt.replace("\n", " ").split()) <= 1
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_max_words_controls_display(toy_results):
    """
    Ensures max_words truncates rendered override labels without affecting generation settings.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = _use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        cid = int(toy_results.cluster_layout().cluster_spans[0][0])
        plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_cluster_labels(
                overrides={cid: "Alpha Beta Gamma"},
                label_fields=("label",),
                max_words=1,
                wrap_text=False,
            )
        )
        plotter.show()
        fig = plotter._fig
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        assert any(t.strip() == "Alpha" for t in texts)
        assert not any("Alpha Beta" in t for t in texts)
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
