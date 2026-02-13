"""
tests/test_plot_contracts
~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.colors import Normalize

from himalayas import Results, cluster
from himalayas.plot import Plotter
from himalayas.plot.renderers.cluster_labels import _build_label_map, _parse_label_overrides
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


@pytest.mark.unit
def test_plot_colorbars_tick_decimals_validation(toy_results):
    """
    Ensures plot_colorbars validates tick_decimals type and range.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        TypeError: If tick_decimals is not an integer.
        ValueError: If tick_decimals is negative.
    """
    with pytest.raises(TypeError, match="tick_decimals"):
        Plotter(toy_results).plot_colorbars(tick_decimals=1.5)
    with pytest.raises(TypeError, match="tick_decimals"):
        Plotter(toy_results).plot_colorbars(tick_decimals=True)
    with pytest.raises(ValueError, match="tick_decimals"):
        Plotter(toy_results).plot_colorbars(tick_decimals=-1)


@pytest.mark.api
def test_plot_colorbars_tick_decimals_caps_display_precision(toy_results):
    """
    Ensures colorbar tick labels honor the configured decimal cap.

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
            .add_colorbar(
                name="precision",
                cmap="viridis",
                norm=Normalize(vmin=0.0, vmax=1.0),
                ticks=[0.0, 0.1234, 1.0],
                label="Precision",
            )
            .plot_colorbars(tick_decimals=2)
        )
        plotter.show()
        fig = plotter._fig
        tick_texts = [t.get_text() for ax in fig.axes for t in ax.get_xticklabels()]
        tick_texts = [txt for txt in tick_texts if txt]
        assert "0.12" in tick_texts
        assert "0.1234" not in tick_texts
        assert all("." not in txt or len(txt.split(".", 1)[1]) <= 2 for txt in tick_texts)
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
        # Cluster labels are asserted via rendered text fragments rather than data objects.
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

        # Empty-string override is a valid way to suppress one cluster label and
        # must not break rendering of cluster-level bars.
        plotter2 = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_cluster_labels(overrides={cid: ""}, label_fields=("label",))
            .plot_cluster_bar(name="sig")
        )
        plotter2.show()
        fig2 = plotter2._fig
        patch_counts = [len(ax.patches) for ax in fig2.axes]
        assert max(patch_counts) > 1
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_override_respects_np_field_contract(toy_results):
    """
    Ensures label_fields controls displayed stats/order even when label overrides are provided.

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
                overrides={cid: "CustomLabelIgnoredByFields"},
                label_fields=("n", "p"),
                wrap_text=False,
            )
        )
        plotter.show()
        fig = plotter._fig
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        # Labels are rendered as raw text; this selects n/p-only cluster entries.
        np_texts = [t for t in texts if t.startswith("(") and "n=" in t and "$p$=" in t]
        assert np_texts, "Expected n/p-only cluster label text."
        assert all("CustomLabelIgnoredByFields" not in t for t in np_texts)
        for txt in np_texts:
            assert txt.find("n=") < txt.find("$p$=")
    finally:
        plt.show = plt_show


@pytest.mark.unit
def test_plot_cluster_label_override_keeps_deterministic_stats(toy_results):
    """
    Ensures label-only overrides preserve deterministic cluster statistics.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    df = toy_results.cluster_labels()
    cid = int(df.loc[0, "cluster"])
    base_pval = float(df.loc[df["cluster"] == cid, "pval"].iloc[0])
    override_map = _parse_label_overrides({cid: f"Custom-{cid}"})

    label_map = _build_label_map(df, override_map)

    assert label_map[cid][0] == f"Custom-{cid}"
    assert float(label_map[cid][1]) == pytest.approx(base_pval)


@pytest.mark.api
def test_plot_cluster_labels_max_words_controls_label_building_compressed(toy_results):
    """
    Ensures max_words controls compressed label building.

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
                max_words=1,
                label_fields=("label",),
                wrap_text=False,
            )
        )
        plotter.show()
        fig = plotter._fig
        texts = [t.get_text().strip() for ax in fig.axes for t in ax.texts]
        # Exclude placeholders to focus on generated cluster label text.
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
