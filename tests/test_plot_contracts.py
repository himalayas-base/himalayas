"""
tests/test_plot_contracts
~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import Normalize, to_rgba
from matplotlib.offsetbox import AnnotationBbox
from scipy.cluster.hierarchy import dendrogram

from conftest import extract_figure_text, use_agg_backend
from himalayas import Analysis, Annotations, Matrix, Results
from himalayas.core.clustering import cluster
from himalayas.plot import Plotter
from himalayas.plot.renderers._label_format import format_label_prefix
from himalayas.plot.renderers.cluster_labels import _build_label_map, _parse_label_overrides
from himalayas.plot.style import DEFAULT_STYLE, StyleConfig
from himalayas.plot.track_layout import TrackLayoutManager


@pytest.fixture(autouse=True)
def _close_all_figures_after_each_test():
    """
    Ensures pyplot figures do not accumulate across tests in this module.
    """
    yield
    plt.close("all")


@pytest.mark.api
def test_plotter_smoke(toy_results):
    """
    Ensures Plotter can render a minimal plot without errors.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
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
def test_plotter_set_background_sets_figure_facecolor(toy_results):
    """
    Ensures set_background() applies the requested figure facecolor when rendered.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = Plotter(toy_results).plot_matrix().set_background("red")
        plotter.show()
        assert plotter._fig.patch.get_facecolor() == to_rgba("red")
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plotter_set_figure_figsize_applies_without_matrix(toy_results):
    """
    Ensures set_figure(figsize=...) sizes a matrix-less colorbars-only figure.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = (
            Plotter(toy_results)
            .set_figure(figsize=(4.2, 1.0))
            .add_colorbar(name="matrix", cmap="RdBu_r", norm=Normalize(-1, 1))
            .plot_colorbars()
        )
        plotter.show()
        assert plotter._fig.get_size_inches() == pytest.approx((4.2, 1.0))
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plotter_set_figure_figsize_applies_with_matrix(toy_results):
    """
    Ensures set_figure(figsize=...) sizes the figure when a matrix layer is declared.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = Plotter(toy_results).set_figure(figsize=(4.2, 1.0)).plot_matrix()
        plotter.show()
        assert plotter._fig.get_size_inches() == pytest.approx((4.2, 1.0))
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plotter_set_figure_subplots_adjust_applies_without_matrix(toy_results):
    """
    Ensures set_figure(subplots_adjust=...) applies to a matrix-less colorbars-only figure.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = (
            Plotter(toy_results)
            .set_figure(subplots_adjust={"left": 0.3})
            .add_colorbar(name="matrix", cmap="RdBu_r", norm=Normalize(-1, 1))
            .plot_colorbars()
        )
        plotter.show()
        assert plotter._fig.subplotpars.left == 0.3
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plotter_set_figure_validation_errors(toy_results):
    """
    Ensures set_figure() rejects malformed figsize/subplots_adjust arguments.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    with pytest.raises(TypeError):
        Plotter(toy_results).set_figure(figsize=(1, 2, 3))
    with pytest.raises(ValueError):
        Plotter(toy_results).set_figure(figsize=(0, 1))
    with pytest.raises(TypeError):
        Plotter(toy_results).set_figure(subplots_adjust="bad")


@pytest.mark.api
def test_plotter_stacked_defaults_smoke(toy_results):
    """
    Ensures a stacked default Plotter chain renders without explicit style kwargs
    and forwards row/column tick padding.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = (
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
            .plot_row_ticks(pad=7)
            .plot_col_ticks(pad=11)
            .plot_cluster_labels()
            .plot_cluster_bar(name="sigbar")
        )
        plotter.show()
        row_tick_axes = [ax for ax in plotter._fig.axes if len(ax.get_yticklabels()) > 0]
        col_tick_axes = [ax for ax in plotter._fig.axes if len(ax.get_xticklabels()) > 0]
        assert row_tick_axes
        assert col_tick_axes
        assert row_tick_axes[0].yaxis.majorTicks[0].get_pad() == pytest.approx(7.0)
        assert col_tick_axes[0].xaxis.majorTicks[0].get_pad() == pytest.approx(11.0)
    finally:
        plt.show = plt_show


@pytest.mark.api
@pytest.mark.parametrize(
    "stack",
    [
        # Keep this matrix to valid combinations only: legend requires row_bar.
        # Cluster bars render standalone, so they need no cluster_labels layer.
        {
            "cluster_labels": False,
            "cluster_bar": False,
            "row_bar": False,
            "legend": False,
            "colorbar": False,
        },
        {
            "cluster_labels": True,
            "cluster_bar": False,
            "row_bar": False,
            "legend": False,
            "colorbar": False,
        },
        {
            "cluster_labels": True,
            "cluster_bar": True,
            "row_bar": False,
            "legend": False,
            "colorbar": False,
        },
        {
            "cluster_labels": False,
            "cluster_bar": False,
            "row_bar": True,
            "legend": False,
            "colorbar": False,
        },
        {
            "cluster_labels": False,
            "cluster_bar": False,
            "row_bar": True,
            "legend": True,
            "colorbar": False,
        },
        {
            "cluster_labels": False,
            "cluster_bar": False,
            "row_bar": False,
            "legend": False,
            "colorbar": True,
        },
        {
            "cluster_labels": False,
            "cluster_bar": False,
            "row_bar": True,
            "legend": True,
            "colorbar": True,
        },
        {
            "cluster_labels": True,
            "cluster_bar": True,
            "row_bar": True,
            "legend": True,
            "colorbar": True,
        },
    ],
    ids=[
        "matrix_only",
        "matrix_plus_cluster_labels",
        "matrix_plus_cluster_labels_and_bar",
        "matrix_plus_row_bar",
        "matrix_plus_row_bar_and_legend",
        "matrix_plus_colorbar",
        "matrix_plus_row_bar_legend_and_colorbar",
        "full_stack",
    ],
)
def test_plotter_stack_combinations_smoke(toy_results, stack):
    """
    Ensures representative plot-layer stack combinations render without fatal interactions.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
        stack (dict): Boolean feature switches for enabled layers.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = Plotter(toy_results).plot_matrix()
        # Each layer is tested in isolation in other tests, so we do not assert on the
        # presence of rendered elements here. We just want to ensure no fatal errors occur
        # when stacking layers in various combinations.
        if stack["cluster_labels"]:
            plotter = plotter.plot_cluster_labels(label_fields=("label",))
        if stack["cluster_bar"]:
            plotter = plotter.plot_cluster_bar(name="sig")
        if stack["row_bar"]:
            plotter = plotter.plot_label_bar(
                values={"a": "1", "b": "2", "c": "1", "d": "2"},
                name="orf_category",
                mode="categorical",
                colors={"1": "#1b9e77", "2": "#d95f02"},
                title="ORF Category",
            )
        if stack["legend"]:
            plotter = plotter.add_label_legend(name="orf_category").plot_label_legends(
                height=0.07, gap=0.02, swatch_scale=0.9
            )
        if stack["colorbar"]:
            plotter = plotter.add_colorbar(
                name="score",
                cmap="viridis",
                norm=Normalize(vmin=0.0, vmax=1.0),
                ticks=[0.0, 1.0],
                label="Score",
            ).plot_colorbars()

        plotter.show()
        assert plotter._fig is not None
        if stack["legend"]:
            assert plotter.label_legend_layout_ is not None
        else:
            assert plotter.label_legend_layout_ is None
        if stack["colorbar"]:
            assert plotter.colorbar_layout_ is not None
        else:
            assert plotter.colorbar_layout_ is None
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plotter_requires_layers(toy_results):
    """
    Ensures Plotter refuses to render with no declared layers.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        RuntimeError: If no plot layers are declared.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        with pytest.raises(RuntimeError):
            Plotter(toy_results).show()
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plotter_colorbars_only_renders_without_matrix(toy_results):
    """
    Ensures a colorbars-only chain renders without plot_matrix() or any label layer.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        (
            Plotter(toy_results)
            .add_colorbar(name="matrix", cmap="RdBu_r", norm=Normalize(-1, 1))
            .plot_colorbars()
            .show()
        )
    finally:
        plt.show = plt_show


@pytest.mark.api
@pytest.mark.parametrize("cmap_as_object", [False, True])
@pytest.mark.parametrize("extension", ["png", "pdf", "svg"])
def test_plotter_colorbar_exports(toy_results, tmp_path, cmap_as_object, extension):
    """
    Ensures named and object colormaps render gradients and export to raster and vector files.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
        tmp_path (Path): Temporary output directory.
        cmap_as_object (bool): Whether to supply a Colormap object instead of a name.
        extension (str): Export format.
    """
    plt = use_agg_backend()
    cmap = plt.get_cmap("RdBu_r") if cmap_as_object else "RdBu_r"
    plotter = (
        Plotter(toy_results)
        .add_colorbar(name="matrix", cmap=cmap, norm=Normalize(-1, 1))
        .plot_colorbars()
    )
    out = tmp_path / f"colorbar.{extension}"
    plotter.save(out, dpi=100, bbox_inches="tight")
    assert out.stat().st_size > 1000
    fig = plt.gcf()
    fig.canvas.draw()
    # Verify the rendered mappable uses the requested end-point colors.
    meshes = [
        artist
        for ax in fig.axes
        for artist in ax.collections
        if artist.get_array() is not None and artist.get_array().size > 1
    ]
    assert meshes
    np.testing.assert_allclose(meshes[-1].cmap([0.0, 1.0]), plt.get_cmap("RdBu_r")([0.0, 1.0]))


@pytest.mark.api
def test_plotter_colorbars_only_save_crops_tightly(toy_results, tmp_path):
    """
    Ensures a colorbars-only save with bbox_inches="tight" crops around the
    colorbar strip instead of retaining the matrix-sized placeholder bbox.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
        tmp_path (Path): Temporary output directory.
    """
    import matplotlib.image as mpimg

    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = (
            Plotter(toy_results)
            .add_colorbar(name="matrix", cmap="RdBu_r", norm=Normalize(-1, 1))
            .plot_colorbars()
        )
        out = tmp_path / "colorbars_only.png"
        dpi = 100
        plotter.save(out, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        height_px = mpimg.imread(out).shape[0]
        # Default figsize is 7in tall; a tight colorbar-only crop should be
        # a small fraction of that, not the full matrix-sized placeholder.
        assert height_px < 2.0 * dpi
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plotter_cluster_labels_without_matrix_hides_placeholder_chrome(toy_results):
    """
    Ensures a matrix-less cluster-label + colorbar chain hides the whole placeholder
    axes instead of leaving a visible empty matrix box, and draws no matrix image.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = (
            Plotter(toy_results)
            .plot_cluster_labels()
            .add_colorbar(name="matrix", cmap="RdBu_r", norm=Normalize(-1, 1))
            .plot_colorbars()
        )
        plotter.show()
        ax0 = plotter._fig.axes[0]
        assert ax0.get_visible() is False
        assert all(len(ax.images) == 0 for ax in plotter._fig.axes)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plotter_compact_labels_without_matrix_hides_placeholder_chrome(toy_results):
    """
    Ensures a matrix-less compact-label + colorbar chain hides the whole placeholder
    axes and draws no matrix image.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = (
            Plotter(toy_results)
            .plot_cluster_labels_compact()
            .add_colorbar(name="enrichment", cmap="YlOrBr", norm=Normalize(0, 30))
            .plot_colorbars()
        )
        plotter.show()
        ax0 = plotter._fig.axes[0]
        assert ax0.get_visible() is False
        assert all(len(ax.images) == 0 for ax in plotter._fig.axes)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plotter_title_without_matrix_hides_placeholder_axes(toy_results):
    """
    Ensures plot_title() without plot_matrix() still sets title text on the
    placeholder axes, even though the whole axes (and thus the title) is now
    excluded from tight-bbox cropping. Standalone title placement is out of
    scope here; see plot_cluster_labels_without_matrix_hides_placeholder_chrome.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = Plotter(toy_results).plot_title("X").plot_cluster_labels()
        plotter.show()
        ax0 = plotter._fig.axes[0]
        assert ax0.get_title() == "X"
        assert ax0.get_visible() is False
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plotter_matrix_present_keeps_placeholder_chrome_visible(toy_results):
    """
    Ensures a standard plot_matrix() chain keeps the main axes patch/spines visible
    and draws exactly one matrix image, unaffected by matrix-less handling.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = Plotter(toy_results).plot_matrix().plot_cluster_labels()
        plotter.show()
        ax0 = plotter._fig.axes[0]
        assert ax0.get_visible() is True
        assert ax0.patch.get_visible() is True
        assert all(spine.get_visible() for spine in ax0.spines.values())
        assert len(ax0.images) == 1
    finally:
        plt.show = plt_show


@pytest.mark.api
@pytest.mark.parametrize("outer_lw", [2.0, 10.0])
def test_plot_cluster_labels_matrix_border_stays_solid_at_boundary(toy_results, outer_lw):
    """
    Ensures the matrix's right spine renders fully at the shared matrix/label-panel
    boundary instead of being clipped or painted over by a flush label gutter.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
        outer_lw (float): Matrix border linewidth under test.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        matrix_right = 0.70
        plotter = (
            Plotter(toy_results)
            .set_figure(figsize=(9, 15))
            .plot_matrix(outer_lw=outer_lw, outer_color="black")
            .set_label_panel(
                axes=[matrix_right, 0.05, 0.29, 0.90],
                gutter_color="white",
            )
            .plot_cluster_labels()
        )
        plotter.show()
        fig = plotter._fig
        ax_matrix = fig.axes[0]

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        h, w, _ = buf.shape
        mid_row = int((1.0 - (ax_matrix.get_position().y0 + ax_matrix.get_position().y1) / 2.0) * h)
        # The spine is centered on the matrix's right edge, so its outward half
        # (from the boundary to half the linewidth past it) must render solid;
        # a painted-over or clipped border shows up as a white gap in this span.
        matrix_right_fig_x = ax_matrix.get_position().x1
        outward_fig_x = matrix_right_fig_x + (outer_lw / 2.0) / (72.0 * fig.get_figwidth())
        x0_px = int(matrix_right_fig_x * w)
        x1_px = int(outward_fig_x * w)

        span = buf[mid_row, x0_px:x1_px, :3]
        assert span.size > 0
        assert (span.max(axis=1) < 50).all()
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
    results = Results(pd.DataFrame(), matrix=toy_matrix, clusters=clusters)

    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        with pytest.raises(ValueError):
            Plotter(results).plot_matrix().show()
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_bar_renders_standalone(toy_results):
    """
    Ensures plot_cluster_bar renders without plot_cluster_labels or
    plot_cluster_labels_compact, using the tracks-only fallback.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        Plotter(toy_results).plot_cluster_bar(name="sig").plot_bar_labels().show()
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_bar_standalone_without_matrix_hides_placeholder_chrome(toy_results):
    """
    Ensures a matrix-less standalone cluster-bar + colorbar chain hides the whole
    placeholder axes instead of leaving a visible empty matrix box, and draws no
    matrix image.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = (
            Plotter(toy_results)
            .plot_cluster_bar(name="sig")
            .plot_bar_labels()
            .add_colorbar(name="matrix", cmap="RdBu_r", norm=Normalize(-1, 1))
            .plot_colorbars()
        )
        plotter.show()
        ax0 = plotter._fig.axes[0]
        assert ax0.get_visible() is False
        assert all(len(ax.images) == 0 for ax in plotter._fig.axes)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_bar_uses_internal_cluster_labels(toy_results):
    """
    Ensures plot_cluster_bar renders from internally generated cluster labels.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
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
def test_plotter_plot_handle_rebuilds_closed_figure(toy_results, tmp_path):
    """
    Ensures Plotter rebuilds after the backing figure is closed.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
        tmp_path (Path): Temporary output directory.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    out = tmp_path / "plotter_rebuild.png"
    plotter = None
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = Plotter(toy_results).plot_matrix()

        # Baseline render: capture the original figure handle.
        plotter.show()
        fig0 = plotter._fig
        # Simulate notebook/backend lifecycle where figures are explicitly closed.
        plt.close(fig0)

        # Closed handle should trigger a rebuild on show().
        plotter.show()
        assert plotter._fig is not None
        assert plotter._fig is not fig0
        assert plotter._fig.number in plt.get_fignums()

        # save() should also rebuild after close and produce an output file.
        plt.close(plotter._fig)
        plotter.save(out, dpi=150)
        assert out.exists()
        assert out.stat().st_size > 0
        assert plotter._fig.number in plt.get_fignums()
    finally:
        plt.show = plt_show
        if (
            plotter is not None
            and plotter._fig is not None
            and plotter._fig.number in plt.get_fignums()
        ):
            plt.close(plotter._fig)


@pytest.mark.api
def test_plot_colorbars_tick_decimals_contract(toy_results):
    """
    Ensures colorbar precision and label spacing contracts are consistent.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        TypeError: If tick_decimals is not an integer or label_pad is invalid.
        ValueError: If tick_decimals is negative or label_pad is negative.
    """
    with pytest.raises(TypeError, match="tick_decimals"):
        Plotter(toy_results).plot_colorbars(tick_decimals=1.5)
    with pytest.raises(TypeError, match="tick_decimals"):
        Plotter(toy_results).plot_colorbars(tick_decimals=True)
    with pytest.raises(ValueError, match="tick_decimals"):
        Plotter(toy_results).plot_colorbars(tick_decimals=-1)
    with pytest.raises(TypeError, match="label_pad"):
        Plotter(toy_results).plot_colorbars(label_pad="2")
    with pytest.raises(TypeError, match="label_pad"):
        Plotter(toy_results).plot_colorbars(label_pad=True)
    with pytest.raises(ValueError, match="label_pad"):
        Plotter(toy_results).plot_colorbars(label_pad=-0.1)

    # Also test that valid arguments are accepted and reflected in the rendered colorbar.
    plt = use_agg_backend()
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
            .plot_colorbars(tick_decimals=2, label_pad=7.0)
        )
        plotter.show()
        fig = plotter._fig
        fig.canvas.draw()
        tick_texts = [t.get_text() for ax in fig.axes for t in ax.get_xticklabels()]
        tick_texts = [txt for txt in tick_texts if txt]
        assert "0.12" in tick_texts
        assert "0.1234" not in tick_texts
        assert all("." not in txt or len(txt.split(".", 1)[1]) <= 2 for txt in tick_texts)
        precision_axes = [ax for ax in fig.axes if ax.get_xlabel() == "Precision"]
        assert len(precision_axes) == 1
        assert precision_axes[0].xaxis.labelpad == pytest.approx(7.0)
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
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        with pytest.raises(TypeError):
            (
                Plotter(toy_results)
                .plot_matrix()
                .plot_label_bar(values={"a": "hit"}, mode="categorical")
                .show()
            )
    finally:
        plt.show = plt_show


@pytest.mark.api
@pytest.mark.parametrize(
    "kwargs, add_once_first, expected_match",
    [
        ({"name": ""}, False, "non-empty string"),
        ({"name": "orf_category"}, True, "already declared"),
        ({"name": "orf_category", "nrows": 0}, False, "nrows"),
        ({"name": "orf_category", "ncols": -1}, False, "ncols"),
        ({"name": "orf_category", "row_pad": -0.01}, False, "row_pad"),
        ({"name": "orf_category", "col_pad": -0.01}, False, "col_pad"),
    ],
    ids=[
        "empty_name",
        "duplicate_name",
        "nrows_nonpositive",
        "ncols_nonpositive",
        "row_pad_negative",
        "col_pad_negative",
    ],
)
def test_add_label_legend_rejects_invalid_inputs(
    toy_results,
    kwargs,
    add_once_first,
    expected_match,
):
    """
    Ensures add_label_legend validates name uniqueness and geometry arguments.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
        kwargs (dict): Keyword arguments for add_label_legend.
        add_once_first (bool): Whether to add a valid legend before asserting duplicate rejection.
        expected_match (str): Regex fragment expected in the raised error message.

    Raises:
        ValueError: If legend declaration arguments are invalid.
    """
    plotter = (
        Plotter(toy_results)
        .plot_matrix()
        .plot_label_bar(
            values={"a": "1", "b": "2", "c": "1", "d": "2"},
            name="orf_category",
            mode="categorical",
            colors={"1": "#1b9e77", "2": "#d95f02"},
        )
    )
    if add_once_first:
        plotter.add_label_legend(name="orf_category")

    with pytest.raises(ValueError, match=expected_match):
        plotter.add_label_legend(**kwargs)


@pytest.mark.api
@pytest.mark.parametrize(
    "kwargs, expected_match",
    [
        ({"height": 0.0}, "height"),
        ({"gap": -0.01}, "gap"),
        ({"vpad": -0.01}, "vpad"),
        ({"title_pad": -1.0}, "title_pad"),
        ({"swatch_scale": 0.0}, "swatch_scale"),
    ],
    ids=[
        "height_nonpositive",
        "gap_negative",
        "vpad_negative",
        "title_pad_negative",
        "swatch_scale_nonpositive",
    ],
)
def test_plot_label_legends_rejects_invalid_geometry(
    toy_results,
    kwargs,
    expected_match,
):
    """
    Ensures plot_label_legends validates strip geometry arguments.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
        kwargs (dict): Keyword arguments for plot_label_legends.
        expected_match (str): Regex fragment expected in the raised error message.

    Raises:
        ValueError: If legend strip geometry arguments are invalid.
    """
    with pytest.raises(ValueError, match=expected_match):
        Plotter(toy_results).plot_label_legends(**kwargs)


@pytest.mark.api
def test_plot_label_legends_require_source_colors_mapping(toy_results):
    """
    Ensures label legends fail when the referenced row bar payload lacks colors mapping.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        ValueError: If the referenced categorical row label bar has no colors mapping.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_label_bar(
                values={"a": "1", "b": "2", "c": "1", "d": "2"},
                name="orf_category",
                mode="categorical",
                colors={"1": "#1b9e77", "2": "#d95f02"},
            )
            .add_label_legend(name="orf_category")
            .plot_label_legends()
        )
        # Corrupt track payload intentionally to verify legend source integrity checks.
        # Replace the row-track renderer to avoid earlier label-bar color validation.
        plotter._track_layout.tracks[0]["renderer"] = lambda *args, **kwargs: None
        plotter._track_layout.tracks[0]["payload"]["colors"] = None
        with pytest.raises(ValueError, match="non-empty `colors` mapping"):
            plotter.show()
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_label_legends_grid_overflow_raises(toy_results):
    """
    Ensures legend rendering errors when configured grid cannot fit all categories.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        ValueError: If nrows * ncols is smaller than the number of legend entries.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        with pytest.raises(ValueError, match="cannot fit all categories"):
            (
                Plotter(toy_results)
                .plot_matrix()
                .plot_label_bar(
                    values={"a": "1", "b": "2", "c": "3", "d": "1"},
                    name="orf_category",
                    mode="categorical",
                    colors={"1": "#1b9e77", "2": "#d95f02", "3": "#7570b3"},
                )
                .add_label_legend(name="orf_category", nrows=1, ncols=2)
                .plot_label_legends()
                .show()
            )
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_label_legends_render_categorical_track(toy_results):
    """
    Ensures categorical label legends render from a declared categorical row label bar.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        values = {"a": "1", "b": "2", "c": "1", "d": "2"}
        colors = {
            "1": "#1b9e77",
            "2": "#d95f02",
            "3": "#7570b3",  # Unused in matrix rows
        }
        plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_label_bar(
                values=values,
                name="orf_category",
                mode="categorical",
                colors=colors,
                title="ORF Category",
            )
            .add_label_legend(
                name="orf_category",
                nrows=1,
                row_pad=0.01,
                col_pad=0.015,
            )
            .plot_label_legends(height=0.07, gap=0.02, swatch_scale=0.9)
        )
        plotter.show()
        texts = extract_figure_text(plotter._fig, strip=True, nonempty=True)
        assert "ORF Category" in texts
        assert "1" in texts
        assert "2" in texts
        assert "3" not in texts
        assert plotter.label_legend_layout_ is not None
        assert plotter._label_legends[0]["row_pad"] == pytest.approx(0.01)
        assert plotter._label_legends[0]["col_pad"] == pytest.approx(0.015)
        assert plotter.label_legend_layout_["swatch_scale"] == pytest.approx(0.9)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_label_legends_require_categorical_track_mode(toy_results):
    """
    Ensures categorical legends reject non-categorical row label bars.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        ValueError: If the referenced label bar is not categorical.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        with pytest.raises(ValueError, match="mode='categorical'"):
            (
                Plotter(toy_results)
                .plot_matrix()
                .plot_label_bar(
                    values={"a": 0.0, "b": 1.0, "c": 0.5, "d": 1.0},
                    name="continuous_track",
                    mode="continuous",
                    cmap="viridis",
                )
                .add_label_legend(name="continuous_track")
                .plot_label_legends()
                .show()
            )
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_label_legends_require_known_row_track(toy_results):
    """
    Ensures categorical legends reject unknown row label-bar references.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        ValueError: If the referenced label-bar name is unknown.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        with pytest.raises(ValueError, match="Unknown label-legend name"):
            (
                Plotter(toy_results)
                .plot_matrix()
                .add_label_legend(name="missing_track")
                .plot_label_legends()
                .show()
            )
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_respect_np_order(toy_results):
    """
    Ensures cluster labels keep label first while respecting q/fe/p/n order from label_fields
    and emitting a single stats block.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        results_q = toy_results.with_qvalues()
        plotter = (
            Plotter(results_q)
            .plot_matrix()
            .plot_cluster_labels(
                label_fields=("label", "q", "fe", "p", "n"),
            )
        )
        plotter.show()
        fig = plotter._fig
        texts = extract_figure_text(fig)
        # Cluster labels are asserted via rendered text fragments rather than data objects.
        label_texts = [t for t in texts if "$q$=" in t and "FE=" in t and "$p$=" in t and "n=" in t]
        assert label_texts, "Expected cluster label text to be rendered."
        for txt in label_texts:
            assert " (" in txt
            assert txt.strip().endswith(")")
            assert "$q$=" in txt
            assert "FE=" in txt
            assert "$p$=" in txt
            assert "n=" in txt
            assert txt.find("$q$=") < txt.find("FE=")
            assert txt.find("FE=") < txt.find("$p$=")
            assert txt.find("$p$=") < txt.find("n=")
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_invalid_label_prefix_raises(toy_results):
    """
    Ensures unsupported label_prefix values raise a ValueError.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        ValueError: If label_prefix is not supported.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        with pytest.raises(ValueError, match="label_prefix"):
            (
                Plotter(toy_results)
                .plot_matrix()
                .plot_cluster_labels(
                    label_fields=("label",),
                    label_prefix="bad",
                )
                .show()
            )
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_none_label_fields_hides_text(toy_results):
    """
    Ensures label_fields=None suppresses cluster text when no prefix is requested.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_cluster_labels(
                label_fields=None,
            )
        )
        plotter.show()
        texts = extract_figure_text(plotter._fig, strip=True, nonempty=True)
        assert not texts
    finally:
        plt.show = plt_show


@pytest.mark.api
@pytest.mark.parametrize(
    "label_prefix",
    ["cid", "alpha"],
    ids=["with_cid_prefix", "with_alpha_prefix"],
)
def test_plot_cluster_labels_label_prefix_supports_compressed_labels(
    toy_results,
    label_prefix,
):
    """
    Ensures supported label_prefix modes prefix compressed display labels.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_cluster_labels(
                label_mode="compressed",
                label_fields=("label", "fe"),
                label_prefix=label_prefix,
                wrap_text=False,
            )
        )
        plotter.show()
        texts = extract_figure_text(plotter._fig, strip=True, nonempty=True)
        tokens = [txt.split(". ", 1)[0] for txt in texts if ". " in txt]
        if label_prefix == "cid":
            assert any(tok.isdigit() for tok in tokens)
        else:
            assert any(tok.isalpha() and tok.isupper() for tok in tokens)
        assert any("FE=" in txt for txt in texts)
    finally:
        plt.show = plt_show


@pytest.mark.api
@pytest.mark.parametrize(
    "label_fields",
    [("label",), ("n", "p"), None],
    ids=["with_label_fields", "np_only", "without_label_fields"],
)
@pytest.mark.parametrize(
    "label_prefix",
    [None, "cid", "alpha"],
    ids=["without_prefix", "with_cid_prefix", "with_alpha_prefix"],
)
def test_plot_cluster_labels_label_prefix_precedence_override_wins(
    toy_results,
    label_fields,
    label_prefix,
):
    """
    Ensures explicit overrides win with and without cid prefix mode.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        spans = toy_results.cluster_layout().cluster_spans
        assert len(spans) >= 2
        override_cid = int(spans[0][0])
        regular_cid = int(spans[1][0])
        override_prefix = format_label_prefix(label_prefix, override_cid)
        override_label = (
            f"{override_prefix} DNA replication & repair"
            if override_prefix
            else "DNA replication & repair"
        )
        plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_cluster_labels(
                label_fields=label_fields,
                label_prefix=label_prefix,
                overrides={override_cid: override_label},
                wrap_text=False,
            )
        )
        plotter.show()
        texts = extract_figure_text(plotter._fig, strip=True, nonempty=True)
        if label_fields is None:
            assert override_label in texts
        else:
            assert any(txt.startswith(override_label) for txt in texts)
        regular_prefix = format_label_prefix(label_prefix, regular_cid)
        if regular_prefix:
            assert all(txt != f"{regular_prefix} {override_label}" for txt in texts)
            if label_fields is None:
                assert regular_prefix in texts
            else:
                assert any(txt.startswith(f"{regular_prefix} ") for txt in texts)
        else:
            assert all(not txt.startswith(f"{regular_cid}. ") for txt in texts)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_accepts_override_mapper(toy_results):
    """
    Ensures cluster label overrides can replace generated labels by cluster id.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        ValueError: If overrides reference unknown cluster ids.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        # Override one cluster label and verify both explicit and empty-string overrides render safely.
        cid = int(toy_results.cluster_layout().cluster_spans[0][0])
        custom_label = f"Custom-{cid}"
        plotter = (
            Plotter(toy_results)
            .set_label_panel(
                axes=[0.70, 0.05, 0.29, 0.90],
                text_pad=0.01,
            )
            .plot_matrix()
            .plot_cluster_labels(
                overrides={cid: custom_label},
                label_fields=("label",),
            )
        )
        plotter.show()
        fig = plotter._fig
        texts = extract_figure_text(fig)
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

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            Plotter(toy_results).plot_cluster_labels(typo_key=True)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_placeholder_style_overrides_global_style(toy_results):
    """
    Ensures placeholder labels use placeholder_color/placeholder_alpha over global color/alpha.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        first_cluster = int(toy_results.cluster_layout().cluster_spans[0][0])
        # Keep layout unchanged but drop one cluster from labels to force placeholder rendering.
        filtered = toy_results.filter(f"cluster == {first_cluster}")
        placeholder_text = "Nonsignificant"
        plotter = (
            Plotter(filtered)
            .plot_matrix()
            .plot_cluster_labels(
                label_fields=("label",),
                skip_unlabeled=False,
                color="gray",
                alpha=0.2,
                placeholder_text=placeholder_text,
                placeholder_color="red",
                placeholder_alpha=1.0,
            )
            .plot_cluster_bar(name="sig")
        )
        plotter.show()

        # Collect rendered labels to validate placeholder-vs-regular style behavior.
        fig = plotter._fig
        texts = [t for ax in fig.axes for t in ax.texts if t.get_text().strip()]
        placeholder_nodes = [t for t in texts if t.get_text() == placeholder_text]
        regular_nodes = [t for t in texts if t.get_text() != placeholder_text]
        assert placeholder_nodes, "Expected placeholder text to be rendered."
        assert regular_nodes, "Expected at least one non-placeholder cluster label."

        # Placeholder text uses placeholder_* style, while labeled clusters keep global color/alpha.
        placeholder_node = placeholder_nodes[0]
        regular_node = regular_nodes[0]
        assert to_rgba(placeholder_node.get_color()) == pytest.approx(to_rgba("red"))
        assert float(placeholder_node.get_alpha()) == pytest.approx(1.0)
        assert to_rgba(regular_node.get_color()) == pytest.approx(to_rgba("gray"))
        assert float(regular_node.get_alpha()) == pytest.approx(0.2)

        # Cluster bars should render for both clusters; unlabeled clusters use min cmap color.
        cmap = plt.get_cmap(plotter._style["sigbar_cmap"])
        min_rgb = to_rgba(cmap(0.0))[:3]
        max_rgb = to_rgba(cmap(1.0))[:3]
        bar_patches = [p for ax in fig.axes for p in ax.patches if p.get_alpha() is not None]
        assert len(bar_patches) == 2, "Expected one sigbar patch per cluster span."
        bar_colors = [to_rgba(p.get_facecolor())[:3] for p in bar_patches]
        assert any(c == pytest.approx(min_rgb) for c in bar_colors)
        assert any(c == pytest.approx(max_rgb) for c in bar_colors)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_override_with_np_fields_keeps_label_and_stats(toy_results):
    """
    Ensures override labels remain visible when label_fields excludes 'label', with stable stats order.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        results_q = toy_results.with_qvalues()
        cid = int(toy_results.cluster_layout().cluster_spans[0][0])
        plotter = (
            Plotter(results_q)
            .plot_matrix()
            .plot_cluster_labels(
                overrides={cid: "CustomLabelVisibleWithFields"},
                label_fields=("n", "q", "fe", "p"),
                wrap_text=False,
            )
        )
        plotter.show()
        fig = plotter._fig
        texts = extract_figure_text(fig)
        custom_texts = [t for t in texts if t.startswith("CustomLabelVisibleWithFields")]
        assert custom_texts, "Expected override label text to remain visible."
        for txt in custom_texts:
            assert "n=" in txt
            assert "$q$=" in txt
            assert "FE=" in txt
            assert "$p$=" in txt
            assert txt.find("n=") < txt.find("$q$=")
            assert txt.find("$q$=") < txt.find("FE=")
            assert txt.find("FE=") < txt.find("$p$=")
    finally:
        plt.show = plt_show


@pytest.mark.unit
def test_plot_cluster_label_override_keeps_deterministic_stats(toy_results):
    """
    Ensures label-only overrides preserve deterministic cluster statistics.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    # Build baseline labels and assert a text-only override leaves numeric stats unchanged.
    df = toy_results.cluster_labels()
    cid = int(df.loc[0, "cluster"])
    base_pval = float(df.loc[df["cluster"] == cid, "pval"].iloc[0])
    base_fe = float(df.loc[df["cluster"] == cid, "fe"].iloc[0])
    override_map = _parse_label_overrides({cid: f"Custom-{cid}"})
    label_map = _build_label_map(df, override_map)

    assert label_map[cid][0] == f"Custom-{cid}"
    assert float(label_map[cid][1]) == pytest.approx(base_pval)
    assert float(label_map[cid][3]) == pytest.approx(base_pval)
    assert float(label_map[cid][4]) == pytest.approx(base_fe)


@pytest.mark.api
def test_plot_cluster_labels_max_words_controls_label_building_compressed(toy_results):
    """
    Ensures max_words controls compressed label building.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        # Render compressed labels with max_words=1 and assert generated labels stay single-word.
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
        texts = extract_figure_text(fig, strip=True)
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
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        # Apply max_words truncation to an override label and verify rendered text is shortened.
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
        texts = extract_figure_text(fig)
        assert any(t.strip() == "Alpha" for t in texts)
        assert not any("Alpha Beta" in t for t in texts)
    finally:
        plt.show = plt_show


def _separator_lines(ax_lab):
    """
    Filters an axes' line artists down to horizontal inter-cluster separator lines.

    Args:
        ax_lab (matplotlib.axes.Axes): Label panel axes to inspect.

    Returns:
        list: Line2D artists whose endpoints share a y-value and differ in x.
    """
    return [
        ln
        for ln in ax_lab.lines
        if ln.get_xdata()[0] != ln.get_xdata()[1] and ln.get_ydata()[0] == ln.get_ydata()[1]
    ]


@pytest.mark.api
def test_plot_cluster_labels_label_sep_xmax_extends_past_one(toy_results):
    """
    Ensures label_sep_xmax > 1.0 is not clamped and reaches the drawn separator line.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_cluster_labels(label_sep_xmax=1.4)
        )
        plotter.show()
        ax_lab = plotter._fig.axes[-1]
        seps = _separator_lines(ax_lab)
        assert seps, "Expected at least one separator line to be drawn."
        assert all(max(ln.get_xdata()) == pytest.approx(1.4) for ln in seps)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_label_sep_lines_not_clipped(toy_results):
    """
    Ensures separator line artists are drawn with clip_on=False so extended spans
    beyond the label panel remain visible.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = Plotter(toy_results).plot_matrix().plot_cluster_labels(label_sep_xmax=1.4)
        plotter.show()
        ax_lab = plotter._fig.axes[-1]
        seps = _separator_lines(ax_lab)
        assert seps, "Expected at least one separator line to be drawn."
        assert all(ln.get_clip_on() is False for ln in seps)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_label_sep_xmax_default_unchanged(toy_results):
    """
    Ensures omitting label_sep_xmax still resolves separator lines to end at 1.0.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = Plotter(toy_results).plot_matrix().plot_cluster_labels()
        plotter.show()
        ax_lab = plotter._fig.axes[-1]
        seps = _separator_lines(ax_lab)
        assert seps, "Expected at least one separator line to be drawn."
        assert all(max(ln.get_xdata()) == pytest.approx(1.0) for ln in seps)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_label_sep_xmin_xmax_swap_outside_unit_range(toy_results):
    """
    Ensures the xmin/xmax swap safety check still applies when values fall outside
    [0, 1], now that they are no longer clamped before comparison.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_cluster_labels(label_sep_xmin=1.5, label_sep_xmax=-0.5)
        )
        plotter.show()
        ax_lab = plotter._fig.axes[-1]
        seps = _separator_lines(ax_lab)
        assert seps, "Expected at least one separator line to be drawn."
        for ln in seps:
            xdata = ln.get_xdata()
            assert min(xdata) == pytest.approx(-0.5)
            assert max(xdata) == pytest.approx(1.5)
    finally:
        plt.show = plt_show


@pytest.mark.unit
def test_style_config_as_dict_merges_defaults_and_overrides():
    """
    Ensures StyleConfig.as_dict() returns merged defaults and independent overrides.
    """
    style = StyleConfig()
    style.set("boundary_color", "red")

    values = style.as_dict()

    assert values["boundary_color"] == "red"
    assert values["boundary_lw"] == DEFAULT_STYLE["boundary_lw"]

    values["boundary_color"] = "blue"
    assert style["boundary_color"] == "red"


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


def _cut_line_artists(plotter):
    """
    Returns the cut line and scissors artists from a rendered Plotter.

    Args:
        plotter (Plotter): Plotter whose figure has been rendered.

    Returns:
        tuple[list, list]: Vertical dashed Line2D artists and AnnotationBbox artists.
    """
    lines, icons = [], []
    for ax in plotter._fig.axes:
        lines += [ln for ln in ax.lines if ln.get_linestyle() == "--"]
        icons += [a for a in ax.artists if isinstance(a, AnnotationBbox)]
    return lines, icons


@pytest.mark.api
def test_plot_cut_line_defaults_to_cluster_threshold(toy_results):
    """
    Ensures the cut line sits at the clustering threshold and draws scissors by default.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    use_agg_backend()
    plotter = Plotter(toy_results).plot_dendrogram().plot_cut_line().plot_matrix()
    plotter._render()
    lines, icons = _cut_line_artists(plotter)
    assert len(lines) == 1
    assert list(lines[0].get_xdata()) == [toy_results.clusters.threshold] * 2
    assert len(icons) == 1


@pytest.mark.api
def test_plot_cut_line_honors_style_options(toy_results):
    """
    Ensures threshold, color, linestyle, linewidth, and scissors options are applied.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    use_agg_backend()
    plotter = (
        Plotter(toy_results)
        .plot_dendrogram()
        .plot_cut_line(threshold=1.5, color="blue", linestyle="--", linewidth=2.0, scissors=False)
        .plot_matrix()
    )
    plotter._render()
    lines, icons = _cut_line_artists(plotter)
    assert list(lines[0].get_xdata()) == [1.5, 1.5]
    assert lines[0].get_color() == "blue"
    assert lines[0].get_linewidth() == 2.0
    assert icons == []


@pytest.mark.api
def test_plot_cut_line_scissors_follow_color_and_size(toy_results):
    """
    Ensures the scissors use the line color and scale with `scissors_size`.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    use_agg_backend()
    widths = {}
    for size in (8.0, 24.0):
        plotter = (
            Plotter(toy_results)
            .plot_dendrogram()
            .plot_cut_line(color="blue", scissors_size=size)
            .plot_matrix()
        )
        plotter._render()
        plotter._fig.canvas.draw()
        _lines, icons = _cut_line_artists(plotter)
        widths[size] = (
            icons[0].offsetbox.get_window_extent(plotter._fig.canvas.get_renderer()).width
        )
        assert {c.get_edgecolor()[:3] for c in icons[0].offsetbox.get_children()[:2]} == {
            (0.0, 0.0, 1.0)
        }
    assert widths[24.0] > 2.5 * widths[8.0]


@pytest.mark.api
def test_plot_cut_line_requires_dendrogram_first(toy_results):
    """
    Ensures plot_cut_line() raises unless plot_dendrogram() was declared before it.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        ValueError: If no dendrogram layer precedes the cut line.
    """
    with pytest.raises(ValueError):
        Plotter(toy_results).plot_cut_line()
    with pytest.raises(ValueError):
        Plotter(toy_results).plot_cut_line().plot_dendrogram()
    with pytest.raises(ValueError):
        Plotter(toy_results).plot_dendrogram().plot_cut_line(scissors_size=0)


def _col_tree_results(n_cols=6):
    """
    Builds finalized Results whose layout carries a column linkage.

    Args:
        n_cols (int): Number of matrix columns. Defaults to 6.

    Returns:
        Results: Results from finalize(col_cluster=True, col_dendrogram=True).
    """
    rng = np.random.default_rng(3)
    labels = list("abcdefghij")
    matrix = Matrix(pd.DataFrame(rng.normal(size=(10, n_cols)), index=labels))
    annotations = Annotations({"t1": labels[:5], "t2": labels[5:]}, matrix)
    analysis = Analysis(matrix, annotations).cluster(linkage_threshold=3.0).enrich()
    return analysis.finalize(col_cluster=True, col_dendrogram=True).results


@pytest.mark.api
def test_plot_col_dendrogram_aligns_leaves_with_matrix_columns():
    """
    Ensures the column dendrogram draws one branch per merge, with its leaves centered over the
    matrix columns and the tree's leaf order equal to the layout's column order.
    """
    use_agg_backend()
    results = _col_tree_results(n_cols=6)
    layout = results.cluster_layout()
    plotter = Plotter(results).plot_col_dendrogram().plot_matrix()
    plotter._render()

    tree_ax = next(ax for ax in plotter._fig.axes if ax.lines)
    assert len(tree_ax.lines) == 5
    leaf_x = sorted(
        x for line in tree_ax.lines for x, y in zip(line.get_xdata(), line.get_ydata()) if y == 0
    )
    assert leaf_x == pytest.approx(list(range(6)))
    leaves = dendrogram(layout.col_linkage, no_plot=True)["leaves"]
    assert leaves == layout.col_order.tolist()


@pytest.mark.api
def test_plot_col_dendrogram_honors_style_options_and_defaults():
    """
    Ensures color, linewidth, and axes position override the style defaults, which otherwise
    match the row dendrogram color and width.
    """
    use_agg_backend()
    results = _col_tree_results()
    default = Plotter(results).plot_col_dendrogram().plot_matrix()
    default._render()
    default_ax = next(ax for ax in default._fig.axes if ax.lines)
    assert default_ax.lines[0].get_color() == DEFAULT_STYLE["dendro_color"]
    assert default_ax.lines[0].get_linewidth() == DEFAULT_STYLE["dendro_lw"]
    matrix = default._fig.axes[0].get_position()
    assert list(default_ax.get_position().bounds) == pytest.approx(
        [matrix.x0, matrix.y1, matrix.width, DEFAULT_STYLE["col_dendro_height"]]
    )

    custom = (
        Plotter(results)
        .plot_col_dendrogram(axes=[0.2, 0.9, 0.4, 0.08], color="red", linewidth=2.5)
        .plot_matrix()
    )
    custom._render()
    custom_ax = next(ax for ax in custom._fig.axes if ax.lines)
    assert custom_ax.lines[0].get_color() == "red"
    assert custom_ax.lines[0].get_linewidth() == 2.5
    assert list(custom_ax.get_position().bounds) == pytest.approx([0.2, 0.9, 0.4, 0.08])

    taller = Plotter(results).plot_col_dendrogram(height=0.07).plot_matrix()
    taller._render()
    taller_ax = next(ax for ax in taller._fig.axes if ax.lines)
    assert taller_ax.get_position().height == pytest.approx(0.07)
    assert taller_ax.get_position().y0 == pytest.approx(default_ax.get_position().y0)
    with pytest.raises(ValueError):
        Plotter(results).plot_col_dendrogram(height=0)


@pytest.mark.api
def test_plot_col_dendrogram_default_boxes_align_with_matrix():
    """
    Ensures the default column dendrogram spans the matrix width above it, and that the row
    dendrogram keeps spanning exactly the matrix height.
    """
    use_agg_backend()
    plotter = Plotter(_col_tree_results()).plot_dendrogram().plot_col_dendrogram().plot_matrix()
    plotter._render()

    matrix_ax = plotter._fig.axes[0]
    row_tree, col_tree = sorted(
        (ax for ax in plotter._fig.axes if ax.lines), key=lambda ax: ax.get_position().x0
    )
    matrix = matrix_ax.get_position()
    assert (col_tree.get_position().x0, col_tree.get_position().width) == pytest.approx(
        (matrix.x0, matrix.width)
    )
    assert col_tree.get_position().y0 >= matrix.y1 - 1e-9
    assert (row_tree.get_position().y0, row_tree.get_position().height) == pytest.approx(
        (matrix.y0, matrix.height)
    )


@pytest.mark.api
def test_plot_col_dendrogram_requires_column_linkage(toy_results):
    """
    Ensures plot_col_dendrogram() raises when results were not finalized with col_dendrogram=True.

    Args:
        toy_results (Results): Results fixture without a column linkage.

    Raises:
        ValueError: If the layout has no column linkage.
    """
    with pytest.raises(ValueError):
        Plotter(toy_results).plot_col_dendrogram()


@pytest.mark.api
@pytest.mark.parametrize("extension", ["png", "svg", "pdf"])
def test_plot_cut_line_and_col_dendrogram_save_to_common_formats(tmp_path, extension):
    """
    Ensures a figure with the cut line, scissors, and column dendrogram saves to raster and
    vector formats.

    Args:
        tmp_path (Path): Temporary directory fixture.
        extension (str): Output file extension.
    """
    use_agg_backend()
    path = tmp_path / f"figure.{extension}"
    (
        Plotter(_col_tree_results())
        .plot_dendrogram()
        .plot_cut_line()
        .plot_col_dendrogram()
        .plot_matrix()
        .save(path)
    )
    assert path.stat().st_size > 0


@pytest.mark.api
def test_plot_col_dendrogram_single_column_has_no_linkage():
    """
    Ensures a single-column matrix finalizes without a column linkage, and that plotting its
    column dendrogram raises.

    Raises:
        ValueError: If the layout has no column linkage.
    """
    labels = list("abcdef")
    matrix = Matrix(pd.DataFrame({"x": np.arange(6.0)}, index=labels))
    annotations = Annotations({"t1": labels[:3], "t2": labels[3:]}, matrix)
    results = (
        Analysis(matrix, annotations)
        .cluster(linkage_threshold=0.5)
        .enrich()
        .finalize(col_cluster=True, col_dendrogram=True)
        .results
    )

    layout = results.cluster_layout()
    assert layout.col_linkage is None
    assert layout.col_order.tolist() == [0]
    with pytest.raises(ValueError):
        Plotter(results).plot_col_dendrogram()


def _col_bar_axes(plotter, n_cols):
    """
    Returns the column label bar axes of a rendered Plotter, lowest first.

    Args:
        plotter (Plotter): Plotter whose figure has been rendered.
        n_cols (int): Number of matrix columns (one patch per column in a bar).

    Returns:
        list[plt.Axes]: Axes holding one rectangle per column, ordered bottom to top.
    """
    bars = [ax for ax in plotter._fig.axes if len(ax.patches) == n_cols]
    return sorted(bars, key=lambda ax: ax.get_position().y0)


@pytest.mark.api
@pytest.mark.parametrize("mode", ["categorical", "continuous"])
def test_plot_col_label_bar_colors_follow_displayed_column_order(mode):
    """
    Ensures each column's cell has the color of its own value in the displayed (clustered)
    column order, with missing values in the missing color.

    Args:
        mode (str): Bar mode under test.
    """
    use_agg_backend()
    results = _col_tree_results(n_cols=6)
    col_order = results.cluster_layout().col_order
    labels = list(results.matrix.df.columns)
    missing = labels[2]
    if mode == "categorical":
        values = {label: ("hi" if i % 2 else "lo") for i, label in enumerate(labels)}
        kwargs = {"colors": {"hi": "#ff0000", "lo": "#0000ff"}}
        expected = {
            label: to_rgba({"hi": "#ff0000", "lo": "#0000ff"}[values[label]]) for label in values
        }
    else:
        values = {label: float(i) for i, label in enumerate(labels)}
        kwargs = {"mode": "continuous", "cmap": "viridis", "vmin": 0.0, "vmax": 5.0}
        cmap = plt.get_cmap("viridis")
        expected = {label: cmap(values[label] / 5.0) for label in values}
    del values[missing]
    expected[missing] = to_rgba("#eeeeee")

    plotter = Plotter(results).plot_col_label_bar(values, **kwargs).plot_matrix()
    plotter._render()

    (bar,) = _col_bar_axes(plotter, 6)
    cells = sorted(bar.patches, key=lambda patch: patch.get_x())
    assert [patch.get_facecolor() for patch in cells] == [
        pytest.approx(expected[labels[i]]) for i in col_order
    ]


@pytest.mark.api
def test_plot_col_label_bars_stack_above_matrix_below_the_column_dendrogram():
    """
    Ensures column bars span the matrix columns, stack upward in declaration order starting at the
    matrix top, and that the column dendrogram sits above them.
    """
    use_agg_backend()
    results = _col_tree_results(n_cols=6)
    labels = list(results.matrix.df.columns)
    colors = {"a": "red", "b": "blue"}
    values = {label: "a" for label in labels}
    plotter = (
        Plotter(results)
        .plot_col_dendrogram()
        .plot_col_label_bar(values, name="first", colors=colors, height=0.02, title="Group")
        .plot_col_label_bar(values, name="second", colors=colors, height=0.03)
        .plot_bar_labels()
        .plot_matrix()
    )
    plotter._render()

    matrix = plotter._fig.axes[0].get_position()
    first, second = (ax.get_position() for ax in _col_bar_axes(plotter, 6))
    tree = next(ax for ax in plotter._fig.axes if ax.lines).get_position()
    for box in (first, second, tree):
        assert (box.x0, box.width) == pytest.approx((matrix.x0, matrix.width))
    assert first.y0 == pytest.approx(matrix.y1)
    assert (first.height, second.height) == pytest.approx((0.02, 0.03))
    assert second.y0 == pytest.approx(first.y1)
    assert tree.y0 == pytest.approx(second.y1)
    assert "Group" in extract_figure_text(plotter._fig)


@pytest.mark.api
@pytest.mark.parametrize(
    "kwargs, x, direction, alignment",
    [
        ({}, 1.0, 1, "left"),
        ({"title_position": "right"}, 1.0, 1, "left"),
        ({"title_position": "left"}, 0.0, -1, "right"),
    ],
)
def test_plot_col_label_bar_title_sits_on_the_requested_side(kwargs, x, direction, alignment):
    """
    Ensures the strip title is anchored at the right edge of the strip by default and at the left
    edge on request, offset and aligned away from the strip.

    Args:
        kwargs (dict): Title position keyword arguments.
        x (float): Expected anchor x-position in strip axes fraction.
        direction (int): Expected sign of the horizontal offset (points).
        alignment (str): Expected horizontal alignment of the title.
    """
    use_agg_backend()
    results = _col_tree_results(n_cols=6)
    values = {label: "a" for label in results.matrix.df.columns}
    plotter = (
        Plotter(results)
        .plot_col_label_bar(values, colors={"a": "red"}, title="Group", **kwargs)
        .plot_bar_labels()
        .plot_matrix()
    )
    plotter._render()

    (bar,) = _col_bar_axes(plotter, 6)
    (title,) = bar.texts
    assert title.get_text() == "Group"
    assert title.xy == (x, 0.5)
    assert title.xyann[0] * direction > 0
    assert title.get_horizontalalignment() == alignment


@pytest.mark.api
def test_plot_col_label_bar_titles_follow_plot_bar_labels():
    """
    Ensures column bar titles appear only when plot_bar_labels() is declared (as for row bars),
    and take its font size, color, and offset.
    """
    use_agg_backend()
    results = _col_tree_results(n_cols=6)
    values = {label: "a" for label in results.matrix.df.columns}

    def render(*layers):
        plotter = Plotter(results).plot_col_label_bar(values, colors={"a": "red"}, title="Group")
        for layer in layers:
            layer(plotter)
        plotter.plot_matrix()._render()
        (bar,) = _col_bar_axes(plotter, 6)
        return list(bar.texts)

    assert render() == []
    (title,) = render(lambda p: p.plot_bar_labels(fontsize=13, color="green", pad=5))
    assert (title.get_fontsize(), title.get_color(), title.xyann[0]) == (13, "green", 5)


@pytest.mark.api
def test_plot_col_label_bar_disabled_draws_nothing_and_does_not_reserve_space():
    """
    Ensures a disabled column bar is not drawn, takes no space, and does not clash by name.
    """
    use_agg_backend()
    results = _col_tree_results(n_cols=6)
    values = {label: "a" for label in results.matrix.df.columns}
    plotter = (
        Plotter(results)
        .plot_col_label_bar(values, name="x", colors={"a": "red"}, enabled=False)
        .plot_col_label_bar(values, name="x", colors={"a": "red"})
        .plot_col_dendrogram()
        .plot_matrix()
    )
    plotter._render()

    matrix = plotter._fig.axes[0].get_position()
    (bar,) = _col_bar_axes(plotter, 6)
    tree = next(ax for ax in plotter._fig.axes if ax.lines).get_position()
    assert bar.get_position().y0 == pytest.approx(matrix.y1)
    assert tree.y0 == pytest.approx(bar.get_position().y1)


@pytest.mark.api
def test_add_label_legend_supports_column_label_bars():
    """
    Ensures a legend can explain a column label bar, and that a name shared with a row bar is
    rejected as ambiguous.

    Raises:
        ValueError: If a legend name matches both a row and a column bar.
    """
    use_agg_backend()
    results = _col_tree_results(n_cols=6)
    labels = list(results.matrix.df.columns)
    values = {label: ("hi" if i % 2 else "lo") for i, label in enumerate(labels)}
    colors = {"hi": "#ff0000", "lo": "#0000ff"}

    plotter = (
        Plotter(results)
        .plot_col_label_bar(values, name="level", colors=colors)
        .add_label_legend(name="level", title="Level")
        .plot_label_legends()
        .plot_matrix()
    )
    plotter._render()
    assert {"Level", "hi", "lo"} <= set(extract_figure_text(plotter._fig, strip=True))

    row_values = {label: "hi" for label in results.matrix.df.index}
    clash = (
        Plotter(results)
        .plot_col_label_bar(values, name="level", colors=colors)
        .plot_label_bar(row_values, name="level", colors=colors)
        .add_label_legend(name="level")
        .plot_label_legends()
        .plot_matrix()
    )
    with pytest.raises(ValueError):
        clash._render()


@pytest.mark.api
def test_plot_col_label_bar_rejects_invalid_arguments(toy_results):
    """
    Ensures plot_col_label_bar() rejects a repeated name, a non-positive height, non-dict values,
    and (when rendered) categorical mode without colors.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        ValueError: If the name repeats or the height is not positive.
        TypeError: If values are not a mapping, or categorical mode has no colors.
    """
    use_agg_backend()
    colors = {"a": "red"}
    with pytest.raises(ValueError):
        Plotter(toy_results).plot_col_label_bar({}, colors=colors).plot_col_label_bar(
            {}, colors=colors
        )
    with pytest.raises(ValueError):
        Plotter(toy_results).plot_col_label_bar({}, colors=colors, height=0)
    with pytest.raises(ValueError):
        Plotter(toy_results).plot_col_label_bar({}, colors=colors, title_position="above")
    with pytest.raises(TypeError):
        Plotter(toy_results).plot_col_label_bar(123)
    with pytest.raises(TypeError):
        plotter = Plotter(toy_results).plot_col_label_bar({"f1": "a"}).plot_matrix()
        plotter._render()
