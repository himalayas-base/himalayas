"""
tests/test_compact_labels
~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_rgba

from conftest import extract_figure_text, use_agg_backend
from himalayas.plot import Plotter


@pytest.fixture(autouse=True)
def _close_all_figures_after_each_test():
    """
    Ensures pyplot figures do not accumulate across tests in this module.
    """
    yield
    plt.close("all")


@pytest.mark.api
def test_plot_cluster_labels_compact_smoke(toy_results):
    """
    Ensures Plotter can render compact labels without errors.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = Plotter(toy_results).plot_matrix().plot_cluster_labels_compact()
        plotter.show()
        assert plotter._fig is not None
        texts = extract_figure_text(plotter._fig, strip=True, nonempty=True)
        assert texts
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_compact_default_draws_no_cluster_marker(toy_results):
    """
    Ensures default compact labels draw no matrix-side marker text (identity lives
    with the floating label only), while floating labels still carry exactly one
    alpha prefix from the default label_prefix="alpha".

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
            .plot_cluster_labels_compact(label_fields=("label",), wrap_text=False)
        )
        plotter.show()
        marker_ax, _bridge_ax, table_ax = plotter._fig.axes[-3:]

        marker_texts = [t.get_text().strip() for t in marker_ax.texts if t.get_text().strip()]
        assert not marker_texts, "Expected no matrix-side marker text by default."

        table_texts = [t.get_text().strip() for t in table_ax.texts if t.get_text().strip()]
        assert table_texts, "Expected floating label text to be rendered."
        for txt in table_texts:
            prefix_token = txt.split(".", 1)[0]
            assert prefix_token.isalpha() and prefix_token.isupper()
            assert txt.count(".") == 1
    finally:
        plt.show = plt_show


@pytest.mark.api
@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "line_shape": "curved",
            "cluster_span": "line",
            "cluster_span_cap_width": 0.15,
            "line_end": "arrow",
        },
        {"line_shape": "elbow", "line_start": "none", "line_end": "none"},
        {"line_start": "round", "line_end": "round"},
        {"cluster_marker": "cid", "font": "serif", "fontsize": 12},
    ],
    ids=["curved_capped_span_arrow", "elbow_none_none", "round_round", "cid_serif"],
)
def test_plot_cluster_labels_compact_style_variants_render(toy_results, kwargs):
    """
    Ensures representative compact-label style combinations render without breaking
    the figure.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
        kwargs (dict): plot_cluster_labels_compact() style kwargs under test.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = Plotter(toy_results).plot_matrix().plot_cluster_labels_compact(**kwargs)
        plotter.show()
        assert plotter._fig is not None
        assert len(plotter._fig.axes) >= 4
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_compact_invalid_style_raises(toy_results):
    """
    Ensures invalid style enum values raise ValueError.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        ValueError: If an unsupported style option is provided.
    """
    with pytest.raises(ValueError, match="cluster_marker"):
        Plotter(toy_results).plot_cluster_labels_compact(cluster_marker="bad")
    with pytest.raises(ValueError, match="line_shape"):
        Plotter(toy_results).plot_cluster_labels_compact(line_shape="zigzag")
    with pytest.raises(ValueError, match="line_style"):
        Plotter(toy_results).plot_cluster_labels_compact(line_style="bad")
    with pytest.raises(ValueError, match="cluster_span"):
        Plotter(toy_results).plot_cluster_labels_compact(cluster_span="bad")
    with pytest.raises(ValueError, match="line_start"):
        Plotter(toy_results).plot_cluster_labels_compact(line_start="bad")
    with pytest.raises(ValueError, match="line_end"):
        Plotter(toy_results).plot_cluster_labels_compact(line_end="bad")


@pytest.mark.api
def test_plot_cluster_labels_compact_and_cluster_labels_mutually_exclusive(toy_results):
    """
    Ensures plot_cluster_labels_compact() and plot_cluster_labels() cannot be declared together.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        ValueError: If both label layers are declared in the same plotting chain.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        with pytest.raises(ValueError, match="mutually exclusive"):
            (
                Plotter(toy_results)
                .plot_matrix()
                .plot_cluster_labels()
                .plot_cluster_labels_compact()
                .show()
            )
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_compact_label_fields_respect_np_order(toy_results):
    """
    Ensures compact table labels support the same label_fields content as standard
    labels, keeping label first while respecting q/fe/p/n order in a single stats block.

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
            .plot_cluster_labels_compact(label_fields=("label", "q", "fe", "p", "n"))
        )
        plotter.show()
        texts = extract_figure_text(plotter._fig, strip=True, nonempty=True)
        label_texts = [t for t in texts if "$q$=" in t and "FE=" in t and "$p$=" in t and "n=" in t]
        assert label_texts, "Expected compact table label text to be rendered."
        for txt in label_texts:
            assert " (" in txt
            assert txt.strip().endswith(")")
            assert txt.find("$q$=") < txt.find("FE=")
            assert txt.find("FE=") < txt.find("$p$=")
            assert txt.find("$p$=") < txt.find("n=")
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_compact_cluster_marker_and_label_prefix_independent(toy_results):
    """
    Ensures cluster_marker (matrix-side glyph) and label_prefix (floating-label identity)
    are independently controllable with no cross-contamination: alpha markers appear only
    in the marker column, and the cid prefix appears only in the floating label,
    not concatenated together.

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
            .plot_cluster_labels_compact(
                cluster_marker="alpha",
                label_prefix="cid",
                label_fields=("label",),
                wrap_text=False,
            )
        )
        plotter.show()
        marker_ax, table_ax = plotter._fig.axes[-3], plotter._fig.axes[-1]
        marker_texts = [t.get_text().strip() for t in marker_ax.texts if t.get_text().strip()]
        table_texts = [t.get_text().strip() for t in table_ax.texts if t.get_text().strip()]
        assert marker_texts and all(t.isalpha() and t.isupper() for t in marker_texts)
        # No alpha marker text should have leaked into the table column, and every
        # table row should lead with a numeric cid prefix, not an alpha token.
        assert table_texts
        for txt in table_texts:
            first_token = txt.split(" ", 1)[0].rstrip(".")
            assert first_token.isdigit()
            assert not any(marker in txt.split(" ", 1)[0] for marker in marker_texts if marker.isalpha())
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_compact_n_matches_layout_cluster_sizes(toy_results):
    """
    Ensures compact table "n=" values come from layout.cluster_sizes, matching
    standard label behavior, rather than being recomputed from span width.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        layout = toy_results.cluster_layout()
        plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_cluster_labels_compact(label_fields=("label", "n"), label_prefix="cid", wrap_text=False)
        )
        plotter.show()
        table_ax = plotter._fig.axes[-1]
        for txt in table_ax.texts:
            text = txt.get_text().strip()
            if not text or "n=" not in text:
                continue
            cid = int(text.split(".", 1)[0])
            rendered_n = int(text.split("n=", 1)[1].rstrip(")"))
            assert rendered_n == layout.cluster_sizes[cid]
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_compact_table_order_follows_dendrogram_span_order(toy_results):
    """
    Ensures table rows are emitted in the same order as ClusterLayout.cluster_spans
    (dendrogram/top-to-bottom order), not by cluster id or any other ordering.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        spans = toy_results.cluster_layout().cluster_spans
        expected_order = [int(cid) for cid, _s, _e in spans]

        plotter = Plotter(toy_results).plot_matrix().plot_cluster_labels_compact(label_prefix="cid")
        plotter.show()

        # Table-column texts render as "<cid prefix>. <label...>"; recover cid order by
        # the vertical (y) position of each text artist in the table axis (last axis
        # created by the compact-label panel).
        table_ax = plotter._fig.axes[-1]
        rows = [
            (t.get_position()[1], t.get_text())
            for t in table_ax.texts
            if t.get_text().strip()
        ]
        rows.sort(key=lambda pair: pair[0])
        rendered_order = [int(text.split(".", 1)[0]) for _y, text in rows]
        assert rendered_order == expected_order
    finally:
        plt.show = plt_show


@pytest.mark.api
@pytest.mark.parametrize("line_shape", ["straight", "curved", "elbow"])
def test_plot_cluster_labels_compact_cluster_span_renders_with_every_line_shape(toy_results, line_shape):
    """
    Ensures a capped cluster_span="line" is compatible with every leader-line shape:
    the cluster-side span sits at the matrix-side edge independent of how the leader
    line travels to the table.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
        line_shape (str): Leader-line shape under test.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_cluster_labels_compact(
                cluster_span="line",
                cluster_span_cap_width=0.15,
                line_shape=line_shape,
                cluster_span_gap=0.2,
            )
        )
        plotter.show()
        assert plotter._fig is not None
    finally:
        plt.show = plt_show


@pytest.mark.unit
def test_draw_cluster_span_clips_gap_and_draws_caps():
    """
    Ensures draw_cluster_span trims (s - 0.5 + gap, e + 0.5 - gap) — the cluster's
    true row extent, matching matrix/boundary/bar geometry — clamps an excessive
    gap to half that extent instead of inverting, and draws horizontal end caps
    only when cap_width > 0.

    Raises:
        AssertionError: If clamped extents or cap presence are wrong.
    """
    import matplotlib.pyplot as _plt

    from himalayas.plot.renderers._cluster_span import draw_cluster_span

    fig = _plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])

    # Normal cluster with caps: vertical stroke trims symmetrically from the true
    # row extent (s - 0.5 to e + 0.5), plus two caps.
    draw_cluster_span(ax, 0.0, 2, 8, gap=1.0, cap_width=0.1, color="black", lw=1.0, alpha=1.0)
    assert len(ax.lines) == 3
    vertical = [ln for ln in ax.lines if ln.get_xdata()[0] == ln.get_xdata()[1]][0]
    assert sorted(vertical.get_ydata()) == pytest.approx([2.5, 7.5])
    cap_ys = sorted(ln.get_ydata()[0] for ln in ax.lines if ln.get_xdata()[0] != ln.get_xdata()[1])
    assert cap_ys == pytest.approx([2.5, 7.5])
    # All span/cap artists disable axes-patch clipping so endpoints landing exactly on
    # the panel's row-index ylim (first/last cluster) aren't visually truncated.
    assert all(ln.get_clip_on() is False for ln in ax.lines)

    # No gap: span edges land exactly on the cluster's true row-extent boundaries
    # (s - 0.5, e + 0.5), matching cluster boundary lines and cluster bar rectangles.
    ax.clear()
    draw_cluster_span(ax, 0.0, 2, 8, gap=0.0, cap_width=0.0, color="black", lw=1.0, alpha=1.0)
    assert sorted(ax.lines[0].get_ydata()) == pytest.approx([1.5, 8.5])

    # Excessive gap: clamps to half the full row extent instead of crossing over.
    ax.clear()
    draw_cluster_span(ax, 0.0, 4, 6, gap=10.0, cap_width=0.0, color="black", lw=1.0, alpha=1.0)
    assert len(ax.lines) == 1
    assert sorted(ax.lines[0].get_ydata()) == pytest.approx([5.0, 5.0])

    # Singleton cluster: a small gap produces a real capped span within the true
    # one-row extent (s - 0.5 to e + 0.5).
    ax.clear()
    draw_cluster_span(ax, 0.0, 4, 4, gap=0.2, cap_width=0.1, color="black", lw=1.0, alpha=1.0)
    vertical = [ln for ln in ax.lines if ln.get_xdata()[0] == ln.get_xdata()[1]][0]
    assert sorted(vertical.get_ydata()) == pytest.approx([3.7, 4.3])

    # Singleton cluster, gap beyond half the row extent: collapses to a zero-height
    # span at the true center.
    ax.clear()
    draw_cluster_span(ax, 0.0, 4, 4, gap=0.6, cap_width=0.1, color="black", lw=1.0, alpha=1.0)
    vertical = [ln for ln in ax.lines if ln.get_xdata()[0] == ln.get_xdata()[1]][0]
    assert sorted(vertical.get_ydata()) == pytest.approx([4.0, 4.0])

    _plt.close(fig)


@pytest.mark.api
def test_plot_cluster_labels_compact_cluster_span_cap_width_has_no_cluster_marker(toy_results):
    """
    Ensures compact cluster_span="line" with a positive cluster_span_cap_width renders
    visible end caps (not a bare line) via the shared draw_cluster_span primitive, draws
    no matrix-side marker text by default, and gives each table row exactly one identity
    prefix (from label_prefix, not duplicated by a cluster marker).

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
            .plot_cluster_labels_compact(
                cluster_span="line",
                cluster_span_cap_width=0.15,
                label_fields=("label",),
                wrap_text=False,
            )
        )
        plotter.show()
        marker_ax, bridge_ax, table_ax = plotter._fig.axes[-3:]
        horizontal_lines = [
            ln for ln in bridge_ax.lines if len(ln.get_xdata()) == 2 and ln.get_xdata()[0] != ln.get_xdata()[1]
        ]
        assert horizontal_lines, "Expected end caps when cluster_span_cap_width > 0."

        marker_texts = [t.get_text().strip() for t in marker_ax.texts if t.get_text().strip()]
        assert not marker_texts, "Expected no matrix-side marker text by default."

        table_texts = [t.get_text().strip() for t in table_ax.texts if t.get_text().strip()]
        assert table_texts
        for txt in table_texts:
            prefix_token = txt.split(".", 1)[0]
            assert prefix_token.isalpha() and prefix_token.isupper()
            assert txt.count(".") == 1
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_cluster_span_opt_in_renders(toy_results):
    """
    Ensures plot_cluster_labels(cluster_span="line") renders a span only when opted in.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        default_plotter = Plotter(toy_results).plot_matrix().plot_cluster_labels()
        default_plotter.show()
        default_lines = len(default_plotter._fig.axes[-1].lines)

        span_plotter = (
            Plotter(toy_results).plot_matrix().plot_cluster_labels(cluster_span="line")
        )
        span_plotter.show()
        span_lines = len(span_plotter._fig.axes[-1].lines)
        assert span_lines > default_lines
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_cluster_span_style_kwargs_propagate(toy_results):
    """
    Ensures cluster_span_color/lw/gap/cap_width propagate to the drawn span artists.

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
                cluster_span="line",
                cluster_span_color="#1b9e77",
                cluster_span_lw=2.5,
                cluster_span_gap=0.3,
                cluster_span_cap_width=0.02,
            )
        )
        plotter.show()
        ax_lab = plotter._fig.axes[-1]
        # Isolate span artists (linewidth 2.5) from separator lines, which use the
        # unrelated default label_sep_lw and would otherwise dilute this assertion.
        span_lines = [ln for ln in ax_lab.lines if ln.get_linewidth() == pytest.approx(2.5)]
        assert span_lines, "Expected span artists to be drawn."
        assert all(ln.get_color() == "#1b9e77" for ln in span_lines)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_invalid_cluster_span_raises(toy_results):
    """
    Ensures an unsupported cluster_span value or a negative gap/cap width/pad raises
    ValueError.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        ValueError: If cluster_span is not one of {None, "line"}, or
            cluster_span_gap, cluster_span_cap_width, cluster_span_left_pad, or
            cluster_span_right_pad is negative.
    """
    with pytest.raises(ValueError, match="cluster_span"):
        Plotter(toy_results).plot_cluster_labels(cluster_span="bad")
    with pytest.raises(ValueError, match="cluster_span_gap"):
        Plotter(toy_results).plot_cluster_labels(cluster_span="line", cluster_span_gap=-0.1)
    with pytest.raises(ValueError, match="cluster_span_cap_width"):
        Plotter(toy_results).plot_cluster_labels(cluster_span="line", cluster_span_cap_width=-0.1)
    with pytest.raises(ValueError, match="cluster_span_left_pad"):
        Plotter(toy_results).plot_cluster_labels(cluster_span="line", cluster_span_left_pad=-0.1)
    with pytest.raises(ValueError, match="cluster_span_right_pad"):
        Plotter(toy_results).plot_cluster_labels(cluster_span="line", cluster_span_right_pad=-0.1)


@pytest.mark.api
def test_plot_cluster_labels_cluster_span_pads_position_span_and_label_text(toy_results):
    """
    Ensures cluster_span_left_pad/right_pad position the span and label text as
    span_x = end_x + left_pad and label_text_x = span_x + right_pad, with no tracks
    registered (end_x = style label_x + label_gutter_width). Guards against span
    position being reverse-derived from label_text_x / label_bar_pad.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        left_pad = 0.05
        right_pad = 0.02
        span_color = "#1b9e77"
        plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_cluster_labels(
                cluster_span="line",
                cluster_span_left_pad=left_pad,
                cluster_span_right_pad=right_pad,
                cluster_span_color=span_color,
                wrap_text=False,
            )
        )
        plotter.show()
        style = plotter._style
        end_x = style["label_x"] + style["label_gutter_width"]
        expected_span_x = end_x + left_pad
        expected_label_text_x = expected_span_x + right_pad

        ax_lab = plotter._fig.axes[-1]
        # Isolate span artists by their distinctive color, not just vertical orientation,
        # since separator/boundary lines in this axis are also vertical or horizontal.
        span_lines = [
            ln
            for ln in ax_lab.lines
            if ln.get_color() == span_color and ln.get_xdata()[0] == ln.get_xdata()[1]
        ]
        assert span_lines, "Expected at least one vertical span line."
        assert all(
            ln.get_xdata()[0] == pytest.approx(expected_span_x) for ln in span_lines
        )

        label_texts = [t for t in ax_lab.texts if t.get_text().strip()]
        assert label_texts, "Expected label text to be drawn."
        assert all(
            t.get_position()[0] == pytest.approx(expected_label_text_x) for t in label_texts
        )
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_cluster_span_pad_defaults_match_explicit_values(toy_results):
    """
    Ensures the default cluster_span_left_pad/right_pad values (0.0 and 0.01) produce
    identical span/label-text positions to passing those values explicitly.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        span_color = "#1b9e77"
        default_plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_cluster_labels(cluster_span="line", cluster_span_color=span_color)
        )
        default_plotter.show()
        default_ax = default_plotter._fig.axes[-1]
        default_span_x = [
            ln.get_xdata()[0]
            for ln in default_ax.lines
            if ln.get_color() == span_color and ln.get_xdata()[0] == ln.get_xdata()[1]
        ]
        default_label_x = [t.get_position()[0] for t in default_ax.texts if t.get_text().strip()]

        explicit_plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_cluster_labels(
                cluster_span="line",
                cluster_span_left_pad=0.0,
                cluster_span_right_pad=0.01,
                cluster_span_color=span_color,
            )
        )
        explicit_plotter.show()
        explicit_ax = explicit_plotter._fig.axes[-1]
        explicit_span_x = [
            ln.get_xdata()[0]
            for ln in explicit_ax.lines
            if ln.get_color() == span_color and ln.get_xdata()[0] == ln.get_xdata()[1]
        ]
        explicit_label_x = [
            t.get_position()[0] for t in explicit_ax.texts if t.get_text().strip()
        ]

        assert default_span_x and explicit_span_x
        assert default_span_x == pytest.approx(explicit_span_x)
        assert default_label_x and explicit_label_x
        assert default_label_x == pytest.approx(explicit_label_x)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_compact_boundary_kwargs_reach_matrix_boundaries(toy_results):
    """
    Ensures compact boundary_color/lw/alpha reach the same matrix boundary-line
    rendering that plot_cluster_labels() already drives, closing the gap where
    _collect_layer_kwargs() only recognized "cluster_labels" layers.

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
            .plot_cluster_labels_compact(
                boundary_color="#e41a1c", boundary_lw=3.0, boundary_alpha=0.9
            )
        )
        plotter.show()
        matrix_ax = plotter._fig.axes[0]
        collections = [c for c in matrix_ax.collections if hasattr(c, "get_linewidths")]
        boundary_collections = [
            c for c in collections if any(lw == pytest.approx(3.0) for lw in c.get_linewidths())
        ]
        assert boundary_collections, "Expected a boundary LineCollection with lw=3.0."
        # The registry merges cluster boundaries with unrelated minor-row gridlines into
        # one LineCollection; isolate the segment matching our distinctive lw=3.0.
        expected_rgba = to_rgba("#e41a1c", 0.9)
        cluster_boundary_rgba = [
            rgba
            for c in boundary_collections
            for lw, rgba in zip(c.get_linewidths(), c.get_colors())
            if lw == pytest.approx(3.0)
        ]
        assert cluster_boundary_rgba
        assert all(tuple(rgba) == pytest.approx(expected_rgba) for rgba in cluster_boundary_rgba)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_compact_cluster_span_cap_width_propagates(toy_results):
    """
    Ensures an explicit cluster_span_cap_width on the compact bridge axis draws end
    caps of exactly that width.

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
            .plot_cluster_labels_compact(cluster_span="line", cluster_span_cap_width=0.15)
        )
        plotter.show()
        bridge_ax = plotter._fig.axes[-2]
        # End caps are horizontal, straddle the matrix-side x=0.0 centerline, and are
        # distinct from the leader line itself (which spans the full x in [0, 1]).
        cap_lines = [
            ln
            for ln in bridge_ax.lines
            if len(ln.get_xdata()) == 2
            and ln.get_ydata()[0] == ln.get_ydata()[1]
            and ln.get_xdata()[0] == pytest.approx(-ln.get_xdata()[1])
        ]
        assert cap_lines, "Expected end caps when cluster_span_cap_width > 0."
        cap_widths = [abs(ln.get_xdata()[1] - ln.get_xdata()[0]) for ln in cap_lines]
        assert all(w == pytest.approx(0.15) for w in cap_widths)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_cluster_span_default_cap_width_is_bare_line(toy_results):
    """
    Ensures cluster_span="line" with no explicit cluster_span_cap_width draws no
    horizontal end-cap lines by default, in both the standard and compact label
    panels — the 0.0 default must not silently reintroduce caps.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        standard_plotter = (
            Plotter(toy_results).plot_matrix().plot_cluster_labels(cluster_span="line")
        )
        standard_plotter.show()
        ax_lab = standard_plotter._fig.axes[-1]
        # Cap lines are short horizontal segments straddling the span centerline;
        # separator lines are unrelated full-width horizontal segments and must be
        # excluded so this assertion isolates cap presence specifically.
        standard_caps = [
            ln
            for ln in ax_lab.lines
            if len(ln.get_xdata()) == 2
            and ln.get_xdata()[0] != ln.get_xdata()[1]
            and abs(ln.get_xdata()[1] - ln.get_xdata()[0]) < 0.1
        ]
        assert not standard_caps, "Expected no end caps by default for cluster_span='line'."

        compact_plotter = (
            Plotter(toy_results).plot_matrix().plot_cluster_labels_compact(cluster_span="line")
        )
        compact_plotter.show()
        bridge_ax = compact_plotter._fig.axes[-2]
        compact_caps = [
            ln
            for ln in bridge_ax.lines
            if len(ln.get_xdata()) == 2
            and ln.get_ydata()[0] == ln.get_ydata()[1]
            and ln.get_xdata()[0] == pytest.approx(-ln.get_xdata()[1])
        ]
        assert not compact_caps, "Expected no end caps by default for cluster_span='line'."
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_compact_cluster_span_style_independent_of_line_style(toy_results):
    """
    Ensures cluster_span_color/lw/alpha are resolved independently of line_color/lw/alpha:
    setting only line_color must not change the rendered span color, and
    cluster_span_color must be set explicitly to affect it.

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
            .plot_cluster_labels_compact(
                cluster_span="line",
                cluster_span_color="#1b9e77",
                cluster_span_lw=2.5,
                line_color="#c0562c",
                line_lw=0.9,
            )
        )
        plotter.show()
        bridge_ax = plotter._fig.axes[-2]
        span_lines = [ln for ln in bridge_ax.lines if ln.get_linewidth() == pytest.approx(2.5)]
        assert span_lines, "Expected span artists drawn at cluster_span_lw."
        assert all(ln.get_color() == "#1b9e77" for ln in span_lines)
        assert not any(ln.get_color() == "#c0562c" for ln in span_lines)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_compact_line_start_rejects_arrow_line_end_allows_it(toy_results):
    """
    Ensures line_start and line_end keep distinct value domains after the rename:
    "arrow" is valid only for line_end (table-side), not line_start (matrix-side),
    guarding against the two enums being accidentally merged.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        ValueError: If line_start="arrow" is rejected as expected.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        with pytest.raises(ValueError, match="line_start"):
            Plotter(toy_results).plot_cluster_labels_compact(line_start="arrow")

        plotter = Plotter(toy_results).plot_matrix().plot_cluster_labels_compact(line_end="arrow")
        plotter.show()
        assert plotter._fig is not None
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_compact_anchors_to_custom_label_panel(toy_results):
    """
    Ensures compact labels default to the label-panel region set via set_label_panel(axes=...),
    matching standard cluster labels' vertical anchoring rather than a stale compact_axes default.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        custom_axes = [0.61, 0.13, 0.35, 0.77]
        plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .set_label_panel(axes=custom_axes)
            .plot_cluster_labels_compact()
        )
        plotter.show()

        compact_axes = plotter._fig.axes[-3:]
        assert len(compact_axes) == 3
        for ax in compact_axes:
            x0, y0, w, h = ax.get_position().bounds
            assert y0 == pytest.approx(custom_axes[1])
            assert h == pytest.approx(custom_axes[3])
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_compact_supports_cluster_bar_and_bar_labels(toy_results):
    """
    Ensures plot_cluster_bar() and plot_bar_labels() render without error alongside
    plot_cluster_labels_compact(), since cluster bars are cluster-frame metadata and
    should not require standard inline labels specifically.

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
            .plot_cluster_labels_compact()
            .plot_cluster_bar(name="sig", title="Enrichment")
            .plot_bar_labels()
        )
        plotter.show()
        assert plotter._fig is not None
        texts = extract_figure_text(plotter._fig, strip=True, nonempty=True)
        assert texts
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_compact_cluster_bar_track_left_of_compact_axes(toy_results):
    """
    Ensures the cluster-bar track occupies a region strictly to the left of the compact
    marker/bridge/table axes, i.e. it does not overlap the compact equal-slot geometry.

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
            .plot_cluster_labels_compact()
            .plot_cluster_bar(name="sig")
        )
        plotter.show()

        track_ax, marker_ax, bridge_ax, table_ax = plotter._fig.axes[-4:]
        track_x0, _, track_w, _ = track_ax.get_position().bounds
        marker_x0, _, _, _ = marker_ax.get_position().bounds

        assert track_x0 + track_w == pytest.approx(marker_x0)
        for ax in (marker_ax, bridge_ax, table_ax):
            ax_x0, _, _, _ = ax.get_position().bounds
            assert track_x0 + track_w <= ax_x0 + 1e-9
    finally:
        plt.show = plt_show


@pytest.mark.api
@pytest.mark.parametrize("line_shape", ["straight", "curved", "elbow"])
def test_plot_cluster_labels_compact_cluster_span_pads_position_span_and_leader_start(
    toy_results, line_shape
):
    """
    Ensures cluster_span_left_pad/right_pad position the compact span and leader-line
    start as span_x = left_pad and leader_start_x = span_x + right_pad, on the bridge
    axis's local x in [0, 1] (0.0 = matrix/marker-side edge, 1.0 = table side), for
    every leader-line shape (each resolves x_start differently in _resolve_line_path).
    Also ensures the pads have no effect when cluster_span is inactive: the leader-line
    start stays at x=0.0.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
        line_shape (str): Leader-line shape under test.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        span_color = "#1b9e77"
        plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_cluster_labels_compact(
                cluster_span="line",
                cluster_span_left_pad=0.05,
                cluster_span_right_pad=0.02,
                cluster_span_color=span_color,
                line_shape=line_shape,
            )
        )
        plotter.show()
        bridge_ax = plotter._fig.axes[-2]

        span_lines = [
            ln
            for ln in bridge_ax.lines
            if ln.get_color() == span_color and ln.get_xdata()[0] == ln.get_xdata()[1]
        ]
        assert span_lines, "Expected at least one vertical span line."
        assert all(ln.get_xdata()[0] == pytest.approx(0.05) for ln in span_lines)

        leader_lines = [
            ln for ln in bridge_ax.lines if ln.get_color() != span_color and len(ln.get_xdata()) > 1
        ]
        assert leader_lines, "Expected leader lines to be drawn."
        assert all(ln.get_xdata()[0] == pytest.approx(0.07) for ln in leader_lines)
    finally:
        plt.show = plt_show

    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        no_span_plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_cluster_labels_compact(
                cluster_span_left_pad=0.05,
                cluster_span_right_pad=0.03,
                line_shape=line_shape,
            )
        )
        no_span_plotter.show()
        no_span_bridge_ax = no_span_plotter._fig.axes[-2]
        no_span_leader_lines = [
            ln
            for ln in no_span_bridge_ax.lines
            if len(ln.get_xdata()) > 1 and ln.get_xdata()[0] != ln.get_xdata()[-1]
        ]
        assert no_span_leader_lines, "Expected leader lines to be drawn."
        assert all(ln.get_xdata()[0] == pytest.approx(0.0) for ln in no_span_leader_lines)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_compact_invalid_cluster_span_pads_raise(toy_results):
    """
    Ensures negative cluster_span_left_pad/right_pad/label_left_pad raise ValueError,
    matching the standard-label validation pattern.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        ValueError: If cluster_span_left_pad, cluster_span_right_pad, or
            label_left_pad is negative.
    """
    with pytest.raises(ValueError, match="cluster_span_left_pad"):
        Plotter(toy_results).plot_cluster_labels_compact(
            cluster_span="line", cluster_span_left_pad=-0.1
        )
    with pytest.raises(ValueError, match="cluster_span_right_pad"):
        Plotter(toy_results).plot_cluster_labels_compact(
            cluster_span="line", cluster_span_right_pad=-0.1
        )
    with pytest.raises(ValueError, match="label_left_pad"):
        Plotter(toy_results).plot_cluster_labels_compact(label_left_pad=-0.1)


@pytest.mark.api
def test_plot_cluster_labels_compact_no_marker_reserves_zero_marker_width(toy_results):
    """
    Ensures cluster_marker=None (the default) reserves no marker-column width, so
    cluster_span_left_pad=0.0 places the bridge axis (and span) flush against the
    matrix/track edge instead of behind an empty compact_marker_width gap. Covers both
    no-track (bridge starts at the marker axis x0) and with-track (bridge starts
    immediately after the cluster-bar track axis, not after an intervening empty
    marker column) cases.

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
            .plot_cluster_labels_compact(cluster_span="line", cluster_span_left_pad=0.0)
        )
        plotter.show()
        marker_ax, bridge_ax, _table_ax = plotter._fig.axes[-3:]
        marker_x0, _, marker_w, _ = marker_ax.get_position().bounds
        bridge_x0, _, _, _ = bridge_ax.get_position().bounds
        assert marker_w == pytest.approx(0.0)
        assert bridge_x0 == pytest.approx(marker_x0)
    finally:
        plt.show = plt_show

    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        tracked_plotter = (
            Plotter(toy_results)
            .plot_matrix()
            .plot_cluster_labels_compact(cluster_span="line", cluster_span_left_pad=0.0)
            .plot_cluster_bar(name="sig")
        )
        tracked_plotter.show()
        track_ax, marker_ax, bridge_ax, _table_ax = tracked_plotter._fig.axes[-4:]
        track_x0, _, track_w, _ = track_ax.get_position().bounds
        marker_x0, _, marker_w, _ = marker_ax.get_position().bounds
        bridge_x0, _, _, _ = bridge_ax.get_position().bounds
        assert marker_w == pytest.approx(0.0)
        assert marker_x0 == pytest.approx(track_x0 + track_w)
        assert bridge_x0 == pytest.approx(track_x0 + track_w)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_compact_cluster_bar_patches_inside_track_axis_xlim(toy_results):
    """
    Ensures compact cluster-bar patches are drawn inside ax_trk's own [0, 1] data range,
    not at ax_trk's figure-coordinate position. TrackLayoutManager stores track x0/x1/width
    in figure coordinates, but ax_trk (created by _setup_compact_axes) has local xlim
    [0, 1]; unlocalized figure-coordinate patches would land far outside that range and
    render invisibly.

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
            .plot_cluster_labels_compact()
            .plot_cluster_bar(name="sig")
        )
        plotter.show()
        track_ax = plotter._fig.axes[-4]
        xlim = track_ax.get_xlim()
        x_lo, x_hi = min(xlim), max(xlim)
        assert track_ax.patches, "Expected cluster-bar patches on ax_trk."
        for patch in track_ax.patches:
            x, _y = patch.get_xy()
            assert x_lo - 1e-9 <= x <= x_hi + 1e-9
            assert x_lo - 1e-9 <= x + patch.get_width() <= x_hi + 1e-9
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_compact_cluster_bar_right_edge_aligns_with_bridge_start(
    toy_results,
):
    """
    Ensures that with plot_cluster_bar(right_pad=0.0), cluster_marker=None,
    cluster_span="line", and cluster_span_left_pad=0.0, the cluster bar's visible right
    edge (converted from ax_trk-local to figure coordinates) aligns with the compact
    bridge axis's figure-coordinate start, i.e. no fake gap between the bar and the
    compact span/leader region.

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
            .plot_cluster_labels_compact(
                cluster_marker=None,
                cluster_span="line",
                cluster_span_left_pad=0.0,
            )
            .plot_cluster_bar(name="sig", right_pad=0.0)
        )
        plotter.show()
        track_ax, _marker_ax, bridge_ax, _table_ax = plotter._fig.axes[-4:]
        track_x0, _, track_w, _ = track_ax.get_position().bounds
        bridge_x0, _, _, _ = bridge_ax.get_position().bounds

        assert track_ax.patches, "Expected cluster-bar patches on ax_trk."
        bar_right_local = max(p.get_xy()[0] + p.get_width() for p in track_ax.patches)
        bar_right_fig = track_x0 + bar_right_local * track_w

        assert bar_right_fig == pytest.approx(bridge_x0, abs=1e-6)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_compact_label_left_pad_moves_table_text_only(toy_results):
    """
    Ensures label_left_pad moves floating label text to that table-axis-local x
    without changing leader-line geometry: the leader line still ends at bridge-axis
    x=1.0, so only the text is repositioned.

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
            .plot_cluster_labels_compact(label_left_pad=0.05)
        )
        plotter.show()
        bridge_ax, table_ax = plotter._fig.axes[-2:]

        table_texts = [t for t in table_ax.texts if t.get_text().strip()]
        assert table_texts, "Expected floating label text to be rendered."
        assert all(t.get_position()[0] == pytest.approx(0.05) for t in table_texts)

        leader_lines = [ln for ln in bridge_ax.lines if len(ln.get_xdata()) > 1]
        assert leader_lines, "Expected leader lines to be drawn."
        assert all(ln.get_xdata()[-1] == pytest.approx(1.0) for ln in leader_lines)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_plot_cluster_labels_compact_label_left_pad_default_is_zero(toy_results):
    """
    Ensures the default label_left_pad (unset) preserves current behavior: floating
    label text starts at table-axis x=0.0.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = Plotter(toy_results).plot_matrix().plot_cluster_labels_compact()
        plotter.show()
        table_ax = plotter._fig.axes[-1]
        table_texts = [t for t in table_ax.texts if t.get_text().strip()]
        assert table_texts, "Expected floating label text to be rendered."
        assert all(t.get_position()[0] == pytest.approx(0.0) for t in table_texts)
    finally:
        plt.show = plt_show
