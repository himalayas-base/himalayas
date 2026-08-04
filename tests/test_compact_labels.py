"""
tests/test_compact_labels
~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import matplotlib.pyplot as plt
import pytest

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
@pytest.mark.parametrize(
    "kwargs",
    [
        {"line_shape": "curved", "source_end": "span", "target_end": "arrow"},
        {"line_shape": "elbow", "source_end": "none", "target_end": "none"},
        {"source_end": "round", "target_end": "round"},
        {"marker_prefix": "cid", "font": "serif", "fontsize": 12},
    ],
    ids=["curved_span_arrow", "elbow_none_none", "round_round", "cid_serif"],
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
    with pytest.raises(ValueError, match="marker_prefix"):
        Plotter(toy_results).plot_cluster_labels_compact(marker_prefix="bad")
    with pytest.raises(ValueError, match="line_shape"):
        Plotter(toy_results).plot_cluster_labels_compact(line_shape="zigzag")
    with pytest.raises(ValueError, match="line_style"):
        Plotter(toy_results).plot_cluster_labels_compact(line_style="bad")
    with pytest.raises(ValueError, match="source_end"):
        Plotter(toy_results).plot_cluster_labels_compact(source_end="bad")
    with pytest.raises(ValueError, match="target_end"):
        Plotter(toy_results).plot_cluster_labels_compact(target_end="bad")


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
def test_plot_cluster_labels_compact_label_prefix_distinct_from_marker_prefix(toy_results):
    """
    Ensures label_prefix (table content) and marker_prefix (marker glyph) act
    independently: markers can be alpha-prefixed while table text is cid-prefixed.

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
                marker_prefix="alpha",
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
        # Table rows read "<alpha marker>.  <cid prefix> <label>"; the cid token
        # appears after the marker, distinct from the alpha marker itself.
        assert any(txt.split(".  ", 1)[-1].split(" ", 1)[0].rstrip(".").isdigit() for txt in table_texts)
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
            .plot_cluster_labels_compact(label_fields=("label", "n"), marker_prefix="cid", wrap_text=False)
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

        plotter = Plotter(toy_results).plot_matrix().plot_cluster_labels_compact(marker_prefix="cid")
        plotter.show()

        # Table-column texts render as "<marker>.  <label...>"; recover marker order by
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
def test_plot_cluster_labels_compact_source_span_renders_with_every_line_shape(toy_results, line_shape):
    """
    Ensures source_end="span" is compatible with every leader-line shape: the source
    bracket sits at the matrix-side edge independent of how the leader line travels
    to the table.

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
            .plot_cluster_labels_compact(source_end="span", line_shape=line_shape, source_gap=0.2)
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

    # No gap: bracket edges land exactly on the cluster's true row-extent boundaries
    # (s - 0.5, e + 0.5), matching cluster boundary lines and cluster bar rectangles.
    ax.clear()
    draw_cluster_span(ax, 0.0, 2, 8, gap=0.0, cap_width=0.0, color="black", lw=1.0, alpha=1.0)
    assert sorted(ax.lines[0].get_ydata()) == pytest.approx([1.5, 8.5])

    # Excessive gap: clamps to half the full row extent instead of crossing over.
    ax.clear()
    draw_cluster_span(ax, 0.0, 4, 6, gap=10.0, cap_width=0.0, color="black", lw=1.0, alpha=1.0)
    assert len(ax.lines) == 1
    assert sorted(ax.lines[0].get_ydata()) == pytest.approx([5.0, 5.0])

    # Singleton cluster: a small gap produces a real bracket within the true
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
def test_plot_cluster_labels_compact_source_span_renders_capped_bracket(toy_results):
    """
    Ensures compact source_end="span" renders visible end caps by default (a bracket,
    not a subtle bare line), via the shared draw_cluster_span primitive.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plotter = Plotter(toy_results).plot_matrix().plot_cluster_labels_compact(source_end="span")
        plotter.show()
        bridge_ax = plotter._fig.axes[-2]
        horizontal_lines = [
            ln for ln in bridge_ax.lines if len(ln.get_xdata()) == 2 and ln.get_xdata()[0] != ln.get_xdata()[1]
        ]
        assert horizontal_lines, "Expected bracket end caps for source_end='span'."
    finally:
        plt.show = plt_show


@pytest.mark.api
@pytest.mark.parametrize("cluster_span", ["line", "bracket"])
def test_plot_cluster_labels_cluster_span_opt_in_renders(toy_results, cluster_span):
    """
    Ensures plot_cluster_labels(cluster_span=...) renders a span only when opted in,
    for both "line" and "bracket" modes.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
        cluster_span (str): Span mode under test.
    """
    plt = use_agg_backend()
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        default_plotter = Plotter(toy_results).plot_matrix().plot_cluster_labels()
        default_plotter.show()
        default_lines = len(default_plotter._fig.axes[-1].lines)

        span_plotter = (
            Plotter(toy_results).plot_matrix().plot_cluster_labels(cluster_span=cluster_span)
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
                cluster_span="bracket",
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
        ValueError: If cluster_span is not one of {None, "line", "bracket"}, or
            cluster_span_gap, cluster_span_cap_width, cluster_span_left_pad, or
            cluster_span_right_pad is negative.
    """
    with pytest.raises(ValueError, match="cluster_span"):
        Plotter(toy_results).plot_cluster_labels(cluster_span="bad")
    with pytest.raises(ValueError, match="cluster_span_gap"):
        Plotter(toy_results).plot_cluster_labels(cluster_span="line", cluster_span_gap=-0.1)
    with pytest.raises(ValueError, match="cluster_span_cap_width"):
        Plotter(toy_results).plot_cluster_labels(cluster_span="bracket", cluster_span_cap_width=-0.1)
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
