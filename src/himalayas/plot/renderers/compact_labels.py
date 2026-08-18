"""
himalayas/plot/renderers/compact_labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._cluster_label_data import _build_label_map, _parse_label_overrides
from ._cluster_span import draw_cluster_span
from ._compact_label_types import (
    CLUSTER_MARKERS,
    CLUSTER_SPANS,
    LINE_ENDS,
    LINE_SHAPES,
    LINE_STARTS,
    LINE_STYLES,
)
from ._label_format import compute_equal_slots, format_label_prefix, resolve_cluster_label_content
from ._text_style import apply_text_style
from ._track_rendering import TrackSpec, _render_tracks, resolve_track_strip

if TYPE_CHECKING:
    from ..style import StyleConfig
    from ..track_layout import TrackLayoutManager
    from ...core.layout import ClusterLayout
    from ...core.matrix import Matrix

# Matplotlib linestyle codes for each LINE_STYLES value.
_MPL_LINESTYLES = {"solid": "-", "dashed": "--", "dotted": ":"}


def _resolve_line_path(
    y0: float,
    y1: float,
    *,
    shape: str,
    x_start: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resolves an (x, y) path from the leader-line start (x=x_start) to the table column (x=1).

    Args:
        y0 (float): Start y-position (true cluster center, row-index space).
        y1 (float): End y-position (equal-pitch table slot).

    Kwargs:
        shape (str): One of {"straight", "curved", "elbow"}.
        x_start (float): Leader-line start x, in bridge-axis fraction. 0.0 is the
            matrix/marker-side edge; > 0.0 when cluster_span_right_pad pushes the
            start past a cluster_span. Defaults to 0.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (x values, y values) along the path.

    Raises:
        ValueError: If shape is unsupported.
    """
    if shape == "straight":
        return np.array([x_start, 1.0]), np.array([y0, y1])
    if shape == "elbow":
        # Horizontal-then-diagonal-then-horizontal elbow, bent at the midpoint.
        mid = (x_start + 1.0) / 2.0
        return np.array([x_start, mid, mid, 1.0]), np.array([y0, y0, y1, y1])
    if shape == "curved":
        xs = np.linspace(x_start, 1.0, 40)
        t = (xs - x_start) / (1.0 - x_start) if x_start != 1.0 else np.zeros_like(xs)
        t_smooth = t * t * (3.0 - 2.0 * t)
        ys = y0 + (y1 - y0) * t_smooth
        return xs, ys
    raise ValueError(f"line_shape must be one of {sorted(LINE_SHAPES)}, got {shape!r}")


def _draw_line_start(
    ax: plt.Axes,
    y_center: float,
    *,
    kind: str,
    color: str,
    alpha: float,
) -> None:
    """
    Draws the connector's matrix-side start-point decoration for one cluster's leader
    line. Callers invoke this only when cluster_span is None, to mark the leader line's
    origin in place of a cluster-extent span.

    Args:
        ax (plt.Axes): Bridge axis spanning x in [0, 1].
        y_center (float): True cluster center, (s + e) / 2.

    Kwargs:
        kind (str): Connector-start decoration, one of {"tick", "round", "none"}.
            Callers validate this against LINE_STARTS before rendering.
        color (str): Marker color.
        alpha (float): Opacity.
    """
    if kind == "none":
        return
    if kind == "tick":
        ax.plot([0.0], [y_center], marker="|", color=color, markersize=5, alpha=alpha)
        return
    ax.plot([0.0], [y_center], marker="o", color=color, markersize=4, alpha=alpha)


def _draw_leader_line(
    ax: plt.Axes,
    y0: float,
    y1: float,
    *,
    shape: str,
    linestyle: str,
    line_end: str,
    color: str,
    lw: float,
    alpha: float,
    x_start: float = 0.0,
) -> None:
    """
    Draws one leader line from a marker's true center to its table slot, with an
    optional decoration at the table-side end.

    Args:
        ax (plt.Axes): Bridge axis spanning x in [0, 1].
        y0 (float): Start y-position (marker side).
        y1 (float): End y-position (table side).

    Kwargs:
        shape (str): Line shape, one of {"straight", "curved", "elbow"}.
        linestyle (str): Matplotlib linestyle, one of {"solid", "dashed", "dotted"}.
        line_end (str): Table-side endpoint decoration, one of {"tick", "arrow", "round", "none"}.
            Callers validate this against LINE_ENDS before rendering.
        color (str): Line color.
        lw (float): Line width.
        alpha (float): Line opacity.
        x_start (float): Leader-line start x, in bridge-axis fraction. Defaults to 0.0.
    """
    xs, ys = _resolve_line_path(y0, y1, shape=shape, x_start=x_start)
    mpl_linestyle = _MPL_LINESTYLES.get(linestyle, "-")
    solid_capstyle = "round" if line_end == "round" else "butt"
    if line_end == "arrow":
        ax.annotate(
            "",
            xy=(1.0, y1),
            xytext=(xs[-2], ys[-2]) if len(xs) > 1 else (x_start, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                alpha=alpha,
                shrinkA=0,
                shrinkB=0,
            ),
        )
        if len(xs) > 2:
            ax.plot(
                xs[:-1],
                ys[:-1],
                linestyle=mpl_linestyle,
                color=color,
                linewidth=lw,
                alpha=alpha,
                solid_capstyle=solid_capstyle,
            )
    else:
        ax.plot(
            xs,
            ys,
            linestyle=mpl_linestyle,
            color=color,
            linewidth=lw,
            alpha=alpha,
            solid_capstyle=solid_capstyle,
        )
        if line_end not in {"none", "round"}:
            ax.plot([1.0], [y1], marker="|", color=color, markersize=5, alpha=alpha)


class _CompactGeometry(NamedTuple):
    """
    Resolved compact label-panel axes plus the label-panel-fraction allocations the
    renderer needs to convert public pads into axis-local coordinates.
    """

    marker: plt.Axes
    bridge: plt.Axes
    table: plt.Axes
    track: Optional[plt.Axes]
    tracks: List[TrackSpec]
    connector_frac: float
    table_frac: float


def _setup_compact_axes(
    fig: plt.Figure,
    n_rows: int,
    style: StyleConfig,
    track_layout: Optional[TrackLayoutManager] = None,
    *,
    reserve_marker: bool = True,
    connector_width: Optional[float] = None,
    span_pads: Optional[Tuple[float, float]] = None,
) -> _CompactGeometry:
    """
    Creates the marker, bridge, and table sub-axes for the compact-label panel, plus an
    optional leading track axis carrying the label-panel tracks (row-level tracks from
    plot_label_bar() and cluster-level tracks from plot_cluster_bar()).

    Every horizontal allocation is a label-panel axes fraction, the same currency used by
    track widths/pads and by plot_cluster_labels(). The sub-axes each span x in [0, 1]
    locally; callers convert public pads into those local frames via the returned
    connector_frac/table_frac.

    Args:
        fig (plt.Figure): Target figure.
        n_rows (int): Number of matrix rows.
        style (StyleConfig): Style configuration.

    Kwargs:
        track_layout (Optional[TrackLayoutManager]): Registered label-panel tracks, if any.
            Defaults to None.
        reserve_marker (bool): Whether to reserve compact_marker_width for the marker
            column. False when no marker text will be drawn (cluster_marker=None), so
            the bridge axis starts flush against the track/matrix edge. Defaults to True.
        connector_width (Optional[float]): Connector/leader-line region width as a
            label-panel axes fraction. Defaults to None (style compact_bridge_width).
        span_pads (Optional[Tuple[float, float]]): Resolved
            (cluster_span_left_pad, cluster_span_right_pad) in label-panel axes fractions,
            checked against the connector region here so every geometry rejection happens
            before any axis is added. None when no cluster span is drawn and the pads
            therefore do not apply. Defaults to None.

    Returns:
        _CompactGeometry: Resolved axes, the tracks the reserved geometry was measured
        from (so callers render exactly what was measured), and the connector/table
        label-panel fractions.

    Raises:
        ValueError: If connector_width is not > 0, if the span pads overrun the connector
            region, or if track widths/pads and connector_width leave no room for the
            floating label column.
    """
    compact_axes = style.get("compact_axes", None)
    x0, y0, w, h = compact_axes if compact_axes is not None else style["label_axes"]

    # Resolve tracks in the label panel's local axes fraction, seeded exactly as
    # _setup_label_axis() does, so a track lands identically in either label renderer.
    # The track axis spans the whole panel with xlim (0, 1), so tracks draw at their
    # resolved positions with no coordinate conversion.
    tracks, end_x = resolve_track_strip(track_layout, style)
    track_x0, track_w = x0, w

    # Label-panel-fraction allocations, left to right: track strip, marker column,
    # connector region, table pad, floating label column.
    marker_frac = float(style.get("compact_marker_width", 0.08)) if reserve_marker else 0.0
    connector_frac = (
        float(connector_width)
        if connector_width is not None
        else float(style.get("compact_bridge_width", 0.45))
    )
    table_pad_frac = float(style.get("compact_table_pad", 0.02))
    table_frac = 1.0 - end_x - marker_frac - connector_frac - table_pad_frac
    # Validate before adding any axes, so a rejected layout leaves the figure untouched.
    # connector_frac is a divisor when pads are converted to bridge-local x, so a
    # non-positive region is rejected here rather than producing a degenerate axis.
    if connector_frac <= 0:
        raise ValueError("connector_width must be > 0")
    if span_pads is not None and sum(span_pads) > connector_frac:
        raise ValueError(
            "cluster_span_left_pad + cluster_span_right_pad must be <= connector_width "
            f"({connector_frac}); the leader line would start past the connector region"
        )
    if table_frac <= 0:
        raise ValueError(
            "label-panel track widths/pads and connector_width exceed the available "
            "compact label-panel width; reduce plot_label_bar()/plot_cluster_bar() "
            "width or pads, or reduce connector_width"
        )

    marker_x0 = x0 + end_x * w
    bridge_x0 = marker_x0 + marker_frac * w
    table_x0 = bridge_x0 + (connector_frac + table_pad_frac) * w

    ax_trk = fig.add_axes([track_x0, y0, track_w, h], frameon=False) if tracks else None
    ax_mrk = fig.add_axes([marker_x0, y0, marker_frac * w, h], frameon=False)
    ax_bridge = fig.add_axes([bridge_x0, y0, connector_frac * w, h], frameon=False)
    ax_tbl = fig.add_axes([table_x0, y0, table_frac * w, h], frameon=False)

    all_axes = (
        (ax_mrk, ax_bridge, ax_tbl, ax_trk) if ax_trk is not None else (ax_mrk, ax_bridge, ax_tbl)
    )
    for ax in all_axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    return _CompactGeometry(
        marker=ax_mrk,
        bridge=ax_bridge,
        table=ax_tbl,
        track=ax_trk,
        tracks=tracks,
        connector_frac=connector_frac,
        table_frac=table_frac,
    )


class CompactLabelsRenderer:
    """
    Class for rendering compact radiating cluster labels: a short marker at each cluster's
    true vertical center, a leader line, and a full label in an equally-spaced floating
    label column on the right.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        overrides: Optional[Dict[int, str]] = None,
        cluster_marker: Optional[str] = None,
        label_fields: Optional[Sequence[str]] = ...,  # type: ignore[assignment]
        label_prefix: Optional[str] = "alpha",
        font: Optional[str] = None,
        fontsize: Optional[float] = None,
        color: Optional[str] = None,
        alpha: Optional[float] = None,
        skip_unlabeled: bool = False,
        placeholder_text: Optional[str] = None,
        placeholder_color: Optional[str] = None,
        placeholder_alpha: Optional[float] = None,
        line_shape: Optional[str] = None,
        line_style: Optional[str] = None,
        cluster_span: Optional[str] = None,
        line_start: Optional[str] = None,
        line_end: Optional[str] = None,
        cluster_span_gap: Optional[float] = None,
        cluster_span_color: Optional[str] = None,
        cluster_span_lw: Optional[float] = None,
        cluster_span_alpha: Optional[float] = None,
        cluster_span_cap_width: Optional[float] = None,
        cluster_span_left_pad: Optional[float] = None,
        cluster_span_right_pad: Optional[float] = None,
        label_left_pad: Optional[float] = None,
        connector_width: Optional[float] = None,
        line_color: Optional[str] = None,
        line_lw: Optional[float] = None,
        line_alpha: Optional[float] = None,
        wrap_text: bool = True,
        wrap_width: Optional[int] = None,
        max_words: Optional[int] = None,
        omit_words: Optional[Sequence[str]] = None,
        overflow: str = "wrap",
    ) -> None:
        """
        Initializes the CompactLabelsRenderer instance.

        Args:
            df (pd.DataFrame): Cluster label table with 'cluster', 'label', and optional
                'pval', 'qval', 'score', and 'fe'.

        Kwargs:
            overrides (Optional[Dict[int, str]]): Per-cluster label overrides keyed by cluster id.
                Defaults to None.
            cluster_marker (Optional[str]): Optional identity marker drawn at the source
                (matrix-side) marker column, one of {None, "cid", "alpha"}. Off by default;
                the cluster span/line alone is the pointer, and identity lives with the
                floating label via label_prefix. Defaults to None.
            label_fields (Optional[Sequence[str]]): Fields to include in floating labels: one
                or more of "label", "n", "p", "q", "fe". If None, suppresses base label/stat
                text. Defaults to ("label", "n", "p").
            label_prefix (Optional[str]): Sole owner of the floating-label identity prefix,
                one of {None, "cid", "alpha"}. Defaults to "alpha".
            font (Optional[str]): Font family for floating label text. Defaults to None.
            fontsize (Optional[float]): Font size for floating label text (points).
                Defaults to None.
            color (Optional[str]): Text color for markers and floating labels. Defaults to None.
            alpha (Optional[float]): Text opacity for markers and floating labels.
                Defaults to None.
            skip_unlabeled (bool): Whether to omit clusters without a label entirely.
                Defaults to False.
            placeholder_text (Optional[str]): Text for unlabeled clusters. Defaults to None.
            placeholder_color (Optional[str]): Color override for placeholder labels. Defaults to None.
            placeholder_alpha (Optional[float]): Alpha override for placeholder labels. Defaults to None.
            line_shape (Optional[str]): Leader-line shape, one of {"straight", "curved", "elbow"}.
                Defaults to None.
            line_style (Optional[str]): Leader-line style, one of {"solid", "dashed", "dotted"}.
                Defaults to None.
            cluster_span (Optional[str]): Matrix-side cluster-extent span, one of
                {None, "line"}, mirroring standard plot_cluster_labels(cluster_span=...).
                Defaults to None.
            line_start (Optional[str]): Connector-start point decoration, one of
                {"tick", "round", "none"}, used only when cluster_span is None. Defaults to None.
            line_end (Optional[str]): Table-side endpoint decoration, one of
                {"tick", "arrow", "round", "none"}. Defaults to None.
            cluster_span_gap (Optional[float]): Row units trimmed from each end of a
                cluster_span, clamped to at most half the cluster's span height.
                Defaults to None.
            cluster_span_color (Optional[str]): Span color, independent of line_color.
                Defaults to None.
            cluster_span_lw (Optional[float]): Span line width, independent of line_lw.
                Defaults to None.
            cluster_span_alpha (Optional[float]): Span opacity, independent of line_alpha.
                Defaults to None.
            cluster_span_cap_width (Optional[float]): Optional end-cap width (label-panel
                axes fraction) for the span; 0 draws a bare line, >0 draws caps.
                Defaults to None (style cluster_span_cap_width).
            cluster_span_left_pad (Optional[float]): Space (label-panel axes fraction)
                between the track/marker-side edge and the span. Only applies when
                cluster_span is not None. Defaults to None (style cluster_span_left_pad).
            cluster_span_right_pad (Optional[float]): Space (label-panel axes fraction)
                between the span and the leader-line start. Only applies when
                cluster_span is not None. Defaults to None (style cluster_span_right_pad).
            label_left_pad (Optional[float]): Space (label-panel axes fraction) between
                the floating label column's left edge and the label text. Moves the label
                text only; leader-line geometry (which still ends at the connector region's
                right edge) is unaffected. Defaults to None (style compact_label_left_pad).
            connector_width (Optional[float]): Width (label-panel axes fraction) reserved
                for the connector/leader-line region between the cluster span and the
                floating label column. Defaults to None (style compact_bridge_width).
            line_color (Optional[str]): Leader-line color. Defaults to None.
            line_lw (Optional[float]): Leader-line width. Defaults to None.
            line_alpha (Optional[float]): Leader-line opacity. Defaults to None.
            wrap_text (bool): Whether to wrap long label text. Defaults to True.
            wrap_width (Optional[int]): Characters per wrapped line. Defaults to None.
            max_words (Optional[int]): Maximum words in rendered labels. Defaults to None.
            omit_words (Optional[Sequence[str]]): Words to omit from labels. Defaults to None.
            overflow (str): Truncation mode, one of {"wrap", "ellipsis"}. Defaults to "wrap".

        Raises:
            ValueError: If df is missing required columns, or cluster_marker, label_fields,
                label_prefix, line_shape, line_style, cluster_span, line_start, or line_end
                is unsupported, or if cluster_span_gap, cluster_span_cap_width,
                cluster_span_left_pad, cluster_span_right_pad, or label_left_pad is
                negative, or if connector_width is not > 0.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("cluster_labels must be a pandas DataFrame.")
        if "cluster" not in df.columns or "label" not in df.columns:
            raise ValueError("cluster_labels DataFrame must contain columns: 'cluster', 'label'.")
        if cluster_marker is not None and cluster_marker not in CLUSTER_MARKERS:
            raise ValueError(f"cluster_marker must be one of {[None] + sorted(CLUSTER_MARKERS)}")
        if label_fields is not ... and label_fields is not None:
            if not isinstance(label_fields, (list, tuple)):
                raise TypeError("label_fields must be None or a list/tuple of strings")
            allowed_fields = {"label", "n", "p", "q", "fe"}
            if any(f not in allowed_fields for f in label_fields):
                raise ValueError(f"label_fields may only contain {allowed_fields}")
        if label_prefix not in {None, "cid", "alpha"}:
            raise ValueError("label_prefix must be one of {None, 'cid', 'alpha'}")
        if line_shape is not None and line_shape not in LINE_SHAPES:
            raise ValueError(f"line_shape must be one of {sorted(LINE_SHAPES)}")
        if line_style is not None and line_style not in LINE_STYLES:
            raise ValueError(f"line_style must be one of {sorted(LINE_STYLES)}")
        if cluster_span is not None and cluster_span not in CLUSTER_SPANS:
            raise ValueError(f"cluster_span must be one of {[None] + sorted(CLUSTER_SPANS)}")
        if line_start is not None and line_start not in LINE_STARTS:
            raise ValueError(f"line_start must be one of {sorted(LINE_STARTS)}")
        if line_end is not None and line_end not in LINE_ENDS:
            raise ValueError(f"line_end must be one of {sorted(LINE_ENDS)}")
        if cluster_span_gap is not None and cluster_span_gap < 0:
            raise ValueError("cluster_span_gap must be >= 0")
        if cluster_span_cap_width is not None and cluster_span_cap_width < 0:
            raise ValueError("cluster_span_cap_width must be >= 0")
        if cluster_span_left_pad is not None and cluster_span_left_pad < 0:
            raise ValueError("cluster_span_left_pad must be >= 0")
        if cluster_span_right_pad is not None and cluster_span_right_pad < 0:
            raise ValueError("cluster_span_right_pad must be >= 0")
        if label_left_pad is not None and label_left_pad < 0:
            raise ValueError("label_left_pad must be >= 0")
        if connector_width is not None and connector_width <= 0:
            raise ValueError("connector_width must be > 0")

        self.df = df
        self.overrides = overrides
        self.cluster_marker = cluster_marker
        self.label_fields = (
            label_fields
            if label_fields is ...
            else (None if label_fields is None else tuple(label_fields))
        )
        self.label_prefix = label_prefix
        self.font = font
        self.fontsize = fontsize
        self.color = color
        self.alpha = alpha
        self.skip_unlabeled = skip_unlabeled
        self.placeholder_text = placeholder_text
        self.placeholder_color = placeholder_color
        self.placeholder_alpha = placeholder_alpha
        self.line_shape = line_shape
        self.line_style = line_style
        self.cluster_span = cluster_span
        self.line_start = line_start
        self.line_end = line_end
        self.cluster_span_gap = cluster_span_gap
        self.cluster_span_color = cluster_span_color
        self.cluster_span_lw = cluster_span_lw
        self.cluster_span_alpha = cluster_span_alpha
        self.cluster_span_cap_width = cluster_span_cap_width
        self.cluster_span_left_pad = cluster_span_left_pad
        self.cluster_span_right_pad = cluster_span_right_pad
        self.label_left_pad = label_left_pad
        self.connector_width = connector_width
        self.line_color = line_color
        self.line_lw = line_lw
        self.line_alpha = line_alpha
        self.wrap_text = wrap_text
        self.wrap_width = wrap_width
        self.max_words = max_words
        self.omit_words = omit_words
        self.overflow = overflow

    def render(
        self,
        fig: plt.Figure,
        matrix: Matrix,
        layout: ClusterLayout,
        style: StyleConfig,
        track_layout: Optional[TrackLayoutManager] = None,
        bar_labels_kwargs: Optional[Dict[str, object]] = None,
    ) -> None:
        """
        Renders markers at true cluster centers, leader lines, and an equally-spaced
        floating label column.

        Args:
            fig (plt.Figure): Target figure.
            matrix (Matrix): Matrix object providing row count.
            layout (ClusterLayout): Cluster layout providing `cluster_spans` in dendrogram order.
            style (StyleConfig): Style configuration.

        Kwargs:
            track_layout (Optional[TrackLayoutManager]): Registered label-panel tracks. Both
                row-kind tracks (plot_label_bar()) and cluster-kind tracks (plot_cluster_bar())
                are drawn, in set_label_track_order() order. Defaults to None.
            bar_labels_kwargs (Optional[Dict[str, object]]): Bar title rendering options,
                consumed only when tracks are drawn. Defaults to None.

        Raises:
            ValueError: If track widths/pads and connector_width leave no room for the
                floating label column, or if the cluster-span pads overrun the connector
                region.
        """
        n_rows = matrix.df.shape[0]
        spans: List[Tuple[int, int, int]] = list(layout.cluster_spans)

        override_map = _parse_label_overrides(self.overrides)
        label_map = _build_label_map(self.df, override_map)
        label_fields = self.label_fields if self.label_fields is not ... else style["label_fields"]

        # Resolve the span pads before any axis exists: _setup_compact_axes() checks them
        # against the connector region alongside its own allocation checks, so a rejected
        # layout leaves the figure untouched. The pads only apply when a span is drawn.
        cluster_span_left_pad = (
            self.cluster_span_left_pad
            if self.cluster_span_left_pad is not None
            else style.get("cluster_span_left_pad", 0.0)
        )
        cluster_span_right_pad = (
            self.cluster_span_right_pad
            if self.cluster_span_right_pad is not None
            else style.get("cluster_span_right_pad", 0.01)
        )
        geometry = _setup_compact_axes(
            fig,
            n_rows,
            style,
            track_layout,
            reserve_marker=self.cluster_marker is not None,
            connector_width=self.connector_width,
            span_pads=(
                (cluster_span_left_pad, cluster_span_right_pad)
                if self.cluster_span is not None
                else None
            ),
        )
        ax_mrk, ax_bridge, ax_tbl, ax_trk = (
            geometry.marker,
            geometry.bridge,
            geometry.table,
            geometry.track,
        )
        if ax_trk is not None:
            _render_tracks(
                ax_trk,
                geometry.tracks,
                matrix=matrix,
                row_order=layout.leaf_order,
                spans=spans,
                label_map=label_map,
                style=style,
                bar_labels_kwargs=bar_labels_kwargs,
            )

        text_color = self.color if self.color is not None else style.get("text_color", "black")
        text_alpha = self.alpha if self.alpha is not None else 0.9
        font = self.font if self.font is not None else "Helvetica"
        fontsize = self.fontsize if self.fontsize is not None else style.get("label_fontsize", 9)
        marker_fontsize = style.get("compact_marker_fontsize", fontsize)
        placeholder_text = (
            self.placeholder_text
            if self.placeholder_text is not None
            else style.get("placeholder_text", "—")
        )
        placeholder_color = (
            self.placeholder_color if self.placeholder_color is not None else text_color
        )
        placeholder_alpha = (
            self.placeholder_alpha
            if self.placeholder_alpha is not None
            else style.get("placeholder_alpha", 0.6)
        )
        line_shape = self.line_shape or style.get("compact_line_shape", "straight")
        line_style = self.line_style or style.get("compact_line_style", "solid")
        cluster_span = self.cluster_span
        line_start = self.line_start or style.get("compact_line_start", "tick")
        line_end = self.line_end or style.get("compact_line_end", "tick")
        line_color = self.line_color or style.get("compact_line_color", "#c0562c")
        line_lw = self.line_lw if self.line_lw is not None else style.get("compact_line_lw", 0.9)
        line_alpha = (
            self.line_alpha
            if self.line_alpha is not None
            else style.get("compact_line_alpha", 0.65)
        )
        # Cluster-span styling is independent of connector line_* styling
        # (mirrors the standard-label span, not the leader line it accompanies).
        cluster_span_color = (
            self.cluster_span_color
            or style.get("cluster_span_color", None)
            or style.get("label_sep_color", "gray")
        )
        cluster_span_lw = (
            self.cluster_span_lw
            if self.cluster_span_lw is not None
            else style.get("cluster_span_lw", 1.0)
        )
        cluster_span_alpha = (
            self.cluster_span_alpha
            if self.cluster_span_alpha is not None
            else style.get("cluster_span_alpha", 0.8)
        )
        cluster_span_gap = (
            self.cluster_span_gap
            if self.cluster_span_gap is not None
            else style.get("cluster_span_gap", 0.15)
        )
        cluster_span_cap_width = (
            self.cluster_span_cap_width
            if self.cluster_span_cap_width is not None
            else style.get("cluster_span_cap_width", 0.0)
        )
        label_left_pad = (
            self.label_left_pad
            if self.label_left_pad is not None
            else style.get("compact_label_left_pad", 0.0)
        )
        # Public horizontal values are label-panel axes fractions, matching
        # plot_cluster_labels() and the track strip. Convert them into the bridge and
        # table axes' local [0, 1] frames here, at the renderer boundary, so the drawing
        # helpers below stay axis-local. The panel width cancels, leaving plain ratios.
        connector_frac = geometry.connector_frac
        # Bridge-axis x in [0, 1]: 0.0 is the matrix/marker-side edge, 1.0 is the table
        # side. Pads only apply when cluster_span is active; span_x replaces the
        # hardcoded 0.0 span position, and leader_start_x replaces the leader line's
        # hardcoded 0.0 start.
        span_x = cluster_span_left_pad / connector_frac
        leader_start_x = (cluster_span_left_pad + cluster_span_right_pad) / connector_frac
        span_cap_width_local = cluster_span_cap_width / connector_frac
        label_text_x = label_left_pad / geometry.table_frac

        # Table slots follow dendrogram/top-to-bottom order (layout.cluster_spans order),
        # equally spaced and mapped onto the shared row-index y-range.
        visible_spans = [
            (cid, s, e) for cid, s, e in spans if not (self.skip_unlabeled and cid not in label_map)
        ]
        k = len(visible_spans)
        if k == 0:
            return
        table_slots = compute_equal_slots(k, pitch=1.0) * (n_rows / k) - 0.5

        for i, (cid, s, e) in enumerate(visible_spans):
            y_center = (s + e) / 2.0
            slot_y = table_slots[i]

            resolved = resolve_cluster_label_content(
                cid,
                label_map,
                layout.cluster_sizes.get(cid, None),
                label_fields=label_fields,
                label_prefix=self.label_prefix,
                is_override=cid in override_map,
                placeholder_text=placeholder_text,
                max_words=self.max_words,
                omit_words=self.omit_words,
                wrap_text=self.wrap_text,
                wrap_width=self.wrap_width,
                overflow=self.overflow,
            )
            label_text = resolved.text
            if resolved.is_placeholder:
                label_color = placeholder_color
                label_alpha = placeholder_alpha
            else:
                label_color = text_color
                label_alpha = text_alpha

            # Optional identity marker at the cluster's true vertical center, in
            # row-index space. Off by default: the cluster span/line is the pointer,
            # and identity lives with the floating label via label_prefix.
            if self.cluster_marker is not None:
                marker_text = format_label_prefix(self.cluster_marker, cid).rstrip(".")
                mrk_txt = ax_mrk.text(
                    0.5,
                    y_center,
                    marker_text,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    clip_on=False,
                )
                apply_text_style(
                    mrk_txt,
                    font=font,
                    fontsize=marker_fontsize,
                    color=line_color,
                    alpha=1.0,
                )

            if cluster_span is not None:
                draw_cluster_span(
                    ax_bridge,
                    span_x,
                    s,
                    e,
                    gap=cluster_span_gap,
                    cap_width=span_cap_width_local,
                    color=cluster_span_color,
                    lw=cluster_span_lw,
                    alpha=cluster_span_alpha,
                )
            else:
                _draw_line_start(
                    ax_bridge,
                    y_center,
                    kind=line_start,
                    color=line_color,
                    alpha=line_alpha,
                )
            _draw_leader_line(
                ax_bridge,
                y_center,
                slot_y,
                shape=line_shape,
                linestyle=line_style,
                line_end=line_end,
                color=line_color,
                lw=line_lw,
                alpha=line_alpha,
                x_start=leader_start_x if cluster_span is not None else 0.0,
            )

            # Full label at the equal-pitch table slot. Identity comes solely from
            # label_prefix, already folded into label_text above.
            tbl_txt = ax_tbl.text(
                label_text_x,
                slot_y,
                label_text,
                ha="left",
                va="center",
                fontweight="normal",
                clip_on=False,
            )
            apply_text_style(
                tbl_txt,
                font=font,
                fontsize=fontsize,
                color=label_color,
                alpha=label_alpha,
            )
