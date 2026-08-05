"""
himalayas/plot/renderers/cluster_labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import (
    Any,
    Optional,
    Dict,
    Tuple,
    List,
    Sequence,
    TYPE_CHECKING,
)

import matplotlib.pyplot as plt
import pandas as pd

from ._cluster_label_data import _build_label_map, _parse_label_overrides
from ._cluster_label_types import ClusterLabelStats
from ._cluster_span import draw_cluster_span
from ._compact_label_types import CLUSTER_SPANS
from ._label_format import resolve_cluster_label_content
from ._text_style import apply_text_style
from ._track_rendering import TrackSpec, _render_tracks

if TYPE_CHECKING:
    from ..style import StyleConfig
    from ..track_layout import TrackLayoutManager
    from ...core.layout import ClusterLayout
    from ...core.matrix import Matrix


def _resolve_labels_and_layout(
    df: pd.DataFrame,
    kwargs: Dict[str, Any],
    fig: plt.Figure,
    matrix: Matrix,
    layout: ClusterLayout,
    style: StyleConfig,
    track_layout: TrackLayoutManager,
) -> Tuple[
    plt.Axes,
    float,
    List[TrackSpec],
    List[Tuple[int, int, int]],
    Dict[int, int],
    Dict[int, ClusterLabelStats],
    Dict[int, str],
    float,
    float,
    str,
    float,
    Optional[Tuple[str, ...]],
    bool,
    Optional[str],
    Optional[float],
]:
    """
    Resolves label data, overrides, axis layout, and text styling.

    Args:
        df (pd.DataFrame): Cluster label table with 'cluster', 'label', and optional
            'pval', 'qval', 'score', and 'fe'.
        kwargs (Dict[str, Any]): Renderer keyword arguments.
        fig (plt.Figure): Target figure.
        matrix (Matrix): Matrix object providing row count.
        layout (ClusterLayout): Cluster layout providing `cluster_spans`.
        style (StyleConfig): Style configuration.
        track_layout (TrackLayoutManager): Track layout manager.

    Returns:
        Tuple containing:
            - plt.Axes: Target label axis.
            - float: X-position for label text.
            - List[TrackSpec]: Resolved track specifications.
            - List[Tuple[int, int, int]]: Iterable of (cluster_id, start, end).
            - Dict[int, int]: Mapping cluster_id -> size.
            - Dict[int, ClusterLabelStats]: Mapping cluster_id -> (label, pval, qval, score, fe).
            - Dict[int, str]: Mapping cluster_id -> validated override label.
            - float: Minimum x-position for separator lines.
            - float: Maximum x-position for separator lines.
            - str: Font name for label text.
            - float: Font size for label text.
            - Optional[Tuple[str, ...]]: Fields to display in labels.
            - bool: Whether to skip unlabeled clusters.
            - Optional[str]: Label prefix mode.
            - Optional[float]: Span/bracket centerline x-position, or None if cluster_span
              is not active.

    Raises:
        TypeError: If inputs have invalid types.
        ValueError: If required columns are missing or invalid.
    """
    # Validate cluster_labels DataFrame.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("cluster_labels must be a pandas DataFrame.")
    if "cluster" not in df.columns or "label" not in df.columns:
        raise ValueError("cluster_labels DataFrame must contain columns: 'cluster', 'label'.")

    # Parse overrides and build label map.
    overrides = kwargs.get("overrides", None)
    override_map = _parse_label_overrides(overrides)
    label_map = _build_label_map(df, override_map)

    # Resolve spans, sizes, and label axis layout.
    spans = layout.cluster_spans
    cluster_sizes = layout.cluster_sizes
    ax_lab, label_text_x, tracks, end_x = _setup_label_axis(fig, matrix, style, track_layout)

    # Cluster-abreast span/bracket: opt-in, positioned from end_x directly so placement
    # never depends on label_bar_pad. Padding is measured from the span centerline.
    cluster_span = kwargs.get("cluster_span", None)
    span_x: Optional[float] = None
    if cluster_span is not None:
        left_pad = kwargs.get("cluster_span_left_pad", style.get("cluster_span_left_pad", 0.0))
        right_pad = kwargs.get("cluster_span_right_pad", style.get("cluster_span_right_pad", 0.01))
        if left_pad < 0:
            raise ValueError("cluster_span_left_pad must be >= 0")
        if right_pad < 0:
            raise ValueError("cluster_span_right_pad must be >= 0")
        span_x = max(end_x + left_pad, 0.0)
        label_text_x = span_x + right_pad

    # Resolve separator line positions.
    sep_xmin = kwargs.get("label_sep_xmin", style.get("label_sep_xmin"))
    sep_xmax = kwargs.get("label_sep_xmax", style.get("label_sep_xmax"))
    if sep_xmin is None:
        sep_xmin = label_text_x
    if sep_xmax is None:
        sep_xmax = 1.0
    sep_xmin = float(sep_xmin)
    sep_xmax = float(sep_xmax)
    if sep_xmin > sep_xmax:
        sep_xmin, sep_xmax = sep_xmax, sep_xmin
    # Resolve text style options.
    font = kwargs.get("font", "Helvetica")
    fontsize = kwargs.get("fontsize", style.get("label_fontsize", 9))
    skip_unlabeled = kwargs.get("skip_unlabeled", False)
    label_fields = kwargs.get("label_fields", style["label_fields"])
    label_prefix = kwargs.get("label_prefix", None)

    # Validate label_fields
    if label_fields is not None and not isinstance(label_fields, (list, tuple)):
        raise TypeError("label_fields must be None or a list/tuple of strings")
    allowed_fields = {"label", "n", "p", "q", "fe"}
    if label_fields is not None and any(f not in allowed_fields for f in label_fields):
        raise ValueError(f"label_fields may only contain {allowed_fields}")
    if label_prefix not in {None, "cid", "alpha"}:
        raise ValueError("label_prefix must be one of {None, 'cid', 'alpha'}")
    resolved_label_fields = None if label_fields is None else tuple(label_fields)

    return (
        ax_lab,
        label_text_x,
        tracks,
        spans,
        cluster_sizes,
        label_map,
        override_map,
        sep_xmin,
        sep_xmax,
        font,
        fontsize,
        resolved_label_fields,
        bool(skip_unlabeled),
        label_prefix,
        span_x,
    )


def _render_cluster_text_and_separators(
    ax_lab: plt.Axes,
    *,
    spans: Sequence[Tuple[int, int, int]],
    cluster_sizes: Dict[int, int],
    label_map: Dict[int, ClusterLabelStats],
    override_map: Dict[int, str],
    label_text_x: float,
    span_x: Optional[float],
    sep_xmin: float,
    sep_xmax: float,
    font: str,
    fontsize: float,
    label_fields: Optional[Tuple[str, ...]],
    label_prefix: Optional[str],
    skip_unlabeled: bool,
    kwargs: Dict[str, Any],
    style: StyleConfig,
) -> None:
    """
    Renders cluster labels and separator lines.

    Args:
        ax_lab (plt.Axes): Target label axis.

    Kwargs:
        spans (Sequence[Tuple[int, int, int]]): Iterable of (cluster_id, start, end).
        cluster_sizes (Dict[int, int]): Mapping cluster_id -> size.
        label_map (Dict[int, ClusterLabelStats]): Mapping cluster_id -> (label, pval, qval, score, fe).
        override_map (Dict[int, str]): Mapping cluster_id -> validated override label.
        label_text_x (float): X-position for label text.
        span_x (Optional[float]): Span/bracket centerline x-position, already resolved by
            _resolve_labels_and_layout from end_x + cluster_span_left_pad. None if
            cluster_span is not active.
        sep_xmin (float): Minimum x-position for separator lines.
        sep_xmax (float): Maximum x-position for separator lines.
        font (str): Font name for label text.
        fontsize (float): Font size for label text.
        label_fields (Optional[Tuple[str, ...]]): Fields to display in labels.
        label_prefix (Optional[str]): Label prefix mode.
        skip_unlabeled (bool): Whether to skip unlabeled clusters.
        kwargs (Dict[str, Any]): Additional rendering options.
        style (StyleConfig): Style configuration.
    """
    max_words = kwargs.get("max_words", None)
    omit_words = kwargs.get("omit_words", style.get("label_omit_words", None))
    wrap_text = kwargs.get("wrap_text", True)
    wrap_width = kwargs.get("wrap_width", style.get("label_wrap_width", None))
    overflow = kwargs.get("overflow", "wrap")

    # Cluster-abreast span/bracket: opt-in, placed just left of label text.
    cluster_span = kwargs.get("cluster_span", None)
    if cluster_span is not None:
        if cluster_span not in CLUSTER_SPANS:
            raise ValueError(f"cluster_span must be one of {[None] + sorted(CLUSTER_SPANS)}")
        span_color = kwargs.get(
            "cluster_span_color", style.get("cluster_span_color", None)
        ) or style.get("label_sep_color", "gray")
        span_lw = kwargs.get("cluster_span_lw", style.get("cluster_span_lw", 1.0))
        span_alpha = kwargs.get("cluster_span_alpha", style.get("cluster_span_alpha", 0.8))
        span_gap = kwargs.get("cluster_span_gap", style.get("cluster_span_gap", 0.15))
        if span_gap < 0:
            raise ValueError("cluster_span_gap must be >= 0")
        span_cap_width = (
            kwargs.get("cluster_span_cap_width", style.get("cluster_span_cap_width", 0.006))
            if cluster_span == "bracket"
            else 0.0
        )
        if span_cap_width < 0:
            raise ValueError("cluster_span_cap_width must be >= 0")

    for cid, s, e in spans:
        y_center = (s + e) / 2.0
        if cid not in label_map:
            if skip_unlabeled:
                continue
            if label_fields is None and label_prefix is None:
                continue
        resolved = resolve_cluster_label_content(
            cid,
            label_map,
            cluster_sizes.get(cid, None),
            label_fields=label_fields,
            label_prefix=label_prefix,
            is_override=cid in override_map,
            placeholder_text=kwargs.get("placeholder_text", style["placeholder_text"]),
            max_words=max_words,
            omit_words=omit_words,
            wrap_text=wrap_text,
            wrap_width=wrap_width,
            overflow=overflow,
        )
        text = resolved.text
        if resolved.is_placeholder:
            text_color = kwargs.get(
                "placeholder_color", kwargs.get("color", style["placeholder_color"])
            )
            text_alpha = kwargs.get(
                "placeholder_alpha", kwargs.get("alpha", style["placeholder_alpha"])
            )
        else:
            text_color = kwargs.get("color", style.get("text_color", "black"))
            text_alpha = kwargs.get("alpha", 0.9)
        if cluster_span is not None:
            draw_cluster_span(
                ax_lab,
                span_x,
                s,
                e,
                gap=span_gap,
                cap_width=span_cap_width,
                color=span_color,
                lw=span_lw,
                alpha=span_alpha,
            )
        # Draw label text and optional separator line.
        txt = ax_lab.text(
            label_text_x,
            y_center,
            text,
            va="center",
            ha="left",
            fontweight="normal",
            clip_on=False,
        )
        apply_text_style(txt, font=font, fontsize=fontsize, color=text_color, alpha=text_alpha)
        if s > 0:
            sep_color = kwargs.get("label_sep_color", style["label_sep_color"])
            sep_lw = kwargs.get("label_sep_lw", style["label_sep_lw"])
            sep_alpha = kwargs.get("label_sep_alpha", style["label_sep_alpha"])
            # Drawn as an explicit line (not axhline) so xmin/xmax may extend past
            # the label axis' [0, 1] range; clip_on=False lets that overshoot show.
            ax_lab.plot(
                [sep_xmin, sep_xmax],
                [s - 0.5, s - 0.5],
                color=sep_color,
                linewidth=sep_lw,
                alpha=sep_alpha,
                zorder=0,
                clip_on=False,
            )


def _setup_label_axis(
    fig: plt.Figure,
    matrix: Matrix,
    style: StyleConfig,
    track_layout: TrackLayoutManager,
) -> Tuple[plt.Axes, float, List[TrackSpec], float]:
    """
    Creates and configures the label axis and computes track layout. Initializes the label panel,
    draws the gutter, and resolves track x-positions.

    Args:
        fig (plt.Figure): Target figure.
        matrix (Matrix): Matrix object providing row count.
        style (StyleConfig): Style configuration.
        track_layout (TrackLayoutManager): Track layout manager.

    Returns:
        Tuple[plt.Axes, float, List[TrackSpec], float]: (label axis, text x-position,
        resolved tracks, x-position immediately after the track/gutter region).
    """
    n_rows = matrix.df.shape[0]
    # Set up label axis.
    label_axes = style["label_axes"]
    ax_lab = fig.add_axes(label_axes, frameon=False)
    ax_lab.set_xlim(0, 1)
    ax_lab.set_ylim(-0.5, n_rows - 0.5)  # align with matrix row indices.
    ax_lab.invert_yaxis()
    ax_lab.set_xticks([])
    ax_lab.set_yticks([])
    # Set up label gutter.
    gutter_w = style["label_gutter_width"]
    gutter_color = style["label_gutter_color"]
    ax_lab.add_patch(
        plt.Rectangle(
            (0.0, -0.5),
            gutter_w,
            n_rows,
            facecolor=gutter_color,
            edgecolor="none",
            zorder=0,
        )
    )
    # Compute track layout.
    label_text_pad = style.get("label_bar_pad", 0.01)
    base_x = style["label_x"]
    track_layout.compute_layout(base_x, gutter_w)
    tracks = track_layout.get_tracks()
    end_x = track_layout.get_end_x()
    if end_x is None:
        end_x = base_x + gutter_w
    label_text_x = end_x + label_text_pad

    return ax_lab, label_text_x, tracks, end_x


class ClusterLabelsRenderer:
    """
    Class for rendering the cluster label panel with tracks and annotations.
    """

    def __init__(self, df: pd.DataFrame, **kwargs: Any) -> None:
        """
        Initializes the ClusterLabelsRenderer instance.

        Args:
            df (pd.DataFrame): Cluster label table.

        Kwargs:
            **kwargs: Renderer keyword arguments. Defaults to {}.
        """
        self.df = df
        self.kwargs = dict(kwargs)

    def render(
        self,
        fig: plt.Figure,
        matrix: Matrix,
        layout: ClusterLayout,
        style: StyleConfig,
        track_layout: TrackLayoutManager,
        bar_labels_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Renders the cluster label panel with tracks and annotations. Coordinates override
        handling, layout, track rendering, and label drawing.

        Args:
            fig (plt.Figure): Target figure.
            matrix (Matrix): Matrix object.
            layout (ClusterLayout): Cluster layout.
            style (StyleConfig): Style configuration.
            track_layout (TrackLayoutManager): Track layout manager.
            bar_labels_kwargs (Optional[Dict[str, Any]]): Bar title rendering options. Defaults to None.
        """
        df = self.df
        kwargs = self.kwargs
        # Resolve labels, layout, and styling.
        (
            ax_lab,
            label_text_x,
            tracks,
            spans,
            cluster_sizes,
            label_map,
            override_map,
            sep_xmin,
            sep_xmax,
            font,
            fontsize,
            label_fields,
            skip_unlabeled,
            label_prefix,
            span_x,
        ) = _resolve_labels_and_layout(
            df,
            kwargs,
            fig,
            matrix,
            layout,
            style,
            track_layout,
        )
        # Render tracks and cluster labels/separators.
        row_order = layout.leaf_order
        _render_tracks(
            ax_lab,
            tracks,
            matrix=matrix,
            row_order=row_order,
            spans=spans,
            label_map=label_map,
            style=style,
            bar_labels_kwargs=bar_labels_kwargs,
        )
        _render_cluster_text_and_separators(
            ax_lab,
            spans=spans,
            cluster_sizes=cluster_sizes,
            label_map=label_map,
            override_map=override_map,
            label_text_x=label_text_x,
            span_x=span_x,
            sep_xmin=sep_xmin,
            sep_xmax=sep_xmax,
            font=font,
            fontsize=fontsize,
            label_fields=label_fields,
            label_prefix=label_prefix,
            skip_unlabeled=skip_unlabeled,
            kwargs=kwargs,
            style=style,
        )
