"""
himalayas/plot/renderers/compact_labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._cluster_label_data import _build_label_map, _parse_label_overrides
from ._cluster_span import draw_cluster_span
from ._compact_label_types import LINE_SHAPES, LINE_STYLES, SOURCE_ENDS, TARGET_ENDS
from ._label_format import compute_equal_slots, format_label_prefix, resolve_cluster_label_content
from ._text_style import apply_text_style

if TYPE_CHECKING:
    from ..style import StyleConfig
    from ...core.layout import ClusterLayout
    from ...core.matrix import Matrix

# Matplotlib linestyle codes for each LINE_STYLES value.
_MPL_LINESTYLES = {"solid": "-", "dashed": "--", "dotted": ":"}


def _resolve_line_path(
    y0: float,
    y1: float,
    *,
    shape: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resolves an (x, y) path from the marker column (x=0) to the table column (x=1).

    Args:
        y0 (float): Source y-position (true cluster center, row-index space).
        y1 (float): Target y-position (equal-pitch table slot).

    Kwargs:
        shape (str): One of {"straight", "curved", "elbow"}.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (x values, y values) along the path.

    Raises:
        ValueError: If shape is unsupported.
    """
    if shape == "straight":
        return np.array([0.0, 1.0]), np.array([y0, y1])
    if shape == "elbow":
        # Horizontal-then-diagonal-then-horizontal elbow, bent at the midpoint.
        return np.array([0.0, 0.5, 0.5, 1.0]), np.array([y0, y0, y1, y1])
    if shape == "curved":
        xs = np.linspace(0.0, 1.0, 40)
        t = xs  # already in [0, 1]
        t_smooth = t * t * (3.0 - 2.0 * t)
        ys = y0 + (y1 - y0) * t_smooth
        return xs, ys
    raise ValueError(f"line_shape must be one of {sorted(LINE_SHAPES)}, got {shape!r}")


def _draw_source_end(
    ax: plt.Axes,
    y_center: float,
    s: int,
    e: int,
    *,
    kind: str,
    gap: float,
    cap_width: float,
    color: str,
    lw: float,
    alpha: float,
) -> None:
    """
    Draws the matrix-side (x=0) endpoint decoration for one cluster's leader line.

    Args:
        ax (plt.Axes): Bridge axis spanning x in [0, 1].
        y_center (float): True cluster center, (s + e) / 2.
        s (int): Cluster span start (row index).
        e (int): Cluster span end (row index).

    Kwargs:
        kind (str): Source decoration, one of {"tick", "span", "round", "none"}. Callers
            validate this against SOURCE_ENDS before rendering.
        gap (float): Row units trimmed from each end of a "span" bracket. See
            draw_cluster_span for clamping behavior.
        cap_width (float): Bracket cap width (axes fraction) for kind="span".
        color (str): Marker/line color.
        lw (float): Line width.
        alpha (float): Opacity.
    """
    if kind == "none":
        return
    if kind == "tick":
        ax.plot([0.0], [y_center], marker="|", color=color, markersize=5, alpha=alpha)
        return
    if kind == "round":
        ax.plot([0.0], [y_center], marker="o", color=color, markersize=4, alpha=alpha)
        return
    draw_cluster_span(ax, 0.0, s, e, gap=gap, cap_width=cap_width, color=color, lw=lw, alpha=alpha)


def _draw_leader_line(
    ax: plt.Axes,
    y0: float,
    y1: float,
    *,
    shape: str,
    linestyle: str,
    target_end: str,
    color: str,
    lw: float,
    alpha: float,
) -> None:
    """
    Draws one leader line from a marker's true center to its table slot, with an
    optional decoration at the table-side (target) end.

    Args:
        ax (plt.Axes): Bridge axis spanning x in [0, 1].
        y0 (float): Source y-position (marker side).
        y1 (float): Target y-position (table side).

    Kwargs:
        shape (str): Line shape, one of {"straight", "curved", "elbow"}.
        linestyle (str): Matplotlib linestyle, one of {"solid", "dashed", "dotted"}.
        target_end (str): Target-end decoration, one of {"tick", "arrow", "round", "none"}.
            Callers validate this against TARGET_ENDS before rendering.
        color (str): Line color.
        lw (float): Line width.
        alpha (float): Line opacity.
    """
    xs, ys = _resolve_line_path(y0, y1, shape=shape)
    mpl_linestyle = _MPL_LINESTYLES.get(linestyle, "-")
    solid_capstyle = "round" if target_end == "round" else "butt"
    if target_end == "arrow":
        ax.annotate(
            "",
            xy=(1.0, y1),
            xytext=(xs[-2], ys[-2]) if len(xs) > 1 else (0.0, y0),
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
        if target_end not in {"none", "round"}:
            ax.plot([1.0], [y1], marker="|", color=color, markersize=5, alpha=alpha)


def _setup_compact_axes(
    fig: plt.Figure,
    n_rows: int,
    style: StyleConfig,
) -> Tuple[plt.Axes, plt.Axes, plt.Axes]:
    """
    Creates the marker, bridge, and table sub-axes for the compact-label panel.

    Args:
        fig (plt.Figure): Target figure.
        n_rows (int): Number of matrix rows.
        style (StyleConfig): Style configuration.

    Returns:
        Tuple[plt.Axes, plt.Axes, plt.Axes]: (marker axis, bridge axis, table axis).
    """
    compact_axes = style.get("compact_axes", None)
    x0, y0, w, h = compact_axes if compact_axes is not None else style["label_axes"]
    marker_w = float(style.get("compact_marker_width", 0.08)) * w
    bridge_w = float(style.get("compact_bridge_width", 0.45)) * w
    table_pad = float(style.get("compact_table_pad", 0.02)) * w
    table_x0 = x0 + marker_w + bridge_w + table_pad
    table_w = max(w - marker_w - bridge_w - table_pad, 0.01)

    ax_mrk = fig.add_axes([x0, y0, marker_w, h], frameon=False)
    ax_bridge = fig.add_axes([x0 + marker_w, y0, bridge_w, h], frameon=False)
    ax_tbl = fig.add_axes([table_x0, y0, table_w, h], frameon=False)

    for ax in (ax_mrk, ax_bridge, ax_tbl):
        ax.set_xlim(0, 1)
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    return ax_mrk, ax_bridge, ax_tbl


class CompactLabelsRenderer:
    """
    Class for rendering compact radiating cluster labels: a short marker at each cluster's
    true vertical center, a leader line, and a full label in an equally-spaced right-side
    table.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        overrides: Optional[Dict[int, str]] = None,
        marker_prefix: str = "alpha",
        label_fields: Optional[Sequence[str]] = ...,  # type: ignore[assignment]
        label_prefix: Optional[str] = None,
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
        source_end: Optional[str] = None,
        target_end: Optional[str] = None,
        source_gap: Optional[float] = None,
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
            marker_prefix (str): Marker text mode, one of {"cid", "alpha"}. Defaults to "alpha".
            label_fields (Optional[Sequence[str]]): Fields to include in table labels: one or
                more of "label", "n", "p", "q", "fe". If None, suppresses base label/stat text.
                Defaults to ("label", "n", "p").
            label_prefix (Optional[str]): Prefix prepended to table label text, one of
                {None, "cid", "alpha"}. Defaults to None.
            font (Optional[str]): Font family for table label text. Defaults to None.
            fontsize (Optional[float]): Font size for table label text (points). Defaults to None.
            color (Optional[str]): Text color for markers and table labels. Defaults to None.
            alpha (Optional[float]): Text opacity for markers and table labels. Defaults to None.
            skip_unlabeled (bool): Whether to omit clusters without a label entirely.
                Defaults to False.
            placeholder_text (Optional[str]): Text for unlabeled clusters. Defaults to None.
            placeholder_color (Optional[str]): Color override for placeholder labels. Defaults to None.
            placeholder_alpha (Optional[float]): Alpha override for placeholder labels. Defaults to None.
            line_shape (Optional[str]): Leader-line shape, one of {"straight", "curved", "elbow"}.
                Defaults to None.
            line_style (Optional[str]): Leader-line style, one of {"solid", "dashed", "dotted"}.
                Defaults to None.
            source_end (Optional[str]): Matrix-side endpoint decoration, one of
                {"tick", "span", "round", "none"}. Defaults to None.
            target_end (Optional[str]): Table-side endpoint decoration, one of
                {"tick", "arrow", "round", "none"}. Defaults to None.
            source_gap (Optional[float]): Row units trimmed from each end of a
                source_end="span" bracket, clamped to at most half the cluster's span
                height. Defaults to None.
            line_color (Optional[str]): Leader-line color. Defaults to None.
            line_lw (Optional[float]): Leader-line width. Defaults to None.
            line_alpha (Optional[float]): Leader-line opacity. Defaults to None.
            wrap_text (bool): Whether to wrap long label text. Defaults to True.
            wrap_width (Optional[int]): Characters per wrapped line. Defaults to None.
            max_words (Optional[int]): Maximum words in rendered labels. Defaults to None.
            omit_words (Optional[Sequence[str]]): Words to omit from labels. Defaults to None.
            overflow (str): Truncation mode, one of {"wrap", "ellipsis"}. Defaults to "wrap".

        Raises:
            ValueError: If df is missing required columns, or marker_prefix, label_fields,
                label_prefix, line_shape, line_style, source_end, target_end, or source_gap
                is unsupported.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("cluster_labels must be a pandas DataFrame.")
        if "cluster" not in df.columns or "label" not in df.columns:
            raise ValueError("cluster_labels DataFrame must contain columns: 'cluster', 'label'.")
        if marker_prefix not in {"cid", "alpha"}:
            raise ValueError("marker_prefix must be one of {'cid', 'alpha'}")
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
        if source_end is not None and source_end not in SOURCE_ENDS:
            raise ValueError(f"source_end must be one of {sorted(SOURCE_ENDS)}")
        if target_end is not None and target_end not in TARGET_ENDS:
            raise ValueError(f"target_end must be one of {sorted(TARGET_ENDS)}")
        if source_gap is not None and source_gap < 0:
            raise ValueError("source_gap must be >= 0")

        self.df = df
        self.overrides = overrides
        self.marker_prefix = marker_prefix
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
        self.source_end = source_end
        self.target_end = target_end
        self.source_gap = source_gap
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
    ) -> None:
        """
        Renders markers at true cluster centers, leader lines, and an equally-spaced label table.

        Args:
            fig (plt.Figure): Target figure.
            matrix (Matrix): Matrix object providing row count.
            layout (ClusterLayout): Cluster layout providing `cluster_spans` in dendrogram order.
            style (StyleConfig): Style configuration.
        """
        n_rows = matrix.df.shape[0]
        spans: List[Tuple[int, int, int]] = list(layout.cluster_spans)

        override_map = _parse_label_overrides(self.overrides)
        label_map = _build_label_map(self.df, override_map)
        label_fields = self.label_fields if self.label_fields is not ... else style["label_fields"]

        ax_mrk, ax_bridge, ax_tbl = _setup_compact_axes(fig, n_rows, style)

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
        source_end = self.source_end or style.get("compact_source_end", "tick")
        target_end = self.target_end or style.get("compact_target_end", "tick")
        source_gap = (
            self.source_gap
            if self.source_gap is not None
            else style.get("compact_source_gap", 0.15)
        )
        source_cap_width = style.get("compact_source_cap_width", 0.15)
        line_color = self.line_color or style.get("compact_line_color", "#c0562c")
        line_lw = self.line_lw if self.line_lw is not None else style.get("compact_line_lw", 0.9)
        line_alpha = (
            self.line_alpha
            if self.line_alpha is not None
            else style.get("compact_line_alpha", 0.65)
        )

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
            marker_text = format_label_prefix(self.marker_prefix, cid).rstrip(".")

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

            # Marker at the cluster's true vertical center, in row-index space.
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

            _draw_source_end(
                ax_bridge,
                y_center,
                s,
                e,
                kind=source_end,
                gap=source_gap,
                cap_width=source_cap_width,
                color=line_color,
                lw=line_lw,
                alpha=line_alpha,
            )
            _draw_leader_line(
                ax_bridge,
                y_center,
                slot_y,
                shape=line_shape,
                linestyle=line_style,
                target_end=target_end,
                color=line_color,
                lw=line_lw,
                alpha=line_alpha,
            )

            # Full label at the equal-pitch table slot.
            tbl_txt = ax_tbl.text(
                0.0,
                slot_y,
                f"{marker_text}.  {label_text}",
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
