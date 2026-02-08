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
    Union,
    TypedDict,
    Callable,
    TYPE_CHECKING,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._label_format import apply_label_text_policy, collect_label_stats, compose_label_text

if TYPE_CHECKING:
    from ..style import StyleConfig
    from ..track_layout import TrackLayoutManager
    from ...core.layout import ClusterLayout
    from ...core.matrix import Matrix


class OverrideInput(TypedDict, total=False):
    """
    Type class for input per-cluster label override specifications.
    """

    label: str
    hide_stats: bool
    pval: Optional[float]


class OverrideSpec(TypedDict, total=False):
    """
    Type class for normalized per-cluster label override specifications.
    """

    label: str
    hide_stats: bool
    pval: Optional[float]


class TrackSpec(TypedDict, total=False):
    """
    Type class for resolved label track specifications.
    """

    name: str
    kind: str
    renderer: Callable[..., Any]
    left_pad: float
    width: float
    right_pad: float
    enabled: bool
    payload: Dict[str, Any]
    x0: float
    x1: float


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
    Dict[int, Tuple[str, Optional[float]]],
    Dict[int, OverrideSpec],
    float,
    float,
    str,
    float,
    Tuple[str, ...],
    bool,
]:
    """
    Resolves label data, overrides, axis layout, and text styling.

    Args:
        df (pd.DataFrame): Cluster label table with 'cluster', 'label', and optional 'pval'.
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
            - Dict[int, Tuple[str, Optional[float]]]: Mapping cluster_id -> (label, pval).
            - Dict[int, OverrideSpec]: Mapping cluster_id -> override dict.
            - float: Minimum x-position for separator lines.
            - float: Maximum x-position for separator lines.
            - str: Font name for label text.
            - float: Font size for label text.
            - Tuple[str, ...]: Fields to display in labels.
            - bool: Whether to skip unlabeled clusters.

    Raises:
        TypeError: If inputs have invalid types.
        ValueError: If required columns are missing or invalid.
    """
    # Validate cluster_labels DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("cluster_labels must be a pandas DataFrame.")
    if "cluster" not in df.columns or "label" not in df.columns:
        raise ValueError("cluster_labels DataFrame must contain columns: 'cluster', 'label'.")

    # Parse overrides and build label map
    overrides = kwargs.get("overrides", None)
    override_map = _parse_label_overrides(overrides)
    label_map = _build_label_map(df, override_map)

    # Resolve spans, sizes, and label axis layout
    spans = layout.cluster_spans
    cluster_sizes = layout.cluster_sizes
    ax_lab, label_text_x, tracks = _setup_label_axis(fig, matrix, style, track_layout, kwargs)
    # Resolve separator line positions
    sep_xmin = kwargs.get("label_sep_xmin", style.get("label_sep_xmin"))
    sep_xmax = kwargs.get("label_sep_xmax", style.get("label_sep_xmax"))
    if sep_xmin is None:
        sep_xmin = label_text_x
    if sep_xmax is None:
        sep_xmax = 1.0
    sep_xmin = float(np.clip(sep_xmin, 0.0, 1.0))
    sep_xmax = float(np.clip(sep_xmax, 0.0, 1.0))
    if sep_xmin > sep_xmax:
        sep_xmin, sep_xmax = sep_xmax, sep_xmin
    # Resolve text style options
    font = kwargs.get("font", "Helvetica")
    fontsize = kwargs.get("fontsize", style.get("label_fontsize", 9))
    skip_unlabeled = kwargs.get("skip_unlabeled", False)
    label_fields = kwargs.get("label_fields", style["label_fields"])

    # Validate label_fields
    if not isinstance(label_fields, (list, tuple)):
        raise TypeError("label_fields must be a list or tuple of strings")
    allowed_fields = {"label", "n", "p"}
    if any(f not in allowed_fields for f in label_fields):
        raise ValueError(f"label_fields may only contain {allowed_fields}")

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
        tuple(label_fields),
        bool(skip_unlabeled),
    )


def _render_tracks(
    ax_lab: plt.Axes,
    tracks: List[TrackSpec],
    *,
    matrix: Matrix,
    row_order: np.ndarray,
    spans: Sequence[Tuple[int, int, int]],
    label_map: Dict[int, Tuple[str, Optional[float]]],
    style: StyleConfig,
    bar_labels_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Renders all row-level and cluster-level tracks, and optional bar titles.

    Args:
        ax_lab (plt.Axes): Target label axis.
        tracks (List[TrackSpec]): List of track specifications.

    Kwargs:
        matrix (Matrix): Data matrix.
        row_order (np.ndarray): Row ordering indices.
        spans (Sequence[Tuple[int, int, int]]): Iterable of (cluster_id, start, end).
        label_map (Dict[int, Tuple[str, Optional[float]]]): Mapping cluster_id -> (label, pval).
        style (StyleConfig): Style configuration.
        bar_labels_kwargs (Optional[Dict[str, Any]]): Bar title rendering options. Defaults to None.
    """
    # Render track content: data tracks and cluster-level tracks
    for track in tracks:
        if track["kind"] == "row":
            track["renderer"](
                ax_lab,
                track["x0"],
                track["width"],
                track["payload"],
                matrix,
                row_order,
                style,
            )
    for track in tracks:
        if track["kind"] == "cluster":
            track["renderer"](
                ax_lab,
                track["x0"],
                track["width"],
                track["payload"],
                spans,
                label_map,
                style,
            )

    # Render optional bar titles
    if bar_labels_kwargs is None:
        return

    # Render bar titles beneath tracks
    bar_pad_pts = bar_labels_kwargs.get("pad", 2)
    bar_rotation = bar_labels_kwargs.get("rotation", 0)
    for track in tracks:
        title = track.get("payload", {}).get("title", None)
        if not title:
            continue
        x_center = (track.get("x0", 0.0) + track.get("x1", 0.0)) / 2.0
        text_kwargs = {
            "font": bar_labels_kwargs.get("font", "Helvetica"),
            "fontsize": bar_labels_kwargs.get("fontsize", 10),
            "color": bar_labels_kwargs.get(
                "color",
                style.get("text_color", "black"),
            ),
            "alpha": bar_labels_kwargs.get("alpha", 1.0),
        }
        ax_lab.annotate(
            title,
            xy=(x_center, 0.0),
            xycoords=ax_lab.transAxes,
            xytext=(0, -bar_pad_pts),
            textcoords="offset points",
            ha="center",
            va="top",
            **text_kwargs,
            rotation=bar_rotation,
            clip_on=False,
        )


def _render_cluster_text_and_separators(
    ax_lab: plt.Axes,
    *,
    spans: Sequence[Tuple[int, int, int]],
    cluster_sizes: Dict[int, int],
    label_map: Dict[int, Tuple[str, Optional[float]]],
    override_map: Dict[int, OverrideSpec],
    label_text_x: float,
    sep_xmin: float,
    sep_xmax: float,
    font: str,
    fontsize: float,
    label_fields: Tuple[str, ...],
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
        label_map (Dict[int, Tuple[str, Optional[float]]]): Mapping cluster_id -> (label, pval).
        override_map (Dict[int, OverrideSpec]): Mapping cluster_id -> override dict.
        label_text_x (float): X-position for label text.
        sep_xmin (float): Minimum x-position for separator lines.
        sep_xmax (float): Maximum x-position for separator lines.
        font (str): Font name for label text.
        fontsize (float): Font size for label text.
        label_fields (Tuple[str, ...]): Fields to display in labels.
        skip_unlabeled (bool): Whether to skip unlabeled clusters.
        kwargs (Dict[str, Any]): Additional rendering options.
        style (StyleConfig): Style configuration.
    """
    for cid, s, e in spans:
        y_center = (s + e) / 2.0
        # Choose placeholder or formatted label text for the cluster
        if cid not in label_map:
            if skip_unlabeled:
                continue
            text = kwargs.get("placeholder_text", style["placeholder_text"])
            text_kwargs = {
                "font": font,
                "fontsize": fontsize,
                "color": kwargs.get(
                    "color",
                    kwargs.get("placeholder_color", style["placeholder_color"]),
                ),
                "alpha": kwargs.get("alpha", style["placeholder_alpha"]),
            }
        else:
            label, pval = label_map[cid]
            n_members = cluster_sizes.get(cid, None)
            text = _format_cluster_label(
                label,
                pval,
                n_members,
                override_map.get(cid),
                label_fields=label_fields,
                kwargs=kwargs,
                style=style,
            )
            text_kwargs = {
                "font": font,
                "fontsize": fontsize,
                "color": kwargs.get("color", style.get("text_color", "black")),
                "alpha": kwargs.get("alpha", 0.9),
            }
        # Draw label text and optional separator line
        ax_lab.text(
            label_text_x,
            y_center,
            text,
            va="center",
            ha="left",
            **text_kwargs,
            fontweight="normal",
            clip_on=False,
        )
        if s > 0:
            sep_color = kwargs.get("label_sep_color", style["label_sep_color"])
            sep_lw = kwargs.get("label_sep_lw", style["label_sep_lw"])
            sep_alpha = kwargs.get("label_sep_alpha", style["label_sep_alpha"])
            ax_lab.axhline(
                s - 0.5,
                xmin=sep_xmin,
                xmax=sep_xmax,
                color=sep_color,
                linewidth=sep_lw,
                alpha=sep_alpha,
                zorder=0,
            )


def _parse_label_overrides(
    overrides: Optional[Dict[int, Union[str, OverrideInput]]] = None,
) -> Dict[int, OverrideSpec]:
    """
    Normalizes and validates per-cluster label overrides. Accepts string or dict overrides and
    enforces allowed keys and precedence rules.

    Args:
        overrides (Dict[int, Union[str, OverrideInput]] | None): Mapping cluster_id -> label
            string or override dict. Defaults to None.

    Returns:
        Dict[int, OverrideSpec]: Normalized override map keyed by cluster id.

    Raises:
        TypeError: If overrides or entries have invalid types.
        ValueError: If unknown keys are provided.
    """
    if overrides is None:
        return {}
    # Validation
    if not isinstance(overrides, dict):
        raise TypeError("overrides must be a dict mapping cluster_id -> label or dict")

    # Normalize string and dict overrides into a single map
    override_map: Dict[int, OverrideSpec] = {}
    for key, value in overrides.items():
        cid = int(key)
        if isinstance(value, str):
            override_map[cid] = {"label": value}
            continue
        if isinstance(value, dict):
            # Validate allowed keys and required label
            allowed = {"label", "pval", "hide_stats"}
            unknown = set(value.keys()) - allowed
            if unknown:
                raise ValueError(
                    "overrides may only include keys " f"{sorted(allowed)}; got {sorted(unknown)}"
                )
            label = value.get("label", None)
            if not isinstance(label, str) or not label:
                raise TypeError("override dict must include non-empty 'label' string")
            hide_stats = value.get("hide_stats", False)
            if not isinstance(hide_stats, bool):
                raise TypeError("override 'hide_stats' must be a boolean")
            entry = {"label": label, "hide_stats": hide_stats}
            if "pval" in value and value["pval"] is not None:
                entry["pval"] = value["pval"]
            override_map[cid] = entry
            continue
        raise TypeError("override values must be str or dict")

    return override_map


def _build_label_map(
    df: pd.DataFrame,
    override_map: Dict[int, OverrideSpec],
) -> Dict[int, Tuple[str, Optional[float]]]:
    """
    Resolves final label and p-value per cluster. Combines base labels from the DataFrame
    with any validated overrides.

    Args:
        df (pd.DataFrame): Cluster label table with 'cluster', 'label', and optional 'pval'.
        override_map (Dict[int, OverrideSpec]): Normalized overrides keyed by cluster id.

    Returns:
        Dict[int, Tuple[str, Optional[float]]]: Mapping cluster id to (label, pval).

    Raises:
        ValueError: If overrides reference unknown cluster ids.
    """
    # Build base label map, then apply overrides
    label_map: Dict[int, Tuple[str, Optional[float]]] = {}
    for _, row in df.iterrows():
        cid = int(row["cluster"])
        base_label = str(row["label"])
        if cid in override_map:
            override = override_map[cid]
            label = override["label"]
            pval = override.get("pval", None)
        else:
            label = base_label
            pval = row.get("pval", None)
        label_map[cid] = (label, pval)
    # Reject overrides that do not match any cluster id
    if override_map:
        unknown = set(override_map) - set(label_map)
        if unknown:
            raise ValueError(
                "overrides contain cluster ids not present in cluster_labels: " f"{sorted(unknown)}"
            )

    return label_map


def _setup_label_axis(
    fig: plt.Figure,
    matrix: Matrix,
    style: StyleConfig,
    track_layout: TrackLayoutManager,
    kwargs: Dict[str, Any],
) -> Tuple[plt.Axes, float, List[TrackSpec]]:
    """
    Creates and configures the label axis and computes track layout. Initializes the label panel,
    draws the gutter, and resolves track x-positions.

    Args:
        fig (plt.Figure): Target figure.
        matrix (Matrix): Matrix object providing row count.
        style (StyleConfig): Style configuration.
        track_layout (TrackLayoutManager): Track layout manager.
        kwargs (Dict[str, Any]): Renderer keyword arguments.

    Returns:
        Tuple[plt.Axes, float, List[TrackSpec]]: (label axis, text x-position, resolved tracks).
    """
    n_rows = matrix.df.shape[0]
    # Set up label axis
    label_axes = kwargs.get("axes", style["label_axes"])
    ax_lab = fig.add_axes(label_axes, frameon=False)
    ax_lab.set_xlim(0, 1)
    ax_lab.set_ylim(-0.5, n_rows - 0.5)  # align with matrix row indices
    ax_lab.invert_yaxis()
    ax_lab.set_xticks([])
    ax_lab.set_yticks([])
    # Set up label gutter
    gutter_w = kwargs.get("label_gutter_width", style["label_gutter_width"])
    gutter_color = kwargs.get("label_gutter_color", style["label_gutter_color"])
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
    # Compute track layout
    label_text_pad = kwargs.get("label_text_pad", style.get("label_bar_pad", 0.01))
    base_x = kwargs.get("label_x", style["label_x"])
    track_layout.compute_layout(base_x, gutter_w)
    tracks = track_layout.get_tracks()
    end_x = track_layout.get_end_x()
    if end_x is None:
        end_x = base_x + gutter_w
    label_text_x = end_x + label_text_pad

    return ax_lab, label_text_x, tracks


def _format_cluster_label(
    label: str,
    pval: Optional[float] = None,
    n_members: Optional[int] = None,
    override: Optional[OverrideSpec] = None,
    *,
    label_fields: Tuple[str, ...],
    kwargs: Dict[str, Any],
    style: StyleConfig,
) -> str:
    """
    Formats the cluster label text for display. Applies overrides, selects fields, formats
    stats, and performs truncation/wrapping.

    Args:
        label (str): Base or overridden label.
        pval (float | None): P-value to display, if any. Defaults to None.
        n_members (int | None): Cluster size. Defaults to None.
        override (OverrideSpec | None): Override entry for the cluster. Defaults to None.

    Kwargs:
        label_fields (Tuple[str, ...]): Fields to display.
        kwargs (Dict[str, Any]): Renderer keyword arguments.
        style (StyleConfig): Style configuration.

    Returns:
        str: Final formatted label text.
    """
    # Resolve fields based on overrides
    if override and override.get("hide_stats", False):
        effective_fields = ("label",)
    else:
        effective_fields = label_fields

    # Assemble stats for requested fields.
    pval_value = pval if pval is not None and not pd.isna(pval) else None
    has_label, stats = collect_label_stats(
        effective_fields,
        n_members=n_members,
        pval=pval_value,
    )

    # Apply label-only text policy, then append stats in a stable format.
    max_words = kwargs.get("max_words", None)
    omit_words = kwargs.get("omit_words", style.get("label_omit_words", None))
    wrap_text = kwargs.get("wrap_text", True)
    wrap_width = kwargs.get("wrap_width", style.get("label_wrap_width", None))
    overflow = kwargs.get("overflow", "wrap")
    label_text = apply_label_text_policy(
        label,
        omit_words=omit_words,
        max_words=max_words,
        overflow=overflow,
        wrap_text=wrap_text,
        wrap_width=wrap_width,
    )
    if not has_label and not stats:
        return label_text
    return compose_label_text(
        label_text,
        has_label=has_label,
        stats=stats,
        wrap_text=wrap_text,
        wrap_width=wrap_width,
    )


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
        # Resolve labels, layout, and styling
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
        ) = _resolve_labels_and_layout(
            df,
            kwargs,
            fig,
            matrix,
            layout,
            style,
            track_layout,
        )
        # Render tracks and cluster labels/separators
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
            sep_xmin=sep_xmin,
            sep_xmax=sep_xmax,
            font=font,
            fontsize=fontsize,
            label_fields=label_fields,
            skip_unlabeled=skip_unlabeled,
            kwargs=kwargs,
            style=style,
        )
