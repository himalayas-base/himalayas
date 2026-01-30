"""Cluster label panel renderer."""

from __future__ import annotations

from typing import Any, Optional, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..track_layout import TrackLayoutManager


def _parse_label_overrides(overrides):
    """
    Args:
        overrides (Dict[int, Any] | None): Mapping cluster_id -> label string or override dict.
    Returns:
        Dict[int, Dict[str, Any]]: Normalized override map keyed by cluster id.
    """
    if overrides is None:
        return {}

    if not isinstance(overrides, dict):
        raise TypeError("overrides must be a dict mapping cluster_id -> label or dict")

    override_map = {}
    for key, value in overrides.items():
        cid = int(key)
        if isinstance(value, str):
            override_map[cid] = {"label": value}
            continue
        if isinstance(value, dict):
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


def _build_label_map(df, override_map):
    """
    Args:
        df (pd.DataFrame): ...
        override_map (Dict[int, Dict[str, Any]]): ...
    Returns:
        Dict[int, Tuple[str, Optional[float]]]: ...
    """
    label_map = {}
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

    if override_map:
        unknown = set(override_map) - set(label_map)
        if unknown:
            raise ValueError(
                "overrides contain cluster ids not present in cluster_labels: " f"{sorted(unknown)}"
            )

    return label_map


def _setup_label_axis(fig, matrix, style, track_layout, kwargs):
    """
    Set up the label axis for cluster labels.

    Args:
        ...
        kwargs (Dict[str, Any]): Renderer keyword arguments.
    Returns:
        Tuple[plt.Axes, float, List[Dict[str, Any]]]: ...
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
    cid,
    label,
    pval,
    n_members,
    override,
    label_fields,
    kwargs,
    style,
):
    """
    Args:
        ...
        override (Dict[str, Any] | None): ...
        label_fields (Tuple[str, ...]): ...
        kwargs (Dict[str, Any]): ...
    Returns:
        str: ...
    """
    if override:
        if override.get("hide_stats", False):
            effective_fields = ("label",)
        elif "pval" in override and override.get("pval") is not None:
            effective_fields = ("label", "p")
        else:
            effective_fields = ("label",)
    else:
        effective_fields = label_fields

    parts = []
    for field in effective_fields:
        if field == "label":
            parts.append(label)
        elif field == "n" and n_members is not None:
            parts.append(f"n={n_members}")
        elif field == "p" and pval is not None and not pd.isna(pval):
            parts.append(rf"$p$={pval:.2e}")

    if not parts:
        text = label
    else:
        if parts[0] == label:
            head = label
            tail = parts[1:]
            text = f"{head} ({', '.join(tail)})" if tail else head
        else:
            text = ", ".join(parts)

    max_words = kwargs.get("max_words", None)
    wrap_text = kwargs.get("wrap_text", True)
    wrap_width = kwargs.get("wrap_width", style.get("label_wrap_width", None))
    overflow = kwargs.get("overflow", "wrap")

    words = text.split()
    if max_words is not None and len(words) > max_words:
        if overflow == "ellipsis":
            text = " ".join(words[:max_words]) + "\u2026"
        else:
            text = " ".join(words[:max_words])

    if wrap_text and wrap_width is not None:
        import textwrap

        text = "\n".join(textwrap.wrap(text, width=wrap_width))

    return text


class ClusterLabelsRenderer:
    """Render the cluster label panel with tracks and annotations."""

    def __init__(self, df: pd.DataFrame, **kwargs: Any) -> None:
        self.df = df
        self.kwargs = dict(kwargs)

    def render(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        matrix: Any,
        layout: Any,
        style: Any,
        track_layout: TrackLayoutManager,
        bar_labels_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            ...
            bar_labels_kwargs (Dict[str, Any] | None): ...
        """
        df = self.df
        kwargs = self.kwargs

        if not isinstance(df, pd.DataFrame):
            raise TypeError("cluster_labels must be a pandas DataFrame.")
        if "cluster" not in df.columns or "label" not in df.columns:
            raise ValueError("cluster_labels DataFrame must contain columns: 'cluster', 'label'.")

        overrides = kwargs.get("overrides", None)
        override_map = _parse_label_overrides(overrides)

        label_map = _build_label_map(df, override_map)

        spans = layout.cluster_spans
        cluster_sizes = layout.cluster_sizes

        ax_lab, label_text_x, tracks = _setup_label_axis(fig, matrix, style, track_layout, kwargs)

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

        font = kwargs.get("font", "Helvetica")
        fontsize = kwargs.get("fontsize", style.get("label_fontsize", 9))
        max_words = kwargs.get("max_words", None)
        skip_unlabeled = kwargs.get("skip_unlabeled", False)
        label_fields = kwargs.get("label_fields", style["label_fields"])
        if not isinstance(label_fields, (list, tuple)):
            raise TypeError("label_fields must be a list or tuple of strings")
        allowed_fields = {"label", "n", "p"}
        if any(f not in allowed_fields for f in label_fields):
            raise ValueError(f"label_fields may only contain {allowed_fields}")

        row_order = layout.leaf_order
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
                    row_order,
                )

        if bar_labels_kwargs is not None:
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

        for cid, s, e in spans:
            y_center = (s + e) / 2.0
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
                    cid,
                    label,
                    pval,
                    n_members,
                    override_map.get(cid),
                    label_fields,
                    kwargs,
                    style,
                )
                text_kwargs = {
                    "font": font,
                    "fontsize": fontsize,
                    "color": kwargs.get("color", style.get("text_color", "black")),
                    "alpha": kwargs.get("alpha", 0.9),
                }

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
