"""
himalayas/plot/renderers/_track_rendering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from ._cluster_label_types import ClusterLabelStats
from ._text_style import apply_text_style

if TYPE_CHECKING:
    from ..style import StyleConfig
    from ...core.matrix import Matrix


class TrackSpec(TypedDict, total=False):
    """
    Typed dictionary of resolved label track specifications.
    """

    name: str
    kind: str
    renderer: Callable[..., None]
    left_pad: float
    width: float
    right_pad: float
    enabled: bool
    payload: Dict[str, Any]
    x0: float
    x1: float


def _render_tracks(
    ax_lab: plt.Axes,
    tracks: List[TrackSpec],
    *,
    matrix: Matrix,
    row_order: np.ndarray,
    spans: Sequence[Tuple[int, int, int]],
    label_map: Dict[int, ClusterLabelStats],
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
        label_map (Dict[int, ClusterLabelStats]): Mapping cluster_id -> (label, pval, qval, score, fe).
        style (StyleConfig): Style configuration.
        bar_labels_kwargs (Optional[Dict[str, Any]]): Bar title rendering options. Defaults to None.
    """
    # Render track content: data tracks and cluster-level tracks.
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

    # Render optional bar titles.
    if bar_labels_kwargs is None:
        return

    # Render bar titles beneath tracks.
    bar_pad_pts = bar_labels_kwargs.get("pad", 2)
    bar_rotation = bar_labels_kwargs.get("rotation", 0)
    for track in tracks:
        title = track.get("payload", {}).get("title", None)
        if not title:
            continue
        x_center = (track.get("x0", 0.0) + track.get("x1", 0.0)) / 2.0
        txt = ax_lab.annotate(
            title,
            xy=(x_center, 0.0),
            xycoords=ax_lab.transAxes,
            xytext=(0, -bar_pad_pts),
            textcoords="offset points",
            ha="center",
            va="top",
            rotation=bar_rotation,
            clip_on=False,
        )
        apply_text_style(
            txt,
            font=bar_labels_kwargs.get("font", "Helvetica"),
            fontsize=bar_labels_kwargs.get("fontsize", 10),
            color=bar_labels_kwargs.get("color", style.get("text_color", "black")),
            alpha=bar_labels_kwargs.get("alpha", 1.0),
        )
