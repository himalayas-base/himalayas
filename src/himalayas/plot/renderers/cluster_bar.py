"""
himalayas/plot/renderers/cluster_bar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from ._cluster_label_types import ClusterLabelStats

if TYPE_CHECKING:
    from ..style import StyleConfig


def _resolve_cluster_values(
    *,
    cluster_spans: Sequence[Tuple[int, int, int]],
    label_map: Dict[int, ClusterLabelStats],
) -> np.ndarray:
    """
    Resolves raw ranking scores per cluster from the internal cluster label map.

    Kwargs:
        cluster_spans (Sequence[Tuple[int, int, int]]): Iterable of (cluster_id, start, end).
        label_map (Dict[int, ClusterLabelStats]): Mapping cluster_id -> (label, pval, qval, score, fe).

    Returns:
        np.ndarray: Raw values per cluster (float, NaN allowed).
    """
    # Extract ranked score per cluster.
    values = np.full(len(cluster_spans), np.nan, dtype=float)
    for i, (cid, _s, _e) in enumerate(cluster_spans):
        _label, _pval, _qval, score, _fe = label_map.get(
            cid, (None, np.nan, np.nan, np.nan, np.nan)
        )
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            continue
        if np.isfinite(score_value):
            values[i] = score_value

    return values


def _scale_cluster_values(
    values: np.ndarray,
    *,
    norm: Optional[Normalize] = None,
) -> np.ndarray:
    """
    Scales raw values into [0, 1] for colormap lookup.

    Args:
        values (np.ndarray): Raw values.

    Kwargs:
        norm (Optional[Normalize]): Optional matplotlib normalization instance. Defaults to None.

    Returns:
        np.ndarray: Scaled values in [0, 1], NaN preserved.
    """
    # Convert to -log10 scale
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = -np.log10(values)
    # Identify valid entries
    valid = np.isfinite(logp) & (logp >= 0)
    if not np.any(valid):
        return np.full_like(logp, np.nan, dtype=float)
    # Apply normalization
    if norm is None:
        vmax = float(np.nanmax(logp[valid]))
        if not np.isfinite(vmax) or vmax <= 0:
            return np.full_like(logp, np.nan, dtype=float)
        norm = plt.Normalize(vmin=0.0, vmax=vmax)
    # Scale values
    scaled = np.full_like(logp, np.nan, dtype=float)
    try:
        scaled[valid] = np.asarray(norm(logp[valid]), dtype=float)
    except TypeError:
        scaled[valid] = np.array([float(norm(v)) for v in logp[valid]], dtype=float)

    return scaled


def render_cluster_bar_track(
    ax: plt.Axes,
    x0: float,
    width: float,
    payload: Dict[str, Any],
    cluster_spans: Sequence[Tuple[int, int, int]],
    label_map: Dict[int, ClusterLabelStats],
    style: StyleConfig,
) -> None:
    """
    Renders a cluster-level bar track (e.g., -log10 ranked score).

    Args:
        ax (plt.Axes): Matplotlib Axes to draw on.
        x0 (float): Left x position for the bar track.
        width (float): Width of the bar track.
        payload (Dict[str, Any]): Track payload.
        cluster_spans (Sequence[Tuple[int, int, int]]): Iterable of (cluster_id, start, end).
        label_map (Dict[int, ClusterLabelStats]): Mapping cluster_id -> (label, pval, qval, score, fe).
        style (StyleConfig): Style configuration.
    """
    cmap_in = payload.get("cmap", style["sigbar_cmap"])
    cmap = plt.get_cmap(cmap_in) if isinstance(cmap_in, str) else cmap_in
    alpha = payload.get("alpha", style["sigbar_alpha"])
    norm = payload.get("norm", None)
    # Resolve raw values and scale
    raw_values = _resolve_cluster_values(
        cluster_spans=cluster_spans,
        label_map=label_map,
    )
    scaled = _scale_cluster_values(
        raw_values,
        norm=norm,
    )
    # Render bars per cluster
    for (_cid, s, e), sv in zip(cluster_spans, scaled):
        # Keep condensed/main sigbar semantics aligned:
        # unlabeled clusters (NaN scale) render at minimum colormap value.
        bar_value = float(sv) if np.isfinite(sv) else 0.0
        bar_value = float(np.clip(bar_value, 0.0, 1.0))
        bar_color = cmap(bar_value)
        ax.add_patch(
            plt.Rectangle(
                (x0, s - 0.5),
                width,
                e - s + 1,
                facecolor=bar_color,
                edgecolor="none",
                alpha=alpha,
                zorder=1,
            )
        )
