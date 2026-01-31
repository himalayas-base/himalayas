"""
himalayas/plot/renderers/cluster_bar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _resolve_cluster_values(
    *,
    cluster_spans: Sequence[Tuple[int, int, int]],
    payload: Dict[str, Any],
    label_map: Dict[int, Tuple[Any, Optional[float]]],
) -> np.ndarray:
    """
    Resolves raw values per cluster.

    Args:
        cluster_spans (Sequence[Tuple[int, int, int]]): Iterable of (cluster_id, start, end).
        payload (Dict[str, Any]): Track payload.
        label_map (Dict[int, Tuple[Any, Optional[float]]]): Mapping cluster_id -> (label, pval).

    Returns:
        np.ndarray: Raw values per cluster (float, NaN allowed).
    """
    source = payload.get("source", None)
    value_map = payload.get("value_map", {})

    values = np.full(len(cluster_spans), np.nan, dtype=float)

    if source == "cluster_labels_pval":
        for i, (cid, _s, _e) in enumerate(cluster_spans):
            _label, pval = label_map.get(cid, (None, np.nan))
            if pval is not None and not pd.isna(pval):
                values[i] = float(pval)
    else:
        for i, (cid, _s, _e) in enumerate(cluster_spans):
            v = value_map.get(cid, np.nan)
            if v is not None and not pd.isna(v):
                values[i] = float(v)

    return values


def _scale_cluster_values(
    values: np.ndarray,
    *,
    norm: Optional[Any] = None,
) -> np.ndarray:
    """
    Scales raw values into [0, 1] for colormap lookup.

    Args:
        values (np.ndarray): Raw values.
        norm (Any): Optional matplotlib normalization instance.

    Returns:
        np.ndarray: Scaled values in [0, 1], NaN preserved.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = -np.log10(values)

    valid = np.isfinite(logp) & (logp >= 0)
    if not np.any(valid):
        return np.full_like(logp, np.nan, dtype=float)

    if norm is None:
        vmax = float(np.nanmax(logp[valid]))
        if not np.isfinite(vmax) or vmax <= 0:
            return np.full_like(logp, np.nan, dtype=float)
        norm = plt.Normalize(vmin=0.0, vmax=vmax)

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
    label_map: Dict[int, Tuple[Any, Optional[float]]],
    style: Any,
) -> None:
    """
    Renders a cluster-level bar track (e.g., -log10 p-values).

    Args:
        ax (plt.Axes): Matplotlib Axes to draw on.
        x0 (float): Left x position for the bar track.
        width (float): Width of the bar track.
        payload (Dict[str, Any]): Track payload.
        cluster_spans (Sequence[Tuple[int, int, int]]): Iterable of (cluster_id, start, end).
        label_map (Dict[int, Tuple[Any, Optional[float]]]): Mapping cluster_id -> (label, pval).
        style (Any): Style configuration.
    """
    cmap_in = payload.get("cmap", style["sigbar_cmap"])
    cmap = plt.get_cmap(cmap_in) if isinstance(cmap_in, str) else cmap_in

    alpha = payload.get("alpha", style["sigbar_alpha"])
    norm = payload.get("norm", None)

    raw_values = _resolve_cluster_values(
        cluster_spans=cluster_spans,
        payload=payload,
        label_map=label_map,
    )

    scaled = _scale_cluster_values(
        raw_values,
        norm=norm,
    )

    for (_cid, s, e), sv in zip(cluster_spans, scaled):
        if not np.isfinite(sv):
            continue
        bar_color = cmap(sv)
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
