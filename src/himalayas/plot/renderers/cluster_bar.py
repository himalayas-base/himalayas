"""Cluster-level bar track renderer."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def render_cluster_bar_track(
    ax: plt.Axes,
    x0: float,
    width: float,
    payload: dict[str, Any],
    cluster_spans,
    label_map,
    style: Any,
    row_order,
) -> None:
    """Render a cluster-level bar track (e.g., -log10 p-values)."""
    value_map = payload.get("value_map", {})

    cmap_in = payload.get("cmap", style["sigbar_cmap"])
    cmap = plt.get_cmap(cmap_in) if isinstance(cmap_in, str) else cmap_in

    alpha = payload.get("alpha", style["sigbar_alpha"])
    norm = payload.get("norm", None)

    source = payload.get("source", None)

    pvals = np.full(len(cluster_spans), np.nan, dtype=float)
    if source == "cluster_labels_pval":
        for i, (cid, _s, _e) in enumerate(cluster_spans):
            _label, pval = label_map.get(cid, (None, np.nan))
            if pval is None or pd.isna(pval):
                pvals[i] = np.nan
            else:
                pvals[i] = float(pval)
    else:
        for i, (cid, _s, _e) in enumerate(cluster_spans):
            v = value_map.get(cid, np.nan)
            if v is None or pd.isna(v):
                pvals[i] = np.nan
            else:
                pvals[i] = float(v)

    with np.errstate(divide="ignore", invalid="ignore"):
        logp = -np.log10(pvals)

    valid = np.isfinite(logp) & (logp >= 0)
    if not np.any(valid):
        return

    if norm is None:
        vmax = float(np.nanmax(logp[valid]))
        if not np.isfinite(vmax) or vmax <= 0:
            return
        norm = plt.Normalize(vmin=0.0, vmax=vmax)

    scaled = np.full_like(logp, np.nan, dtype=float)
    try:
        scaled[valid] = np.asarray(norm(logp[valid]), dtype=float)
    except TypeError:
        scaled[valid] = np.array([float(norm(v)) for v in logp[valid]], dtype=float)

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
