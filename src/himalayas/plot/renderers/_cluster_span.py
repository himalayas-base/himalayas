"""
himalayas/plot/renderers/_cluster_span
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

import matplotlib.pyplot as plt


def draw_cluster_span(
    ax: plt.Axes,
    x: float,
    s: int,
    e: int,
    *,
    gap: float,
    cap_width: float = 0.0,
    color: str,
    lw: float,
    alpha: float,
) -> None:
    """
    Draws a gapped vertical line spanning one cluster's row extent, optionally capped
    into a bracket. Shared by inline and compact label renderers so cluster-abreast
    spans are drawn identically everywhere.

    Args:
        ax (plt.Axes): Target axis, with y in row-index space and x in axes fraction.
        x (float): X-position of the span (axes fraction).
        s (int): Cluster span start (row index), from layout.cluster_spans.
        e (int): Cluster span end (row index), from layout.cluster_spans.

    Kwargs:
        gap (float): Row units trimmed from each end of the span. Clamped to at most
            half the span height, so the span never inverts; a singleton cluster
            (s == e) degenerates to a point at the true center.
        cap_width (float): Width of horizontal end caps (axes fraction). 0 draws a
            bare line with no caps. Defaults to 0.0.
        color (str): Line color.
        lw (float): Line width.
        alpha (float): Line opacity.
    """
    gap = min(max(gap, 0.0), (e - s) / 2.0)
    top, bottom = s + gap, e - gap
    ax.plot([x, x], [top, bottom], color=color, linewidth=lw, alpha=alpha, solid_capstyle="butt")
    if cap_width > 0:
        half = cap_width / 2.0
        ax.plot([x - half, x + half], [top, top], color=color, linewidth=lw, alpha=alpha)
        ax.plot([x - half, x + half], [bottom, bottom], color=color, linewidth=lw, alpha=alpha)
