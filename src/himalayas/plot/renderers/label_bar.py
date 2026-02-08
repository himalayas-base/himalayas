"""
himalayas/plot/renderers/label_bar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Dict, List, Sequence, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ..style import StyleConfig
    from ...core.matrix import Matrix


def _resolve_label_bar_colors(
    *,
    values: Mapping[Any, Any],
    row_ids: Sequence[Any],
    mode: str,
    colors: Optional[Dict[Any, Any]] = None,
    cmap_name: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    missing_color: Any,
) -> List[Any]:
    """
    Resolves per-row facecolors for a label bar.

    Kwargs:
        values (Mapping[Any, Any]): Mapping from row ID to value.
        row_ids (Sequence[Any]): Ordered row identifiers.
        mode (str): "categorical" or "continuous".
        colors (Optional[Dict[Any, Any]]): Category-to-color mapping. Defaults to None.
        cmap_name (str): Colormap name for continuous mode.
        vmin (Optional[float]): Minimum value for normalization. Defaults to None.
        vmax (Optional[float]): Maximum value for normalization. Defaults to None.
        missing_color (Any): Color for missing values.

    Returns:
        List[Any]: Facecolors per row.

    Raises:
        TypeError: If `colors` is not provided in categorical mode.
        ValueError: If `mode` is not recognized.
    """
    # Categorical mode: map categories to colors
    if mode == "categorical":
        if colors is None or not isinstance(colors, dict):
            raise TypeError(
                "categorical label_bar requires `colors` as a dict mapping category -> color"
            )
        return [colors.get(values.get(gid, None), missing_color) for gid in row_ids]

    # Continuous mode: map numeric values to colormap
    if mode == "continuous":
        cmap = plt.get_cmap(cmap_name)
        vals = np.array(
            [values.get(gid, np.nan) for gid in row_ids],
            dtype=float,
        )
        finite = np.isfinite(vals)
        if not np.any(finite):
            return [missing_color] * len(row_ids)
        # Determine normalization
        vmin_ = vmin if vmin is not None else float(np.nanmin(vals[finite]))
        vmax_ = vmax if vmax is not None else float(np.nanmax(vals[finite]))
        if vmin_ == vmax_:
            vmax_ = vmin_ + 1e-12
        norm = plt.Normalize(vmin=vmin_, vmax=vmax_)
        return [missing_color if not np.isfinite(v) else cmap(norm(v)) for v in vals]

    # Unknown mode
    raise ValueError("label_bar mode must be 'categorical' or 'continuous'")


def _draw_label_bar_cells(
    *,
    ax: plt.Axes,
    x0: float,
    width: float,
    colors: List[Any],
    zorder: int = 2,
) -> None:
    """
    Draws rectangular label bar cells.

    Kwargs:
        ax (plt.Axes): Target axis.
        x0 (float): Left x-position.
        width (float): Bar width.
        colors (List[Any]): Facecolors per row.
        zorder (int): Patch z-order. Defaults to 2.
    """
    # Draw one rectangle per row
    for i, c in enumerate(colors):
        ax.add_patch(
            plt.Rectangle(
                (x0, i - 0.5),
                width,
                1.0,
                facecolor=c,
                edgecolor="none",
                zorder=zorder,
            )
        )


def render_label_bar_track(
    ax: plt.Axes,
    x0: float,
    width: float,
    payload: Dict[str, Any],
    matrix: Matrix,
    row_order: Sequence[int],
    style: StyleConfig,
) -> None:
    """
    Renders a label bar track inside the label panel.

    Args:
        ax (plt.Axes): Target axis.
        x0 (float): Left x-position.
        width (float): Bar width.
        payload (Dict[str, Any]): Track payload data.
        matrix (Matrix): Data matrix.
        row_order (Sequence[int]): Row ordering indices.
        style (StyleConfig): Plot style configuration.
    """
    # Resolve facecolors and draw label bar
    values = payload["values"]
    mode = payload.get("mode", "categorical")
    missing_color = payload.get("missing_color", style["label_bar_missing_color"])
    row_ids = matrix.df.index.to_numpy()[row_order]
    facecolors = _resolve_label_bar_colors(
        values=values,
        row_ids=list(row_ids),
        mode=mode,
        colors=payload.get("colors"),
        cmap_name=payload.get("cmap", "viridis"),
        vmin=payload.get("vmin"),
        vmax=payload.get("vmax"),
        missing_color=missing_color,
    )
    _draw_label_bar_cells(
        ax=ax,
        x0=x0,
        width=width,
        colors=facecolors,
        zorder=2,
    )
