"""
himalayas/plot/renderers/gene_bar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _resolve_gene_bar_colors(
    *,
    values: Mapping[Any, Any],
    row_ids: List[Any],
    mode: str,
    colors: Optional[Dict[Any, Any]],
    cmap_name: str,
    vmin: Optional[float],
    vmax: Optional[float],
    missing_color: Any,
) -> List[Any]:
    """
    Resolves per-row facecolors for a gene bar.

    Args:
        values (Mapping[Any, Any]): Mapping from row ID to value.
        row_ids (List[Any]): Ordered row identifiers.
        mode (str): "categorical" or "continuous".
        colors (Optional[Dict[Any, Any]]): Category-to-color mapping.
        cmap_name (str): Colormap name for continuous mode.
        vmin (Optional[float]): Minimum value for normalization.
        vmax (Optional[float]): Maximum value for normalization.
        missing_color (Any): Color for missing values.

    Returns:
        List[Any]: Facecolors per row.
    """
    # Categorical mode: map categories to colors
    if mode == "categorical":
        if colors is None or not isinstance(colors, dict):
            raise TypeError(
                "categorical gene_bar requires `colors` as a dict mapping category -> color"
            )
        return [colors.get(values.get(gid, None), missing_color) for gid in row_ids]

    # Continuous mode: map numeric values to colormap
    if mode == "continuous":
        # Get colormap
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
    raise ValueError("gene_bar mode must be 'categorical' or 'continuous'")


def _draw_gene_bar_cells(
    *,
    ax: plt.Axes,
    x0: float,
    width: float,
    colors: List[Any],
    zorder: int = 2,
) -> None:
    """
    Draws rectangular gene bar cells.

    Args:
        ax (plt.Axes): Target axis.
        x0 (float): Left x-position.
        width (float): Bar width.
        colors (List[Any]): Facecolors per row.
        zorder (int): Patch z-order.
    """
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


class GeneBarRenderer:
    """
    Class for rendering a row-level gene annotation bar.
    """

    def __init__(
        self,
        *,
        values: Mapping[Any, Any],
        mode: str = "categorical",
        colors: Optional[dict[Any, Any]] = None,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        missing_color: Optional[str] = None,
        axes: Optional[list[float]] = None,
        gene_bar_gap: Optional[float] = None,
        gene_bar_width: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the GeneBarRenderer instance.

        Args:
            values (Mapping[Any, Any]): Mapping from row ID to value.
            mode (str): "categorical" or "continuous".
            colors (Optional[dict[Any, Any]]): Category-to-color mapping.
            cmap (str): Colormap name for continuous mode.
            vmin (Optional[float]): Minimum value for normalization.
            vmax (Optional[float]): Maximum value for normalization.
            missing_color (Optional[str]): Color for missing values.
            axes (Optional[list[float]]): Axes position [x0, y0, width, height].
            gene_bar_gap (Optional[float]): Gap between dendrogram and gene bar.
            gene_bar_width (Optional[float]): Width of the gene bar.
            **kwargs: Additional keyword arguments.
        """
        self.values = values
        self.mode = mode
        self.colors = colors
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.missing_color = missing_color
        self.axes = axes
        self.gene_bar_gap = gene_bar_gap
        self.gene_bar_width = gene_bar_width
        self._extra = dict(kwargs)

    def render(
        self,
        fig: plt.Figure,
        matrix: Any,
        layout: Any,
        style: Any,
    ) -> None:
        """
        Renders a gene annotation bar aligned to matrix rows.

        Args:
            fig (plt.Figure): Target figure.
            matrix (Any): Data matrix.
            layout (Any): Track layout information.
            style (Any): Plot style configuration.

        Raises:
            TypeError: If `values` is not a dict mapping row IDs to values.
        """
        # Validation
        if not isinstance(self.values, dict):
            raise TypeError("plot_gene_bar expects `values` as a dict mapping row IDs to values")

        # Determine parameters; use style defaults if not set
        missing_color = (
            self.missing_color
            if self.missing_color is not None
            else style["gene_bar_missing_color"]
        )
        dendro_axes = style["dendro_axes"]
        gap = self.gene_bar_gap if self.gene_bar_gap is not None else style["gene_bar_gap"]
        bar_w = self.gene_bar_width if self.gene_bar_width is not None else style["gene_bar_width"]

        # Determine axes and position gene bar to the right of the dendrogram
        x0 = dendro_axes[0] + dendro_axes[2] + gap
        bar_axes = (
            self.axes if self.axes is not None else [x0, dendro_axes[1], bar_w, dendro_axes[3]]
        )
        ax_bar = fig.add_axes(bar_axes, frameon=False)
        ax_bar.set_xlim(0, 1)
        n_rows = matrix.df.shape[0]
        ax_bar.set_ylim(-0.5, n_rows - 0.5)
        ax_bar.invert_yaxis()
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])

        # Resolve facecolors and draw gene bar
        row_ids = matrix.df.index.to_numpy()[layout.leaf_order]
        facecolors = _resolve_gene_bar_colors(
            values=self.values,
            row_ids=list(row_ids),
            mode=self.mode,
            colors=self.colors,
            cmap_name=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            missing_color=missing_color,
        )
        _draw_gene_bar_cells(
            ax=ax_bar,
            x0=0.0,
            width=1.0,
            colors=facecolors,
            zorder=2,
        )


def render_gene_bar_track(
    ax: plt.Axes,
    x0: float,
    width: float,
    payload: dict[str, Any],
    matrix: Any,
    row_order,
    style: Any,
) -> None:
    """
    Renders a gene bar track inside the label panel.

    Args:
        ax (plt.Axes): Target axis.
        x0 (float): Left x-position.
        width (float): Bar width.
        payload (dict[str, Any]): Track payload data.
        matrix (Any): Data matrix.
        row_order (List[int]): Row ordering indices.
        style (Any): Plot style configuration.
    """
    # Resolve facecolors and draw gene bar
    values = payload["values"]
    mode = payload.get("mode", "categorical")
    missing_color = payload.get("missing_color", style["gene_bar_missing_color"])
    row_ids = matrix.df.index.to_numpy()[row_order]
    facecolors = _resolve_gene_bar_colors(
        values=values,
        row_ids=list(row_ids),
        mode=mode,
        colors=payload.get("colors"),
        cmap_name=payload.get("cmap", "viridis"),
        vmin=payload.get("vmin"),
        vmax=payload.get("vmax"),
        missing_color=missing_color,
    )
    _draw_gene_bar_cells(
        ax=ax,
        x0=x0,
        width=width,
        colors=facecolors,
        zorder=2,
    )
