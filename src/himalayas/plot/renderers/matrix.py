"""Matrix heatmap renderer."""

from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from .base import BoundaryRegistry


class MatrixRenderer:
    """Render the main heatmap matrix."""

    def __init__(
        self,
        *,
        cmap: str = "viridis",
        center: Optional[float] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show_minor_rows: bool = True,
        minor_row_step: int = 1,
        minor_row_lw: float = 0.15,
        minor_row_alpha: float = 0.15,
        outer_lw: float = 1.2,
        outer_color: str = "black",
        figsize: Optional[tuple[float, float]] = None,
        subplots_adjust: Optional[dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        self.cmap = cmap
        self.center = center
        self.vmin = vmin
        self.vmax = vmax
        self.show_minor_rows = show_minor_rows
        self.minor_row_step = minor_row_step
        self.minor_row_lw = minor_row_lw
        self.minor_row_alpha = minor_row_alpha
        self.outer_lw = outer_lw
        self.outer_color = outer_color
        self.figsize = figsize
        self.subplots_adjust = subplots_adjust
        self._extra = dict(kwargs)

    def render(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        matrix: Any,
        layout: Any,
        style: Any,
        **kwargs: Any,
    ) -> None:
        row_order = layout.leaf_order
        col_order = layout.col_order

        if row_order is None:
            raise ValueError("Row order is required for plotting.")

        if len(row_order) != matrix.df.shape[0]:
            raise ValueError("Dendrogram leaf order does not match matrix dimensions.")

        data = matrix.df.iloc[row_order, :]

        if col_order is not None:
            if len(col_order) != matrix.df.shape[1]:
                raise ValueError("Column order does not match matrix dimensions.")
            data = data.iloc[:, col_order]

        data = data.values
        n_rows, n_cols = data.shape

        if self.center is not None:
            vmin = np.nanmin(data) if self.vmin is None else self.vmin
            vmax = np.nanmax(data) if self.vmax is None else self.vmax
            norm = TwoSlopeNorm(vmin=vmin, vcenter=self.center, vmax=vmax)
            imshow_vmin = None
            imshow_vmax = None
        else:
            norm = None
            imshow_vmin = self.vmin
            imshow_vmax = self.vmax

        extent = (-0.5, n_cols - 0.5, n_rows - 0.5, -0.5)
        ax.imshow(
            data,
            cmap=self.cmap,
            norm=norm,
            vmin=imshow_vmin,
            vmax=imshow_vmax,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            extent=extent,
        )
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(n_rows - 0.5, -0.5)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(self.outer_lw)
            spine.set_color(self.outer_color)

        boundary_registry = kwargs.get("boundary_registry")
        if isinstance(boundary_registry, BoundaryRegistry):
            boundary_registry.render(ax, -0.5, n_cols - 0.5, zorder=2)
