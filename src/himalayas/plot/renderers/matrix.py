"""
himalayas/plot/renderers/matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from .base import BoundaryRegistry


def _resolve_matrix_data(
    matrix: Any,
    layout: Any,
) -> Tuple[np.ndarray, int, int]:
    """
    Resolves and reorder matrix data for rendering.

    Args:
        matrix (Any): Matrix object with DataFrame `df`.
        layout (Any): Layout providing `leaf_order` and optional `col_order`.

    Returns:
        Tuple[np.ndarray, int, int]: (data array, n_rows, n_cols).

    Raises:
        ValueError: If row or column orders do not match matrix dimensions.
    """
    row_order = layout.leaf_order
    col_order = layout.col_order

    # Validation
    if row_order is None:
        raise ValueError("Row order is required for plotting.")
    if len(row_order) != matrix.df.shape[0]:
        raise ValueError("Dendrogram leaf order does not match matrix dimensions.")

    # Reorder data and extract numpy array for plotting
    data = matrix.df.iloc[row_order, :]
    if col_order is not None:
        if len(col_order) != matrix.df.shape[1]:
            raise ValueError("Column order does not match matrix dimensions.")
        data = data.iloc[:, col_order]
    data = data.values
    n_rows, n_cols = data.shape

    return data, n_rows, n_cols


def _resolve_color_normalization(
    data: np.ndarray,
    *,
    center: Optional[float],
    vmin: Optional[float],
    vmax: Optional[float],
) -> Tuple[Optional[Any], Optional[float], Optional[float]]:
    """
    Resolves color normalization for matrix rendering.

    Args:
        data (np.ndarray): Matrix data.
        center (Optional[float]): Center value for diverging normalization.
        vmin (Optional[float]): Minimum value override.
        vmax (Optional[float]): Maximum value override.

    Returns:
        Tuple[Optional[Any], Optional[float], Optional[float]]:
            (norm, imshow_vmin, imshow_vmax)
    """
    if center is not None:
        vmin_ = np.nanmin(data) if vmin is None else vmin
        vmax_ = np.nanmax(data) if vmax is None else vmax
        norm = TwoSlopeNorm(vmin=vmin_, vcenter=center, vmax=vmax_)
        return norm, None, None

    return None, vmin, vmax


def _draw_matrix(
    *,
    ax: plt.Axes,
    data: np.ndarray,
    n_rows: int,
    n_cols: int,
    cmap: str,
    norm: Optional[Any],
    imshow_vmin: Optional[float],
    imshow_vmax: Optional[float],
    gutter_color: Optional[str],
    outer_lw: float,
    outer_color: str,
    boundary_registry: Optional[Any],
) -> None:
    """
    Draws the heatmap matrix and associated decorations.

    Args:
        ax (plt.Axes): Target axis.
        data (np.ndarray): Matrix data.
        n_rows (int): Number of rows.
        n_cols (int): Number of columns.
        cmap (str): Colormap name.
        norm (Optional[Any]): Normalization object.
        imshow_vmin (Optional[float]): vmin for imshow.
        imshow_vmax (Optional[float]): vmax for imshow.
        gutter_color (Optional[str]): Background gutter color.
        outer_lw (float): Outer border linewidth.
        outer_color (str): Outer border color.
        boundary_registry (Optional[Any]): Boundary registry for overlays.
    """
    if gutter_color is not None:
        ax.set_facecolor(gutter_color)
    # Draw matrix heatmap and set axis limits
    extent = (-0.5, n_cols - 0.5, n_rows - 0.5, -0.5)
    ax.imshow(
        data,
        cmap=cmap,
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
    if outer_lw is not None and outer_lw <= 0:
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(outer_lw)
            spine.set_color(outer_color)
    # Draw boundaries if provided
    if isinstance(boundary_registry, BoundaryRegistry):
        boundary_registry.render(ax, -0.5, n_cols - 0.5, zorder=2)


class MatrixRenderer:
    """
    Class for rendering the main heatmap matrix panel.
    """

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
        gutter_color: Optional[str] = None,
        figsize: Optional[tuple[float, float]] = None,
        subplots_adjust: Optional[dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the MatrixRenderer instance.
        """
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
        self.gutter_color = gutter_color
        self.figsize = figsize
        self.subplots_adjust = subplots_adjust
        self._extra = dict(kwargs)

    def render(
        self,
        ax: plt.Axes,
        matrix: Any,
        layout: Any,
        style: Any,
        **kwargs: Any,
    ) -> None:
        """
        Renders the heatmap matrix panel.

        Args:
            ax (plt.Axes): Target axis.
            matrix (Any): Matrix object.
            layout (Any): Layout object.
            style (Any): Style configuration.
            **kwargs (Any): Renderer keyword arguments.
        """
        data, n_rows, n_cols = _resolve_matrix_data(matrix, layout)
        # Resolve color normalization and draw matrix
        gutter_color = (
            self.gutter_color
            if self.gutter_color is not None
            else style.get("matrix_gutter_color", None)
        )
        norm, imshow_vmin, imshow_vmax = _resolve_color_normalization(
            data,
            center=self.center,
            vmin=self.vmin,
            vmax=self.vmax,
        )
        _draw_matrix(
            ax=ax,
            data=data,
            n_rows=n_rows,
            n_cols=n_cols,
            cmap=self.cmap,
            norm=norm,
            imshow_vmin=imshow_vmin,
            imshow_vmax=imshow_vmax,
            gutter_color=gutter_color,
            outer_lw=self.outer_lw,
            outer_color=self.outer_color,
            boundary_registry=kwargs.get("boundary_registry"),
        )
