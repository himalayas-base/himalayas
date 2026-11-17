"""
himalayas/plot/renderers/col_label_bar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt

from .label_bar import _resolve_label_bar_colors

if TYPE_CHECKING:
    from ..style import StyleConfig
    from ...core.layout import ClusterLayout
    from ...core.matrix import Matrix


class ColLabelBarRenderer:
    """
    Class for rendering a column-level annotation bar as a strip above the matrix.
    """

    def __init__(
        self,
        *,
        name: str,
        values: Dict[Any, Any],
        mode: str = "categorical",
        colors: Optional[Dict[Any, Any]] = None,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        missing_color: Optional[str] = None,
        height: Optional[float] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        Initializes the ColLabelBarRenderer instance.

        Kwargs:
            name (str): Bar name.
            values (Dict[Any, Any]): Mapping from column label to category or numeric value.
            mode (str): "categorical" or "continuous". Defaults to "categorical".
            colors (Optional[Dict[Any, Any]]): Category-to-color mapping. Defaults to None.
            cmap (str): Colormap name for continuous mode. Defaults to "viridis".
            vmin (Optional[float]): Minimum value for normalization. Defaults to None.
            vmax (Optional[float]): Maximum value for normalization. Defaults to None.
            missing_color (Optional[str]): Color for missing values. Defaults to None.
            height (Optional[float]): Strip height in figure fraction. Defaults to None.
            title (Optional[str]): Title drawn to the left of the strip. Defaults to None.
        """
        self.name = name
        self.values = values
        self.mode = mode
        self.colors = colors
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.missing_color = missing_color
        self.height = height
        self.title = title

    def render(
        self,
        fig: plt.Figure,
        ax_matrix: plt.Axes,
        matrix: Matrix,
        layout: ClusterLayout,
        style: StyleConfig,
        *,
        y_offset: float = 0.0,
    ) -> None:
        """
        Renders the strip directly above the matrix axis, aligned with its columns.

        Args:
            fig (plt.Figure): Target figure.
            ax_matrix (plt.Axes): Matrix axis providing the strip's horizontal extent.
            matrix (Matrix): Matrix object providing column labels.
            layout (ClusterLayout): Cluster layout providing the optional `col_order`.
            style (StyleConfig): Style configuration.

        Kwargs:
            y_offset (float): Figure-fraction gap between the matrix and the strip, used to
                stack several strips. Defaults to 0.0.

        Raises:
            TypeError: If categorical mode has no `colors` dict.
            ValueError: If `mode` is not recognized.
        """
        # Resolve one color per column, in displayed order.
        col_ids = matrix.df.columns.to_numpy()
        if layout.col_order is not None:
            col_ids = col_ids[layout.col_order]
        facecolors = _resolve_label_bar_colors(
            values=self.values,
            row_ids=list(col_ids),
            mode=self.mode,
            colors=self.colors,
            cmap_name=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            missing_color=(
                self.missing_color
                if self.missing_color is not None
                else style["label_bar_missing_color"]
            ),
        )
        # Place the strip over the matrix columns and draw one cell per column.
        box = ax_matrix.get_position()
        height = self.height if self.height is not None else style["col_label_bar_height"]
        ax_bar = fig.add_axes([box.x0, box.y1 + y_offset, box.width, height], frameon=False)
        for i, color in enumerate(facecolors):
            ax_bar.add_patch(
                plt.Rectangle((i - 0.5, 0.0), 1.0, 1.0, facecolor=color, edgecolor="none")
            )
        ax_bar.set_xlim(-0.5, len(facecolors) - 0.5)
        ax_bar.set_ylim(0.0, 1.0)
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        if self.title:
            ax_bar.text(
                -0.005,
                0.5,
                self.title,
                ha="right",
                va="center",
                transform=ax_bar.transAxes,
                fontsize=style["label_fontsize"],
                color=style["text_color"],
            )
