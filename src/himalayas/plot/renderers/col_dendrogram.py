"""
himalayas/plot/renderers/col_dendrogram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

if TYPE_CHECKING:
    from ..style import StyleConfig
    from ...core.layout import ClusterLayout
    from ...core.matrix import Matrix


# SciPy places leaf i of a dendrogram at 5 + 10 * i along the leaf axis.
_LEAF_START = 5.0
_LEAF_SPACING = 10.0


def _resolve_col_dendrogram_config(
    renderer: "ColDendrogramRenderer",
    ax_matrix: plt.Axes,
    style: StyleConfig,
    y_offset: float,
) -> Dict[str, Any]:
    """
    Resolves column dendrogram rendering configuration. The default axes box spans the matrix
    columns and sits directly above the matrix, shifted up by `y_offset`.

    Args:
        renderer (ColDendrogramRenderer): Renderer instance holding optional overrides.
        ax_matrix (plt.Axes): Matrix axis providing the default horizontal extent and base.
        style (StyleConfig): Style configuration.
        y_offset (float): Figure-fraction shift applied to the default axes box only.

    Returns:
        Dict[str, Any]: Normalized configuration values.
    """
    box = ax_matrix.get_position()
    default_axes = [box.x0, box.y1 + y_offset, box.width, style["col_dendro_height"]]
    return {
        "axes": renderer.axes if renderer.axes is not None else default_axes,
        "color": renderer.color if renderer.color is not None else style["dendro_color"],
        "linewidth": renderer.linewidth if renderer.linewidth is not None else style["dendro_lw"],
        "data_pad": renderer.data_pad,
    }


class ColDendrogramRenderer:
    """
    Class for rendering a dendrogram aligned to matrix columns.
    """

    def __init__(
        self,
        *,
        axes: Optional[Sequence[float]] = None,
        color: Optional[str] = None,
        linewidth: Optional[float] = None,
        data_pad: float = 0.25,
    ) -> None:
        """
        Initializes the ColDendrogramRenderer instance.

        Kwargs:
            axes (Optional[Sequence[float]]): Axes position [x0, y0, width, height]. Defaults to None.
            color (Optional[str]): Dendrogram line color. Defaults to None.
            linewidth (Optional[float]): Dendrogram line width. Defaults to None.
            data_pad (float): Padding around data in the x-direction. Defaults to 0.25.
        """
        self.axes = axes
        self.color = color
        self.linewidth = linewidth
        self.data_pad = data_pad

    def render(
        self,
        fig: plt.Figure,
        ax_matrix: plt.Axes,
        matrix: Matrix,
        layout: ClusterLayout,
        style: StyleConfig,
        *,
        y_offset: float = 0.0,
    ) -> plt.Axes:
        """
        Renders a dendrogram aligned to matrix columns, with the root at the top.

        Args:
            fig (plt.Figure): Target figure.
            ax_matrix (plt.Axes): Matrix axis the default axes box is aligned to.
            matrix (Matrix): Matrix object providing the column count.
            layout (ClusterLayout): Cluster layout providing `col_linkage`.
            style (StyleConfig): Style configuration.

        Kwargs:
            y_offset (float): Figure-fraction shift of the default axes box, used to sit above
                column label bars. Ignored when `axes` is set. Defaults to 0.0.

        Returns:
            plt.Axes: Column dendrogram axis.

        Raises:
            ValueError: If the layout has no column linkage.
        """
        # Validation
        if layout.col_linkage is None:
            raise ValueError("Column dendrogram rendering requires a column linkage.")

        # Resolve configuration and create dendrogram axis.
        cfg = _resolve_col_dendrogram_config(self, ax_matrix, style, y_offset)
        ax_dend = fig.add_axes(cfg["axes"], frameon=False)
        dendro = dendrogram(
            layout.col_linkage,
            orientation="top",
            no_labels=True,
            distance_sort=False,
            no_plot=True,
        )
        # Map leaf positions onto matrix column centers and draw segments.
        for icoord, dcoord in zip(dendro["icoord"], dendro["dcoord"]):
            ax_dend.plot(
                [(x - _LEAF_START) / _LEAF_SPACING for x in icoord],
                dcoord,
                color=cfg["color"],
                linewidth=cfg["linewidth"],
            )
        # Align limits with the matrix columns and hide the frame.
        n_cols = matrix.df.shape[1]
        ax_dend.set_xlim(-0.5 - cfg["data_pad"], n_cols - 0.5 + cfg["data_pad"])
        ax_dend.set_xticks([])
        ax_dend.set_yticks([])
        for spine in ax_dend.spines.values():
            spine.set_visible(False)
        return ax_dend
