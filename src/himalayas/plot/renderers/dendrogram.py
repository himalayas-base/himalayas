"""
himalayas/plot/renderers/dendrogram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, Optional, Dict, Tuple, Sequence, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from scipy.cluster.hierarchy import dendrogram

if TYPE_CHECKING:
    from ..style import StyleConfig
    from ...core.layout import ClusterLayout
    from ...core.matrix import Matrix


def _resolve_dendrogram_config(
    renderer: "DendrogramRenderer",
    style: StyleConfig,
) -> Dict[str, Any]:
    """
    Resolves dendrogram rendering configuration.

    Args:
        renderer (DendrogramRenderer): Renderer instance holding optional overrides.
        style (StyleConfig): Style configuration.

    Returns:
        Dict[str, Any]: Normalized configuration values.
    """
    return {
        "axes": renderer.axes if renderer.axes is not None else style["dendro_axes"],
        "color": renderer.color if renderer.color is not None else style["dendro_color"],
        "linewidth": renderer.linewidth if renderer.linewidth is not None else style["dendro_lw"],
        "data_pad": renderer.data_pad,
    }


def _resolve_linkage_matrix(
    kwargs: Dict[str, Any],
) -> np.ndarray:
    """
    Resolves the linkage matrix used to compute the dendrogram.

    Args:
        kwargs (Dict[str, Any]): Renderer keyword arguments.

    Returns:
        np.ndarray: SciPy linkage matrix.

    Raises:
        ValueError: If neither `linkage_matrix` nor a `results.clusters` linkage is provided.
    """
    # Try to get linkage matrix from kwargs
    linkage_matrix = kwargs.get("linkage_matrix")
    if linkage_matrix is not None:
        return linkage_matrix
    # Try to get linkage matrix from results.clusters; raise error if not available
    results = kwargs.get("results")
    if results is None or not hasattr(results, "clusters"):
        raise ValueError("Dendrogram rendering requires `results` or `linkage_matrix`.")
    return results.clusters.linkage_matrix


def _compute_y_affine_map(
    icoord: Sequence[Sequence[float]],
    n_rows: int,
) -> Tuple[float, float, float, float]:
    """
    Computes an affine mapping from dendrogram y-coordinates to matrix row coordinates.

    Args:
        icoord (Sequence[Sequence[float]]): Dendrogram y-coordinates from SciPy (dendro["icoord"]).
        n_rows (int): Number of matrix rows.

    Returns:
        Tuple[float, float, float, float]: (scale, offset, target_y_min, target_y_max).
    """
    # Compute dendrogram y-bounds and target y-bounds, then derive scale and offset
    dendro_y_min = min(min(y) for y in icoord)
    dendro_y_max = max(max(y) for y in icoord)
    target_y_min = -0.5
    target_y_max = n_rows - 0.5
    scale = (target_y_max - target_y_min) / (dendro_y_max - dendro_y_min)
    offset = target_y_min - scale * dendro_y_min
    return scale, offset, target_y_min, target_y_max


def _render_dendrogram_segments(
    ax_dend: plt.Axes,
    dendro: Dict[str, Any],
    scale: float,
    offset: float,
    *,
    color: str,
    linewidth: float,
) -> None:
    """
    Renders dendrogram line segments.

    Args:
        ax_dend (plt.Axes): Dendrogram axis.
        dendro (Dict[str, Any]): SciPy dendrogram output with "icoord" and "dcoord".
        scale (float): Y scale factor.
        offset (float): Y offset.

    Kwargs:
        color (str): Line color.
        linewidth (float): Line width.
    """
    # Map dendrogram y-coordinates into matrix row space and draw segments
    for icoord, dcoord in zip(dendro["icoord"], dendro["dcoord"]):
        icoord_mapped = [scale * y + offset for y in icoord]
        ax_dend.plot(
            dcoord,
            icoord_mapped,
            color=color,
            linewidth=linewidth,
        )


def _render_cluster_boundaries(
    ax_dend: plt.Axes,
    layout: ClusterLayout,
    target_y_max: float,
    boundary_style: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Renders horizontal boundary lines aligned to cluster starts.

    Args:
        ax_dend (plt.Axes): Dendrogram axis.
        layout (ClusterLayout): Layout providing `cluster_spans`.
        target_y_max (float): Maximum y-value in matrix coordinates.
        boundary_style (Optional[Dict[str, Any]]): Boundary styling dict. Defaults to None.
    """
    # Skip if no boundary style provided
    if boundary_style is None:
        return
    # Compute y-positions for boundaries
    ys = [
        s - 0.5
        for _cid, s, _e in layout.cluster_spans
        if (s - 0.5) > -0.5 and (s - 0.5) < target_y_max
    ]
    if not ys:
        return
    # Render horizontal lines
    x0, x1 = ax_dend.get_xlim()
    segs = [((x0, y), (x1, y)) for y in ys]
    boundary_color = to_rgba(
        boundary_style["color"],
        boundary_style["alpha"],
    )
    ax_dend.add_collection(
        LineCollection(
            segs,
            linewidths=[boundary_style["lw"]] * len(segs),
            colors=[boundary_color] * len(segs),
            zorder=3,
        )
    )


def _finalize_dendrogram_axis(
    ax_dend: plt.Axes,
    *,
    target_y_min: float,
    target_y_max: float,
    data_pad: float,
) -> None:
    """
    Finalizes dendrogram axis limits and visibility.

    Args:
        ax_dend (plt.Axes): Dendrogram axis.

    Kwargs:
        target_y_min (float): Minimum y-value in matrix coordinates.
        target_y_max (float): Maximum y-value in matrix coordinates.
        data_pad (float): Padding added to y-limits.
    """
    # Invert axes and set limits
    ax_dend.set_ylim(target_y_min, target_y_max + data_pad)
    ax_dend.invert_yaxis()
    ax_dend.invert_xaxis()
    ax_dend.set_xticks([])
    ax_dend.set_yticks([])
    ax_dend.spines["top"].set_visible(False)
    ax_dend.spines["right"].set_visible(False)
    ax_dend.spines["bottom"].set_visible(False)
    ax_dend.spines["left"].set_visible(False)


class DendrogramRenderer:
    """
    Class for rendering a dendrogram aligned to matrix rows.
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
        Initializes the DendrogramRenderer instance.

        Kwargs:
            axes (Optional[Sequence[float]]): Axes position [x0, y0, width, height]. Defaults to None.
            color (Optional[str]): Dendrogram line color. Defaults to None.
            linewidth (Optional[float]): Dendrogram line width. Defaults to None.
            data_pad (float): Padding around data in the y-direction. Defaults to 0.25.
        """
        self.axes = axes
        self.color = color
        self.linewidth = linewidth
        self.data_pad = data_pad

    def render(
        self,
        fig: plt.Figure,
        matrix: Matrix,
        layout: ClusterLayout,
        style: StyleConfig,
        **kwargs: Any,
    ) -> None:
        """
        Renders a dendrogram aligned to matrix rows.

        Args:
            fig (plt.Figure): Target figure.
            matrix (Matrix): Matrix object providing the row index.
            layout (ClusterLayout): Cluster layout providing `cluster_spans`.
            style (StyleConfig): Style configuration.

        Kwargs:
            **kwargs (Any): Renderer keyword arguments. Defaults to {}.
        """
        # Resolve configuration and create dendrogram axis
        cfg = _resolve_dendrogram_config(self, style)
        ax_dend = fig.add_axes(cfg["axes"], frameon=False)
        linkage_matrix = _resolve_linkage_matrix(kwargs)
        dendro = dendrogram(
            linkage_matrix,
            orientation="left",
            no_labels=True,
            color_threshold=-1,
            above_threshold_color="#888888",
            distance_sort=False,
            no_plot=True,
        )
        # Set up coordinate mapping and render dendrogram
        scale, offset, target_y_min, target_y_max = _compute_y_affine_map(
            dendro["icoord"],
            matrix.df.shape[0],
        )
        # Render dendrogram segments, cluster boundaries, and finalize axis
        _render_dendrogram_segments(
            ax_dend,
            dendro,
            scale,
            offset,
            color=cfg["color"],
            linewidth=cfg["linewidth"],
        )
        _render_cluster_boundaries(
            ax_dend,
            layout,
            target_y_max,
            kwargs.get("boundary_style"),
        )
        _finalize_dendrogram_axis(
            ax_dend,
            target_y_min=target_y_min,
            target_y_max=target_y_max,
            data_pad=cfg["data_pad"],
        )
