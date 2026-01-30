"""Dendrogram renderer."""

from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from scipy.cluster.hierarchy import dendrogram


class DendrogramRenderer:
    """Render a dendrogram aligned to the matrix rows."""

    def __init__(
        self,
        *,
        axes: Optional[list[float]] = None,
        color: Optional[str] = None,
        linewidth: Optional[float] = None,
        data_pad: float = 0.25,
        **kwargs: Any,
    ) -> None:
        self.axes = axes
        self.color = color
        self.linewidth = linewidth
        self.data_pad = data_pad
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
        dendro_axes = self.axes if self.axes is not None else style["dendro_axes"]
        dendro_color = self.color if self.color is not None else style["dendro_color"]
        dendro_lw = self.linewidth if self.linewidth is not None else style["dendro_lw"]
        data_pad = self.data_pad

        ax_dend = fig.add_axes(dendro_axes, frameon=False)

        linkage_matrix = kwargs.get("linkage_matrix")
        if linkage_matrix is None:
            results = kwargs.get("results")
            if results is None or not hasattr(results, "clusters"):
                raise ValueError(
                    "Dendrogram rendering requires `results` or `linkage_matrix`."
                )
            linkage_matrix = results.clusters.linkage_matrix

        dendro = dendrogram(
            linkage_matrix,
            orientation="left",
            no_labels=True,
            color_threshold=-1,
            above_threshold_color="#888888",
            distance_sort=False,
            no_plot=True,
        )

        dendro_y_min = min(min(y) for y in dendro["icoord"])
        dendro_y_max = max(max(y) for y in dendro["icoord"])

        target_y_min = -0.5
        target_y_max = matrix.df.shape[0] - 0.5

        scale = (target_y_max - target_y_min) / (dendro_y_max - dendro_y_min)
        offset = target_y_min - scale * dendro_y_min

        for icoord, dcoord in zip(dendro["icoord"], dendro["dcoord"]):
            icoord_mapped = [scale * y + offset for y in icoord]
            ax_dend.plot(
                dcoord,
                icoord_mapped,
                color=dendro_color,
                linewidth=dendro_lw,
            )

        boundary_style = kwargs.get("boundary_style")
        if boundary_style is not None:
            ys = [
                s - 0.5
                for _cid, s, _e in layout.cluster_spans
                if (s - 0.5) > -0.5 and (s - 0.5) < target_y_max
            ]
            if ys:
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

        ax_dend.set_ylim(target_y_min - data_pad, target_y_max + data_pad)
        ax_dend.invert_yaxis()
        ax_dend.invert_xaxis()

        ax_dend.set_xticks([])
        ax_dend.set_yticks([])
        ax_dend.spines["top"].set_visible(False)
        ax_dend.spines["right"].set_visible(False)
        ax_dend.spines["bottom"].set_visible(False)
        ax_dend.spines["left"].set_visible(False)
