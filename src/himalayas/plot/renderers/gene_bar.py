"""Gene annotation bar renderer."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np


class GeneBarRenderer:
    """Render a row-level gene annotation bar."""

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
        ax: plt.Axes,
        matrix: Any,
        layout: Any,
        style: Any,
        **kwargs: Any,
    ) -> None:
        if not isinstance(self.values, dict):
            raise TypeError(
                "plot_gene_bar expects `values` as a dict mapping row IDs to values"
            )

        mode = self.mode
        missing_color = (
            self.missing_color
            if self.missing_color is not None
            else style["gene_bar_missing_color"]
        )

        dendro_axes = style["dendro_axes"]
        gap = (
            self.gene_bar_gap
            if self.gene_bar_gap is not None
            else style["gene_bar_gap"]
        )
        bar_w = (
            self.gene_bar_width
            if self.gene_bar_width is not None
            else style["gene_bar_width"]
        )

        x0 = dendro_axes[0] + dendro_axes[2] + gap
        bar_axes = (
            self.axes
            if self.axes is not None
            else [x0, dendro_axes[1], bar_w, dendro_axes[3]]
        )

        ax_bar = fig.add_axes(bar_axes, frameon=False)
        ax_bar.set_xlim(0, 1)
        n_rows = matrix.df.shape[0]
        ax_bar.set_ylim(-0.5, n_rows - 0.5)
        ax_bar.invert_yaxis()
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])

        row_ids = matrix.df.index.to_numpy()[layout.leaf_order]

        if mode == "categorical":
            colors = self.colors
            if colors is None or not isinstance(colors, dict):
                raise TypeError(
                    "categorical gene_bar requires `colors` as a dict mapping category -> color"
                )
            for i, gid in enumerate(row_ids):
                cat = self.values.get(gid, None)
                c = colors.get(cat, missing_color)
                ax_bar.add_patch(
                    plt.Rectangle(
                        (0.0, i - 0.5),
                        1.0,
                        1.0,
                        facecolor=c,
                        edgecolor="none",
                    )
                )
        elif mode == "continuous":
            cmap = plt.get_cmap(self.cmap)
            vals = np.array(
                [self.values.get(gid, np.nan) for gid in row_ids],
                dtype=float,
            )

            finite = np.isfinite(vals)
            if not np.any(finite):
                for i in range(len(row_ids)):
                    ax_bar.add_patch(
                        plt.Rectangle(
                            (0.0, i - 0.5),
                            1.0,
                            1.0,
                            facecolor=missing_color,
                            edgecolor="none",
                        )
                    )
            else:
                vmin = (
                    self.vmin
                    if self.vmin is not None
                    else float(np.nanmin(vals[finite]))
                )
                vmax = (
                    self.vmax
                    if self.vmax is not None
                    else float(np.nanmax(vals[finite]))
                )
                if vmin == vmax:
                    vmax = vmin + 1e-12
                norm = plt.Normalize(vmin=vmin, vmax=vmax)

                for i, v in enumerate(vals):
                    c = missing_color if not np.isfinite(v) else cmap(norm(v))
                    ax_bar.add_patch(
                        plt.Rectangle(
                            (0.0, i - 0.5),
                            1.0,
                            1.0,
                            facecolor=c,
                            edgecolor="none",
                        )
                    )
        else:
            raise ValueError("gene_bar mode must be 'categorical' or 'continuous'")


def render_gene_bar_track(
    ax: plt.Axes,
    x0: float,
    width: float,
    payload: dict[str, Any],
    matrix: Any,
    row_order,
    style: Any,
) -> None:
    """Render a gene bar track inside the label panel."""
    values = payload["values"]
    mode = payload.get("mode", "categorical")
    missing_color = payload.get("missing_color", style["gene_bar_missing_color"])
    row_ids = matrix.df.index.to_numpy()[row_order]
    if mode == "categorical":
        colors = payload.get("colors", None)
        if colors is None or not isinstance(colors, dict):
            raise TypeError("categorical label_panel gene bar requires `colors` as a dict")
        for i, gid in enumerate(row_ids):
            cat = values.get(gid, None)
            c = colors.get(cat, missing_color)
            ax.add_patch(
                plt.Rectangle(
                    (x0, i - 0.5),
                    width,
                    1.0,
                    facecolor=c,
                    edgecolor="none",
                    zorder=2,
                )
            )
    elif mode == "continuous":
        cmap = plt.get_cmap(payload.get("cmap", "viridis"))
        vals = np.array(
            [values.get(gid, np.nan) for gid in row_ids],
            dtype=float,
        )
        finite = np.isfinite(vals)
        if not np.any(finite):
            for i in range(len(row_ids)):
                ax.add_patch(
                    plt.Rectangle(
                        (x0, i - 0.5),
                        width,
                        1.0,
                        facecolor=missing_color,
                        edgecolor="none",
                        zorder=2,
                    )
                )
        else:
            vmin = payload.get(
                "vmin",
                float(np.nanmin(vals[finite])),
            )
            vmax = payload.get(
                "vmax",
                float(np.nanmax(vals[finite])),
            )
            if vmin == vmax:
                vmax = vmin + 1e-12
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            for i, v in enumerate(vals):
                c = missing_color if not np.isfinite(v) else cmap(norm(v))
                ax.add_patch(
                    plt.Rectangle(
                        (x0, i - 0.5),
                        width,
                        1.0,
                        facecolor=c,
                        edgecolor="none",
                        zorder=2,
                    )
                )
    else:
        raise ValueError(
            "label_panel gene bar mode must be 'categorical' or 'continuous'"
        )
