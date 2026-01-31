"""
himalayas/plot/renderers/colorbar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase


def _resolve_colorbar_layout(
    layout: Dict[str, Any],
    style: Any,
    n_colorbars: int,
) -> Dict[str, Any]:
    """
    Resolves colorbar layout parameters with defaults.

    Args:
        layout (Dict[str, Any]): Initial layout parameters.
        style (Any): Plot style for defaults.
        n_colorbars (int): Number of colorbars to arrange.

    Returns:
        Dict[str, Any]: Resolved layout parameters.
    """
    colorbar_layout = dict(layout)

    nrows = colorbar_layout.get("nrows")
    ncols = colorbar_layout.get("ncols")

    if nrows is None and ncols is None:
        nrows, ncols = 1, n_colorbars
    elif nrows is None:
        nrows = int(np.ceil(n_colorbars / ncols))
    elif ncols is None:
        ncols = int(np.ceil(n_colorbars / nrows))

    colorbar_layout["nrows"] = nrows
    colorbar_layout["ncols"] = ncols

    colorbar_layout.setdefault("height", 0.05)
    colorbar_layout.setdefault("hpad", 0.01)
    colorbar_layout.setdefault("vpad", 0.01)
    colorbar_layout.setdefault("gap", 0.02)
    colorbar_layout.setdefault("label_position", "below")

    border_color = colorbar_layout.get("border_color")
    if border_color is None:
        border_color = style.get("text_color", "black")
    colorbar_layout["border_color"] = border_color
    colorbar_layout.setdefault("border_width", 0.8)
    colorbar_layout.setdefault("border_alpha", 1.0)

    fontsize = colorbar_layout.get("fontsize")
    if fontsize is None:
        fontsize = style.get("label_fontsize", 9)
    colorbar_layout["fontsize"] = fontsize

    text_color = colorbar_layout.get("color")
    if text_color is None:
        text_color = style.get("text_color", "black")
    colorbar_layout["color"] = text_color

    return colorbar_layout


def _compute_colorbar_geometry(
    ax: plt.Axes,
    layout: Dict[str, Any],
) -> Dict[str, float]:
    """
    Computes colorbar strip and cell geometry.

    Args:
        ax (plt.Axes): Matplotlib Axes to base geometry on.
        layout (Dict[str, Any]): Resolved layout parameters.

    Returns:
        Dict[str, float]: Computed geometry with keys: strip_x0, strip_y0, cell_w, cell_h.
    """
    bbox = ax.get_position()

    strip_h = layout["height"]
    strip_y0 = bbox.y0 - strip_h - layout["gap"]
    strip_x0 = bbox.x0
    strip_w = bbox.width

    nrows = layout["nrows"]
    ncols = layout["ncols"]

    cell_w = (strip_w - layout["hpad"] * (ncols - 1)) / ncols
    cell_h = (strip_h - layout["vpad"] * (nrows - 1)) / nrows

    return {
        "strip_x0": strip_x0,
        "strip_y0": strip_y0,
        "cell_w": cell_w,
        "cell_h": cell_h,
    }


def _render_colorbar_cell(
    fig: plt.Figure,
    cb: Dict[str, Any],
    x0: float,
    y0: float,
    w: float,
    h: float,
    layout: Dict[str, Any],
) -> None:
    """
    Renders a single colorbar cell.

    Args:
        fig (plt.Figure): Matplotlib Figure.
        cb (Dict[str, Any]): Colorbar parameters.
        x0 (float): X-coordinate of the cell.
        y0 (float): Y-coordinate of the cell.
        w (float): Width of the cell.
        h (float): Height of the cell.
        layout (Dict[str, Any]): Layout parameters.
    """
    ax_cb = fig.add_axes([x0, y0, w, h], frameon=True)

    text_color = cb.get("color", layout["color"])
    if text_color is None:
        text_color = layout["color"]
    font = layout.get("font", None)

    cbar = ColorbarBase(
        ax_cb,
        cmap=cb["cmap"],
        norm=cb["norm"],
        orientation="horizontal",
        ticks=cb.get("ticks", None),
    )

    outline = getattr(cbar, "outline", None)
    if outline is not None:
        outline.set_edgecolor(layout["border_color"])
        outline.set_linewidth(layout["border_width"])
        outline.set_alpha(layout["border_alpha"])
    else:
        for spine in ax_cb.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(layout["border_width"])
            spine.set_edgecolor(layout["border_color"])
            spine.set_alpha(layout["border_alpha"])

    ax_cb.tick_params(
        axis="x",
        labelsize=layout["fontsize"],
        colors=text_color,
    )
    ax_cb.set_yticks([])

    if font is not None:
        for t in ax_cb.get_xticklabels():
            t.set_fontname(font)

    label = cb.get("label")
    if label:
        if layout["label_position"] == "below":
            ax_cb.set_xlabel(
                label,
                fontsize=layout["fontsize"],
                color=text_color,
                labelpad=2,
                fontname=font if font is not None else None,
            )
        else:
            ax_cb.set_title(
                label,
                fontsize=layout["fontsize"],
                color=text_color,
                pad=2,
                fontname=font if font is not None else None,
            )


class ColorbarRenderer:
    """
    Class for rendering the bottom colorbar strip.
    """

    def __init__(
        self,
        colorbars: Iterable[Dict[str, Any]],
        layout: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the ColorbarRenderer instance.
        """
        self.colorbars = list(colorbars)
        self.layout = dict(layout) if layout is not None else {}

    def render(self, fig: plt.Figure, ax: plt.Axes, style: Any) -> Dict[str, Any]:
        """
        Renders the colorbar strip.

        Args:
            fig (plt.Figure): Matplotlib Figure.
            ax (plt.Axes): Matplotlib Axes to base geometry on.
            style (Any): Plot style for defaults.

        Returns:
            Dict[str, Any]: Resolved layout parameters.
        """
        layout = _resolve_colorbar_layout(self.layout, style, len(self.colorbars))
        geom = _compute_colorbar_geometry(ax, layout)

        for i, cb in enumerate(self.colorbars):
            r = i // layout["ncols"]
            c = i % layout["ncols"]

            x0 = geom["strip_x0"] + c * (geom["cell_w"] + layout["hpad"])
            y0 = geom["strip_y0"] + (layout["nrows"] - 1 - r) * (geom["cell_h"] + layout["vpad"])

            _render_colorbar_cell(
                fig,
                cb,
                x0,
                y0,
                geom["cell_w"],
                geom["cell_h"],
                layout,
            )

        return layout
