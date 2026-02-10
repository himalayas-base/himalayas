"""
himalayas/plot/renderers/colorbar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Iterable, Optional, Dict, Sequence, TypedDict, TYPE_CHECKING, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Colormap, Normalize
from matplotlib.ticker import FuncFormatter

if TYPE_CHECKING:
    from ..style import StyleConfig


class ColorbarSpec(TypedDict, total=False):
    """
    Typed dictionary for colorbar specifications.
    """

    name: str
    cmap: Union[str, Colormap]
    norm: Normalize
    label: Optional[str]
    ticks: Optional[Sequence[float]]
    color: Optional[str]


class ColorbarLayout(TypedDict, total=False):
    """
    Typed dictionary for colorbar layout parameters.
    """

    nrows: int
    ncols: int
    height: float
    hpad: float
    vpad: float
    gap: float
    label_position: str
    border_color: str
    border_width: float
    border_alpha: float
    fontsize: float
    color: str
    font: Optional[str]
    tick_decimals: Optional[int]


def _max_decimal_formatter(max_decimals: int) -> FuncFormatter:
    """
    Creates a formatter that caps decimal precision and trims trailing zeros.

    Args:
        max_decimals (int): Maximum number of decimal places.

    Returns:
        FuncFormatter: Matplotlib tick formatter.
    """

    def _format_value(value: float, _pos: int) -> str:
        text = f"{value:.{max_decimals}f}".rstrip("0").rstrip(".")
        return "0" if text == "-0" else text

    return FuncFormatter(_format_value)


def _resolve_colorbar_layout(
    layout: ColorbarLayout,
    style: StyleConfig,
    n_colorbars: int,
) -> ColorbarLayout:
    """
    Resolves colorbar layout parameters with defaults.

    Args:
        layout (ColorbarLayout): Initial layout parameters.
        style (StyleConfig): Plot style for defaults.
        n_colorbars (int): Number of colorbars to arrange.

    Returns:
        ColorbarLayout: Resolved layout parameters.
    """
    # Determine number of rows and columns
    colorbar_layout = dict(layout)
    nrows = colorbar_layout.get("nrows")
    ncols = colorbar_layout.get("ncols")
    # Auto-calculate rows/columns if not specified
    if nrows is None and ncols is None:
        nrows, ncols = 1, n_colorbars
    elif nrows is None:
        nrows = int(np.ceil(n_colorbars / ncols))
    elif ncols is None:
        ncols = int(np.ceil(n_colorbars / nrows))
    # Set resolved values back into layout
    colorbar_layout["nrows"] = nrows
    colorbar_layout["ncols"] = ncols
    colorbar_layout.setdefault("height", 0.05)
    colorbar_layout.setdefault("hpad", 0.01)
    colorbar_layout.setdefault("vpad", 0.01)
    colorbar_layout.setdefault("gap", 0.02)
    colorbar_layout.setdefault("label_position", "below")
    colorbar_layout.setdefault("tick_decimals", None)
    # Resolve border properties
    border_color = colorbar_layout.get("border_color")
    if border_color is None:
        border_color = style.get("text_color", "black")
    colorbar_layout["border_color"] = border_color
    colorbar_layout.setdefault("border_width", 0.8)
    colorbar_layout.setdefault("border_alpha", 1.0)
    # Resolve font properties
    fontsize = colorbar_layout.get("fontsize")
    if fontsize is None:
        fontsize = style.get("label_fontsize", 9)
    colorbar_layout["fontsize"] = fontsize
    # Resolve text color
    text_color = colorbar_layout.get("color")
    if text_color is None:
        text_color = style.get("text_color", "black")
    colorbar_layout["color"] = text_color

    return colorbar_layout


def _compute_colorbar_geometry(
    ax: plt.Axes,
    layout: ColorbarLayout,
) -> Dict[str, float]:
    """
    Computes colorbar strip and cell geometry.

    Args:
        ax (plt.Axes): Matplotlib Axes to base geometry on.
        layout (ColorbarLayout): Resolved layout parameters.

    Returns:
        Dict[str, float]: Computed geometry with keys: strip_x0, strip_y0, cell_w, cell_h.
    """
    bbox = ax.get_position()
    # Compute strip geometry
    strip_h = layout["height"]
    strip_y0 = bbox.y0 - strip_h - layout["gap"]
    strip_x0 = bbox.x0
    strip_w = bbox.width
    # Compute cell geometry
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
    cb: ColorbarSpec,
    x0: float,
    y0: float,
    w: float,
    h: float,
    layout: ColorbarLayout,
) -> None:
    """
    Renders a single colorbar cell.

    Args:
        fig (plt.Figure): Matplotlib Figure.
        cb (ColorbarSpec): Colorbar parameters.
        x0 (float): X-coordinate of the cell.
        y0 (float): Y-coordinate of the cell.
        w (float): Width of the cell.
        h (float): Height of the cell.
        layout (ColorbarLayout): Layout parameters.
    """
    ax_cb = fig.add_axes([x0, y0, w, h], frameon=True)
    # Determine text color and font
    text_color = cb.get("color", layout["color"])
    if text_color is None:
        text_color = layout["color"]
    font = layout.get("font", None)
    # Create colorbar
    cbar = ColorbarBase(
        ax_cb,
        cmap=cb["cmap"],
        norm=cb["norm"],
        orientation="horizontal",
        ticks=cb.get("ticks", None),
    )
    tick_decimals = layout.get("tick_decimals", None)
    if tick_decimals is not None:
        cbar.formatter = _max_decimal_formatter(int(tick_decimals))
        cbar.update_ticks()
    # Style colorbar outline and ticks
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
    # Style ticks and labels
    ax_cb.tick_params(
        axis="x",
        labelsize=layout["fontsize"],
        colors=text_color,
    )
    ax_cb.set_yticks([])
    # Apply font to tick labels
    if font is not None:
        for t in ax_cb.get_xticklabels():
            t.set_fontname(font)
    # Set colorbar label
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
        colorbars: Iterable[ColorbarSpec],
        layout: Optional[ColorbarLayout] = None,
    ) -> None:
        """
        Initializes the ColorbarRenderer instance.

        Args:
            colorbars (Iterable[ColorbarSpec]): Colorbar specifications.
            layout (Optional[ColorbarLayout]): Layout parameters. Defaults to None.
        """
        self.colorbars = list(colorbars)
        self.layout = dict(layout) if layout is not None else {}

    def render(self, fig: plt.Figure, ax: plt.Axes, style: StyleConfig) -> ColorbarLayout:
        """
        Renders the colorbar strip.

        Args:
            fig (plt.Figure): Matplotlib Figure.
            ax (plt.Axes): Matplotlib Axes to base geometry on.
            style (StyleConfig): Plot style for defaults.

        Returns:
            ColorbarLayout: Resolved layout parameters.
        """
        layout = _resolve_colorbar_layout(self.layout, style, len(self.colorbars))
        geom = _compute_colorbar_geometry(ax, layout)
        # Render each colorbar cell
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
