"""
himalayas/plot/renderers/label_legend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Sequence, TypedDict

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from ..style import StyleConfig


class LabelLegendItem(TypedDict, total=False):
    """
    Typed dictionary for one categorical legend item.
    """

    value: Any
    label: str
    color: Any


class LabelLegendSpec(TypedDict, total=False):
    """
    Typed dictionary for one categorical legend block.
    """

    name: str
    title: str
    items: Sequence[LabelLegendItem]
    nrows: Optional[int]
    ncols: Optional[int]
    row_pad: Optional[float]
    col_pad: Optional[float]


class LabelLegendLayout(TypedDict, total=False):
    """
    Typed dictionary for categorical legend-strip layout parameters.
    """

    height: float
    gap: float
    vpad: float
    title_pad: float
    swatch_scale: float
    fontsize: float
    font: Optional[str]
    color: str


def _resolve_label_legend_layout(
    layout: LabelLegendLayout,
    style: StyleConfig,
) -> LabelLegendLayout:
    """
    Resolves label-legend layout parameters with defaults.

    Args:
        layout (LabelLegendLayout): Initial layout parameters.
        style (StyleConfig): Plot style for defaults.

    Returns:
        LabelLegendLayout: Resolved layout parameters.
    """
    legend_layout = dict(layout)
    legend_layout.setdefault("height", 0.08)
    legend_layout.setdefault("gap", 0.01)
    legend_layout.setdefault("vpad", 0.01)
    legend_layout.setdefault("title_pad", 2.0)
    legend_layout.setdefault("swatch_scale", 0.75)

    fontsize = legend_layout.get("fontsize")
    if fontsize is None:
        fontsize = style.get("label_fontsize", 9)
    legend_layout["fontsize"] = fontsize

    text_color = legend_layout.get("color")
    if text_color is None:
        text_color = style.get("text_color", "black")
    legend_layout["color"] = text_color

    return legend_layout


def _resolve_grid(
    n_items: int,
    nrows: Optional[int],
    ncols: Optional[int],
) -> tuple[int, int]:
    """
    Resolves legend grid shape from item count and optional row/column hints.

    Args:
        n_items (int): Number of legend entries.
        nrows (Optional[int]): Requested number of rows.
        ncols (Optional[int]): Requested number of columns.

    Returns:
        tuple[int, int]: Resolved (nrows, ncols).

    Raises:
        ValueError: If `nrows * ncols` cannot fit all items.
    """
    if nrows is None and ncols is None:
        return 1, n_items
    if nrows is None:
        nrows = int(ceil(n_items / float(ncols)))
    elif ncols is None:
        ncols = int(ceil(n_items / float(nrows)))
    # Validation
    if int(nrows) * int(ncols) < n_items:
        raise ValueError("label legend grid cannot fit all categories; increase nrows or ncols")
    return int(nrows), int(ncols)


def _resolve_strip_geometry(
    ax: plt.Axes,
    layout: LabelLegendLayout,
    colorbar_layout: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Computes categorical legend-strip geometry.

    Args:
        ax (plt.Axes): Matrix axes used as horizontal anchor.
        layout (LabelLegendLayout): Resolved legend-strip layout.
        colorbar_layout (Optional[Dict[str, Any]]): Resolved colorbar layout.

    Returns:
        Dict[str, float]: Geometry for strip placement.
    """
    # Anchor the legend strip to matrix width and place it below the colorbar strip when present.
    bbox = ax.get_position()
    strip_x0 = bbox.x0
    strip_w = bbox.width
    anchor_y0 = bbox.y0
    if colorbar_layout is not None:
        anchor_y0 = bbox.y0 - float(colorbar_layout["height"]) - float(colorbar_layout["gap"])
    strip_h = float(layout["height"])
    strip_y0 = anchor_y0 - float(layout["gap"]) - strip_h
    return {
        "strip_x0": strip_x0,
        "strip_w": strip_w,
        "strip_y0": strip_y0,
        "strip_h": strip_h,
    }


def _title_fraction(
    fig: plt.Figure,
    *,
    block_height: float,
    fontsize: float,
    title_pad: float,
) -> float:
    """
    Converts title line-height + title pad from points to a block-relative fraction.

    Args:
        fig (plt.Figure): Matplotlib Figure.

    Kwargs:
        block_height (float): Block height in figure-relative units.
        fontsize (float): Title font size in points.
        title_pad (float): Title padding in points.

    Returns:
        float: Title height as a fraction of block height.
    """
    block_inches = max(fig.get_figheight() * block_height, 1e-12)
    line_inches = (fontsize * 1.2) / 72.0
    pad_inches = title_pad / 72.0
    return min(0.40, (line_inches + pad_inches) / block_inches)


def _render_block(
    fig: plt.Figure,
    *,
    x0: float,
    y0: float,
    w: float,
    h: float,
    block: Dict[str, Any],
    layout: LabelLegendLayout,
) -> None:
    """
    Renders one categorical legend block.

    Args:
        fig (plt.Figure): Matplotlib Figure.

    Kwargs:
        x0 (float): Block x-position.
        y0 (float): Block y-position.
        w (float): Block width.
        h (float): Block height.
        block (Dict[str, Any]): Resolved block specification.
        layout (LabelLegendLayout): Resolved legend-strip layout.

    Raises:
        ValueError: If computed cell geometry is invalid.
    """
    # Create a local axes for this legend block in normalized [0, 1] coordinates.
    ax_block = fig.add_axes([x0, y0, w, h], frameon=False)
    ax_block.set_xlim(0.0, 1.0)
    ax_block.set_ylim(0.0, 1.0)
    ax_block.set_xticks([])
    ax_block.set_yticks([])
    # Read block metadata and reserve optional vertical space for the title.
    nrows = int(block["nrows"])
    ncols = int(block["ncols"])
    name = str(block.get("name", "<unknown>"))
    title = str(block.get("title", ""))
    fontsize = float(layout["fontsize"])
    text_color = layout["color"]
    font = layout.get("font", None)
    title_frac = 0.0
    if title:
        title_frac = _title_fraction(
            fig,
            block_height=h,
            fontsize=fontsize,
            title_pad=float(layout["title_pad"]),
        )
        ax_block.text(
            0.0,
            1.0,
            title,
            ha="left",
            va="top",
            fontsize=fontsize,
            color=text_color,
            fontname=font if font is not None else None,
            clip_on=False,
        )

    content_top = 1.0 - title_frac
    content_h = content_top
    # Use per-legend row/column spacing when provided, otherwise keep v1 defaults.
    col_pad = block.get("col_pad", None)
    row_pad = block.get("row_pad", None)
    if ncols > 1:
        col_gap = float(col_pad) if col_pad is not None else 0.02
    else:
        col_gap = 0.0
    if nrows > 1:
        row_gap = float(row_pad) if row_pad is not None else 0.02
    else:
        row_gap = 0.0
    # Compute per-cell geometry after applying internal row/column gaps.
    cell_w = (1.0 - col_gap * (ncols - 1)) / float(ncols)
    cell_h = (content_h - row_gap * (nrows - 1)) / float(nrows)
    # Validation
    if cell_w <= 0 or cell_h <= 0:
        raise ValueError(
            "label legend geometry is too tight for "
            f"{name!r}; increase plot_label_legends(height=...)"
        )

    # Convert data-unit widths so swatches remain visually square in screen space.
    x_unit_in = w * fig.get_figwidth()
    y_unit_in = h * fig.get_figheight()
    xy_ratio = y_unit_in / max(x_unit_in, 1e-12)

    # Render each legend item into its grid cell as a color swatch plus label text.
    for idx, item in enumerate(block["items"]):
        r = idx // ncols
        c = idx % ncols
        if r >= nrows:
            break
        cell_x0 = c * (cell_w + col_gap)
        cell_y0 = content_top - (r + 1) * cell_h - r * row_gap

        swatch_h = min(
            cell_h * float(layout["swatch_scale"]),
            cell_h * 0.95,
        )
        swatch_w = min(swatch_h * xy_ratio, cell_w * 0.35)
        swatch_x0 = cell_x0
        swatch_y0 = cell_y0 + 0.5 * (cell_h - swatch_h)
        ax_block.add_patch(
            plt.Rectangle(
                (swatch_x0, swatch_y0),
                swatch_w,
                swatch_h,
                facecolor=item["color"],
                edgecolor="none",
                zorder=2,
            )
        )
        # Use a fixed swatch-to-text gap so labels align consistently across cells.
        text_gap = 0.012
        label = str(item.get("label", item.get("value", "")))
        ax_block.text(
            swatch_x0 + swatch_w + text_gap,
            cell_y0 + 0.5 * cell_h,
            label,
            ha="left",
            va="center",
            fontsize=fontsize,
            color=text_color,
            fontname=font if font is not None else None,
            clip_on=False,
        )


class LabelLegendRenderer:
    """
    Class for rendering stacked categorical legend blocks.
    """

    def __init__(
        self,
        specs: Iterable[LabelLegendSpec],
        layout: Optional[LabelLegendLayout] = None,
    ) -> None:
        """
        Initializes the LabelLegendRenderer instance.

        Args:
            specs (Iterable[LabelLegendSpec]): Legend block specifications.
            layout (Optional[LabelLegendLayout]): Strip layout parameters. Defaults to None.
        """
        self.specs = list(specs)
        self.layout = dict(layout) if layout is not None else {}

    def render(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        style: StyleConfig,
        *,
        colorbar_layout: Optional[Dict[str, Any]] = None,
    ) -> LabelLegendLayout:
        """
        Renders stacked categorical legend blocks below matrix/colorbars.

        Args:
            fig (plt.Figure): Matplotlib Figure.
            ax (plt.Axes): Matrix axes used as anchor.
            style (StyleConfig): Plot style for defaults.

        Kwargs:
            colorbar_layout (Optional[Dict[str, Any]]): Resolved colorbar layout.
                If provided, legends anchor below the colorbar strip.

        Returns:
            LabelLegendLayout: Resolved legend-strip layout.

        Raises:
            ValueError: If strip/block geometry cannot fit requested content.
        """
        # Resolve defaults once so downstream geometry and styling use concrete values.
        layout = _resolve_label_legend_layout(self.layout, style)
        if not self.specs:
            return layout

        # Normalize specs into renderable blocks and drop legends with no items.
        blocks: list[Dict[str, Any]] = []
        for spec in self.specs:
            items = list(spec.get("items", ()))
            if not items:
                continue
            nrows, ncols = _resolve_grid(
                n_items=len(items),
                nrows=spec.get("nrows", None),
                ncols=spec.get("ncols", None),
            )
            blocks.append(
                {
                    "name": spec.get("name", ""),
                    "title": spec.get("title", ""),
                    "nrows": nrows,
                    "ncols": ncols,
                    "row_pad": spec.get("row_pad", None),
                    "col_pad": spec.get("col_pad", None),
                    "items": items,
                }
            )
        if not blocks:
            return layout

        # Compute strip geometry and validate total vertical space across stacked blocks.
        geom = _resolve_strip_geometry(ax, layout, colorbar_layout)
        n_blocks = len(blocks)
        block_vpad = float(layout["vpad"])
        usable_h = geom["strip_h"] - block_vpad * (n_blocks - 1)
        if usable_h <= 0:
            raise ValueError(
                "label legend strip height is too small for requested vpad; "
                "increase plot_label_legends(height=...)"
            )

        # Allocate block heights proportionally to row counts for stable cell sizing.
        weights = [float(block["nrows"]) for block in blocks]
        total_weight = sum(weights)
        if total_weight <= 0:
            return layout

        # Stack legend blocks top-to-bottom with heights proportional to their row counts.
        y_cursor = geom["strip_y0"] + geom["strip_h"]
        for block, weight in zip(blocks, weights):
            block_h = usable_h * (weight / total_weight)
            block_y0 = y_cursor - block_h
            _render_block(
                fig,
                x0=geom["strip_x0"],
                y0=block_y0,
                w=geom["strip_w"],
                h=block_h,
                block=block,
                layout=layout,
            )
            y_cursor = block_y0 - block_vpad

        return layout
