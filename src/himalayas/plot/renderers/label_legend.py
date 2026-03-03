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
    title_pad: float,
) -> float:
    """
    Converts title pad from points to a block-relative fraction.

    Args:
        fig (plt.Figure): Matplotlib Figure.

    Kwargs:
        block_height (float): Block height in figure-relative units.
        title_pad (float): Title padding in points.

    Returns:
        float: Title height as a fraction of block height.
    """
    block_inches = max(fig.get_figheight() * block_height, 1e-12)
    # Matplotlib text sizing APIs use points; typography defines 72 points per inch.
    pad_inches = title_pad / 72.0
    return max(0.0, pad_inches / block_inches)


def _measure_text_width_axes(
    ax: plt.Axes,
    text: str,
    *,
    fontsize: float,
    font: Optional[str],
) -> float:
    """
    Measures text width in x-axis normalized units for a legend block.

    Args:
        ax (plt.Axes): Legend block axes.
        text (str): Label text to measure.

    Kwargs:
        fontsize (float): Font size in points.
        font (Optional[str]): Font family name.

    Returns:
        float: Text width in normalized [0, 1] axes units.
    """
    renderer = ax.figure.canvas.get_renderer()
    probe = ax.text(
        0.0,
        0.0,
        text,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        color="black",
        fontname=font if font is not None else None,
        alpha=0.0,
        clip_on=False,
    )
    text_bbox = probe.get_window_extent(renderer=renderer)
    probe.remove()
    ax_bbox = ax.get_window_extent(renderer=renderer)
    return text_bbox.width / max(ax_bbox.width, 1e-12)


def _resolve_block_geometry(
    fig: plt.Figure,
    *,
    block_width: float,
    block_height: float,
    content_top: float,
    nrows: int,
    ncols: int,
    row_pad: Optional[float],
    col_pad: Optional[float],
    swatch_scale: float,
) -> dict[str, float]:
    """
    Resolves row and swatch geometry for one legend block.

    Args:
        fig (plt.Figure): Matplotlib Figure.

    Kwargs:
        block_width (float): Block width in figure-relative units.
        block_height (float): Block height in figure-relative units.
        content_top (float): Top of content area after title reservation.
        nrows (int): Requested row count.
        ncols (int): Requested column count.
        row_pad (Optional[float]): Optional override for row gap.
        col_pad (Optional[float]): Optional override for column gap.
        swatch_scale (float): Relative swatch size inside one row.

    Returns:
        dict[str, float]: Geometry values for row and swatch rendering.
    """
    # Compute row and column gaps with defaults that preserve visual balance across typical grid shapes.
    if ncols > 1:
        col_gap = float(col_pad) if col_pad is not None else 0.02
    else:
        col_gap = 0.0
    if nrows > 1:
        row_gap = float(row_pad) if row_pad is not None else 0.02
    else:
        row_gap = 0.0

    row_h = (content_top - row_gap * (nrows - 1)) / float(max(nrows, 1))
    # Clamp to a tiny positive value and let visual clipping reflect over-constrained settings.
    row_h = max(row_h, 1e-6)
    # Compute swatch size with respect to block dimensions and row height to maintain visual balance.
    x_unit_in = block_width * fig.get_figwidth()
    y_unit_in = block_height * fig.get_figheight()
    xy_ratio = y_unit_in / max(x_unit_in, 1e-12)
    swatch_h = min(
        row_h * swatch_scale,
        row_h * 0.95,
    )
    nominal_cell_w = 1.0 / float(max(ncols, 1))
    swatch_w = min(swatch_h * xy_ratio, nominal_cell_w * 0.35)

    return {
        "row_gap": row_gap,
        "col_gap": col_gap,
        "row_h": row_h,
        "swatch_h": swatch_h,
        "swatch_w": swatch_w,
        "text_gap": 0.012,
    }


def _render_justified_row(
    ax_block: plt.Axes,
    *,
    row_items: Sequence[LabelLegendItem],
    row_top: float,
    row_h: float,
    swatch_h: float,
    swatch_w: float,
    text_gap: float,
    col_gap: float,
    fontsize: float,
    text_color: str,
    font: Optional[str],
) -> None:
    """
    Renders one legend row with justified horizontal spacing.

    Args:
        ax_block (plt.Axes): Legend block axes.

    Kwargs:
        row_items (Sequence[LabelLegendItem]): Items assigned to this row.
        row_top (float): Row top y-position in axes coordinates.
        row_h (float): Row height in axes-relative units.
        swatch_h (float): Swatch height in axes-relative units.
        swatch_w (float): Swatch width in axes-relative units.
        text_gap (float): Gap between swatch and label text.
        col_gap (float): Minimum gap between adjacent row items.
        fontsize (float): Label font size.
        text_color (str): Label text color.
        font (Optional[str]): Label font family.
    """
    # Extract labels and measure text widths to inform horizontal justification with respect to
    # swatch size and column gap.
    labels = [str(item.get("label", item.get("value", ""))) for item in row_items]
    text_widths = [
        _measure_text_width_axes(
            ax_block,
            label,
            fontsize=fontsize,
            font=font,
        )
        for label in labels
    ]
    item_widths = [swatch_w + text_gap + text_w for text_w in text_widths]
    k = len(item_widths)

    # Compute horizontal gap to justify items across the row with at least col_gap spacing. If content
    # is too wide, gaps collapse to col_gap and overflow is visually apparent.
    gap = 0.0
    if k > 1:
        min_total = sum(item_widths) + col_gap * max(k - 1, 0)
        # Preserve minimum spacing and let overflow be visually apparent if content is too wide.
        gap = max(col_gap, col_gap + (1.0 - min_total) / float(k - 1))

    # Render items left-to-right with justified spacing and vertical centering within the row.
    x_cursor = 0.0
    for item, label, item_w in zip(row_items, labels, item_widths):
        # Top-pack row content so title_pad directly controls title-to-items spacing.
        swatch_y0 = row_top - swatch_h
        row_mid_y = swatch_y0 + 0.5 * swatch_h
        ax_block.add_patch(
            plt.Rectangle(
                (x_cursor, swatch_y0),
                swatch_w,
                swatch_h,
                facecolor=item["color"],
                edgecolor="none",
                zorder=2,
            )
        )
        ax_block.text(
            x_cursor + swatch_w + text_gap,
            row_mid_y,
            label,
            ha="left",
            va="center",
            fontsize=fontsize,
            color=text_color,
            fontname=font if font is not None else None,
            clip_on=False,
        )
        x_cursor += item_w + gap


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
    title = str(block.get("title", ""))
    fontsize = float(layout["fontsize"])
    text_color = layout["color"]
    font = layout.get("font", None)
    title_frac = 0.0
    if title:
        title_frac = _title_fraction(
            fig,
            block_height=h,
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

    # Resolve row and swatch geometry after reserving title space.
    content_top = max(0.0, 1.0 - title_frac)
    geom = _resolve_block_geometry(
        fig,
        block_width=w,
        block_height=h,
        content_top=content_top,
        nrows=nrows,
        ncols=ncols,
        row_pad=block.get("row_pad", None),
        col_pad=block.get("col_pad", None),
        swatch_scale=float(layout["swatch_scale"]),
    )

    # Group items by requested row-major grid rows.
    rows: list[list[LabelLegendItem]] = [[] for _ in range(nrows)]
    for idx, item in enumerate(block["items"]):
        r = idx // ncols
        if r >= nrows:
            break
        rows[r].append(item)

    # Prime the renderer once before text width measurements.
    fig.canvas.draw()

    # Render each row by justifying items across the row with col_gap as minimum spacing.
    for r, row_items in enumerate(rows):
        if not row_items:
            continue
        row_top = content_top - r * (geom["row_h"] + geom["row_gap"])
        _render_justified_row(
            ax_block,
            row_items=row_items,
            row_top=row_top,
            row_h=geom["row_h"],
            swatch_h=geom["swatch_h"],
            swatch_w=geom["swatch_w"],
            text_gap=geom["text_gap"],
            col_gap=geom["col_gap"],
            fontsize=fontsize,
            text_color=text_color,
            font=font,
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
            ValueError: If a legend grid declaration cannot fit requested items.
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
        # Clamp to a tiny positive strip height and let overflow/clipping remain visual.
        usable_h = max(usable_h, 1e-6)

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
