"""Colorbar strip renderer."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
from matplotlib.colorbar import ColorbarBase


class ColorbarRenderer:
    """Render the bottom colorbar strip."""

    def __init__(
        self,
        colorbars: Iterable[dict[str, Any]],
        layout: Optional[dict[str, Any]] = None,
    ) -> None:
        self.colorbars = list(colorbars)
        self.layout = dict(layout) if layout is not None else {}

    def render(self, fig, ax, style: Any) -> dict[str, Any]:
        colorbar_layout = dict(self.layout)
        nrows = colorbar_layout.get("nrows")
        ncols = colorbar_layout.get("ncols")
        height = colorbar_layout.get("height", 0.05)
        hpad = colorbar_layout.get("hpad", 0.01)
        vpad = colorbar_layout.get("vpad", 0.01)
        gap = colorbar_layout.get("gap", 0.02)
        label_position = colorbar_layout.get("label_position", "below")

        border_color = colorbar_layout.get("border_color")
        if border_color is None:
            border_color = style.get("text_color", "black")
        border_width = colorbar_layout.get("border_width", 0.8)
        border_alpha = colorbar_layout.get("border_alpha", 1.0)

        fontsize = colorbar_layout.get("fontsize")
        if fontsize is None:
            fontsize = style.get("label_fontsize", 9)

        text_color = colorbar_layout.get("color")
        if text_color is None:
            text_color = style.get("text_color", "black")

        font = colorbar_layout.get("font", None)

        N = len(self.colorbars)
        if nrows is None and ncols is None:
            nrows, ncols = 1, N
        elif nrows is None:
            nrows = int(np.ceil(N / ncols))
        elif ncols is None:
            ncols = int(np.ceil(N / nrows))

        bbox = ax.get_position()
        strip_y0 = bbox.y0 - height - gap
        strip_x0 = bbox.x0
        strip_w = bbox.width
        strip_h = height

        cell_w = (strip_w - hpad * (ncols - 1)) / ncols
        cell_h = (strip_h - vpad * (nrows - 1)) / nrows

        for i, cb in enumerate(self.colorbars):
            r = i // ncols
            c = i % ncols

            x0 = strip_x0 + c * (cell_w + hpad)
            y0 = strip_y0 + (nrows - 1 - r) * (cell_h + vpad)

            ax_cb = fig.add_axes([x0, y0, cell_w, cell_h], frameon=True)
            bar_text_color = cb.get("color")
            if bar_text_color is None:
                bar_text_color = text_color

                cbar = ColorbarBase(
                    ax_cb,
                    cmap=cb["cmap"],
                    norm=cb["norm"],
                    orientation="horizontal",
                    ticks=cb.get("ticks", None),
                )
                outline = getattr(cbar, "outline", None)
                if outline is not None:
                    outline.set_edgecolor(border_color)
                    outline.set_linewidth(border_width)
                    outline.set_alpha(border_alpha)
                else:
                    for spine in ax_cb.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(border_width)
                        spine.set_edgecolor(border_color)
                        spine.set_alpha(border_alpha)

            ax_cb.tick_params(
                axis="x",
                labelsize=fontsize,
                colors=bar_text_color,
            )
            ax_cb.set_yticks([])

            if font is not None:
                for t in ax_cb.get_xticklabels():
                    t.set_fontname(font)

            label = cb.get("label")
            if label:
                if label_position == "below":
                    ax_cb.set_xlabel(
                        label,
                        fontsize=fontsize,
                        color=bar_text_color,
                        labelpad=2,
                        fontname=font if font is not None else None,
                    )
                else:
                    ax_cb.set_title(
                        label,
                        fontsize=fontsize,
                        color=bar_text_color,
                        pad=2,
                        fontname=font if font is not None else None,
                    )
        return colorbar_layout
