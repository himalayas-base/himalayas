"""Axis label and tick renderers."""

from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np


class AxesRenderer:
    """
    Class for rendering axis labels, ticks, and titles for plot layers.
    """

    def __init__(self, kind: str, **kwargs: Any) -> None:
        """
        Initializes the AxesRenderer instance.
        """
        self.kind = kind
        self.kwargs = dict(kwargs)

    def render(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        matrix: Any,
        layout: Any,
        style: Any,
        **kwargs: Any,
    ) -> None:
        if self.kind == "matrix_axis_labels":
            self._render_axis_labels(fig, ax, style)
            return
        if self.kind == "row_ticks":
            self._render_row_ticks(fig, ax, matrix, layout)
            return
        if self.kind == "col_ticks":
            self._render_col_ticks(fig, ax, matrix, layout)
            return
        if self.kind == "title":
            self._render_title(ax, style)
            return
        raise NotImplementedError(f"Unknown axes layer: {self.kind}")

    def _render_axis_labels(self, fig: plt.Figure, ax: plt.Axes, style: Any) -> None:
        xlabel = self.kwargs.get("xlabel", "")
        ylabel = self.kwargs.get("ylabel", "")
        fontsize = self.kwargs.get("fontsize", 12)
        fontweight = self.kwargs.get("fontweight", "normal")
        xlabel_pad = self.kwargs.get("xlabel_pad", 8)
        font = self.kwargs.get("font", None)
        color = self.kwargs.get("color", style.get("text_color", "black"))
        alpha = self.kwargs.get("alpha", 1.0)

        txt_xlabel = ax.set_xlabel(
            xlabel,
            fontweight=fontweight,
            labelpad=xlabel_pad,
        )
        self._apply_text_style(
            txt_xlabel,
            font=font,
            fontsize=fontsize,
            color=color,
            alpha=alpha,
            fontweight=fontweight,
        )

        if isinstance(ylabel, str) and ylabel.strip():
            bbox = ax.get_position()
            pad_frac = self.kwargs.get("ylabel_pad", style.get("ylabel_pad", 0.015))
            x = bbox.x1 + pad_frac
            y = bbox.y0
            width = 0.015
            height = bbox.height
            ax_ylabel = fig.add_axes([x, y, width, height], frameon=False)
            ax_ylabel.set_xticks([])
            ax_ylabel.set_yticks([])
            for spine in ax_ylabel.spines.values():
                spine.set_visible(False)
            text_kwargs = {
                "fontname": font if font is not None else "Helvetica",
                "fontsize": fontsize,
                "color": color,
                "alpha": alpha,
                "fontweight": fontweight,
            }
            ax_ylabel.text(
                0.5,
                0.5,
                ylabel,
                transform=ax_ylabel.transAxes,
                rotation=90,
                va="center",
                ha="center",
                **text_kwargs,
            )

    def _render_row_ticks(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        matrix: Any,
        layout: Any,
    ) -> None:
        labels = self.kwargs.get("labels", None)
        font_size = self.kwargs.get("fontsize", 9)
        max_labels = self.kwargs.get("max_labels", None)
        position = self.kwargs.get("position", "right")

        row_order = layout.leaf_order
        base_labels = labels if labels is not None else list(matrix.df.index)
        ordered_labels = np.array(base_labels)[row_order]
        n = len(ordered_labels)

        visible = np.ones(n, dtype=bool)
        if max_labels is not None and max_labels < n:
            idxs = np.linspace(0, n - 1, num=max_labels, dtype=int)
            visible[:] = False
            visible[idxs] = True

        bbox = ax.get_position()
        ax_row = fig.add_axes(bbox, frameon=False, zorder=10)
        ax_row.set_xlim(ax.get_xlim())
        ax_row.set_ylim(ax.get_ylim())

        ax_row.set_yticks(np.arange(n))
        ax_row.set_yticklabels(ordered_labels, fontsize=font_size)
        ax_row.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            labelleft=(position == "left"),
            labelright=(position == "right"),
        )
        ax_row.set_xticks([])

        for tick, vis in zip(ax_row.get_yticklabels(), visible):
            tick.set_visible(vis)

    def _render_col_ticks(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        matrix: Any,
        layout: Any,
    ) -> None:
        labels = self.kwargs.get("labels", None)
        font_size = self.kwargs.get("fontsize", 9)
        rotation = self.kwargs.get("rotation", 90)
        max_labels = self.kwargs.get("max_labels", None)
        position = self.kwargs.get("position", "top")

        base_labels = labels if labels is not None else list(matrix.df.columns)
        if layout.col_order is not None:
            ordered_labels = np.array(base_labels)[layout.col_order]
        else:
            ordered_labels = np.array(base_labels)
        n = len(ordered_labels)

        visible = np.ones(n, dtype=bool)
        if max_labels is not None and max_labels < n:
            idxs = np.linspace(0, n - 1, num=max_labels, dtype=int)
            visible[:] = False
            visible[idxs] = True

        bbox = ax.get_position()
        ax_col = fig.add_axes(bbox, frameon=False, zorder=10)
        ax_col.set_xlim(ax.get_xlim())
        ax_col.set_ylim(ax.get_ylim())

        ax_col.set_xticks(np.arange(n))
        ax_col.set_xticklabels(ordered_labels, fontsize=font_size, rotation=rotation)
        ax_col.tick_params(
            axis="x",
            which="both",
            top=False,
            bottom=False,
            labeltop=(position == "top"),
            labelbottom=(position == "bottom"),
        )
        ax_col.set_yticks([])

        for tick, vis in zip(ax_col.get_xticklabels(), visible):
            tick.set_visible(vis)

    def _render_title(self, ax: plt.Axes, style: Any) -> None:
        ax.set_title(
            self.kwargs["title"],
            fontsize=self.kwargs.get("fontsize", style.get("title_fontsize", 14)),
            pad=self.kwargs.get("pad", style.get("title_pad", 15)),
            color=self.kwargs.get("color", style.get("text_color", "black")),
        )

    @staticmethod
    def _apply_text_style(
        text_obj,
        *,
        font: Optional[str] = None,
        fontsize: Optional[float] = None,
        color: Optional[str] = None,
        alpha: Optional[float] = None,
        fontweight: Optional[str] = None,
    ) -> None:
        if text_obj is None:
            return
        if font is not None:
            if hasattr(text_obj, "set_fontfamily"):
                text_obj.set_fontfamily(font)
            elif hasattr(text_obj, "set_fontname"):
                text_obj.set_fontname(font)
        if fontsize is not None:
            text_obj.set_fontsize(fontsize)
        if color is not None:
            text_obj.set_color(color)
        if alpha is not None:
            text_obj.set_alpha(alpha)
        if fontweight is not None:
            text_obj.set_fontweight(fontweight)
