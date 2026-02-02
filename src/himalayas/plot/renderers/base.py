"""
himalayas/plot/renderers/base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, Protocol

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

from ..style import StyleConfig


class Renderer(Protocol):
    """
    Class for defining the renderer interface used by plot layers.
    Protocol only; implement in concrete renderers.
    """

    def render(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        matrix: Any,
        layout: Any,
        style: StyleConfig,
        **kwargs: Any,
    ) -> None:
        """Executes rendering logic."""
        # Protocol stub; no runtime implementation
        ...


class BoundaryRegistry:
    """
    Class for deduplicating and rendering matrix-aligned boundary lines.
    """

    def __init__(self) -> None:
        """
        Initializes the BoundaryRegistry instance.
        """
        self._boundaries: dict[float, tuple[float, str, float]] = {}

    def register(self, y: float, *, lw: float, color: str, alpha: float) -> None:
        """
        Registers a boundary line, keeping the thickest line per y coordinate.

        Args:
            y (float): Y coordinate of the boundary line.
            lw (float): Line width.
            color (str): Line color.
            alpha (float): Line alpha (opacity).
        """
        # Store the thickest line for each y coordinate
        y = float(y)
        cur = self._boundaries.get(y)
        if cur is None:
            self._boundaries[y] = (float(lw), color, float(alpha))
            return
        # Compare line widths and keep the thicker one
        cur_lw, _cur_color, _cur_alpha = cur
        if float(lw) > float(cur_lw):
            self._boundaries[y] = (float(lw), color, float(alpha))

    def render(self, ax: plt.Axes, x0: float, x1: float, *, zorder: int = 2) -> None:
        """
        Renders all registered boundaries on the given axes.

        Args:
            ax (plt.Axes): Axes to render on.
            x0 (float): Left x coordinate.
            x1 (float): Right x coordinate.
            zorder (int): Z-order for rendering. Defaults to 2.
        """
        if not self._boundaries:
            return
        # Prepare line segments and their styles
        ys = sorted(self._boundaries.keys())
        segments = [((x0, y), (x1, y)) for y in ys]
        lws = []
        cols = []
        for y in ys:
            lw, color, alpha = self._boundaries[y]
            lws.append(lw)
            cols.append(to_rgba(color, alpha))
        # Add line collection to axes
        ax.add_collection(
            LineCollection(
                segments,
                linewidths=lws,
                colors=cols,
                zorder=zorder,
            )
        )
