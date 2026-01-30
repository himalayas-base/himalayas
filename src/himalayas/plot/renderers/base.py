"""Renderer protocol and shared helpers."""

from __future__ import annotations

from typing import Any, Protocol

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

from ..style import StyleConfig


class Renderer(Protocol):
    """Base protocol for plot layer renderers."""

    def render(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        matrix: Any,
        layout: Any,
        style: StyleConfig,
        **kwargs: Any,
    ) -> None:
        """Execute rendering logic."""
        ...


class BoundaryRegistry:
    """Deduplicate matrix-aligned boundary lines."""

    def __init__(self) -> None:
        self._boundaries: dict[float, tuple[float, str, float]] = {}

    def register(self, y: float, *, lw: float, color: str, alpha: float) -> None:
        """Register a boundary line, keeping the thickest line per y coordinate."""
        y = float(y)
        cur = self._boundaries.get(y)
        if cur is None:
            self._boundaries[y] = (float(lw), color, float(alpha))
            return
        cur_lw, _cur_color, _cur_alpha = cur
        if float(lw) > float(cur_lw):
            self._boundaries[y] = (float(lw), color, float(alpha))

    def render(self, ax: plt.Axes, x0: float, x1: float, *, zorder: int = 2) -> None:
        """Render all registered boundaries on the given axes."""
        if not self._boundaries:
            return
        ys = sorted(self._boundaries.keys())
        segments = [((x0, y), (x1, y)) for y in ys]
        lws = []
        cols = []
        for y in ys:
            lw, color, alpha = self._boundaries[y]
            lws.append(lw)
            cols.append(to_rgba(color, alpha))
        ax.add_collection(
            LineCollection(
                segments,
                linewidths=lws,
                colors=cols,
                zorder=zorder,
            )
        )
