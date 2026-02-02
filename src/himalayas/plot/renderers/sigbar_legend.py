"""
himalayas/plot/renderers/sigbar_legend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ..style import StyleConfig


class SigbarLegendRenderer:
    """
    Class for rendering the significance bar legend.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the SigbarLegendRenderer instance.
        """
        self.kwargs = dict(kwargs)

    def render(self, fig: plt.Figure, style: StyleConfig) -> None:
        """
        Renders the significance bar legend.

        Args:
            fig (plt.Figure): Matplotlib Figure.
            style (StyleConfig): Plot style configuration.
        """
        kwargs = self.kwargs
        cmap = plt.get_cmap(kwargs.get("sigbar_cmap", style.get("sigbar_cmap", "YlOrBr")))
        norm = kwargs.get("norm", None)
        # Set legend limits
        if norm is not None and hasattr(norm, "vmin") and hasattr(norm, "vmax"):
            lo = float(norm.vmin)
            hi = float(norm.vmax)
        else:
            lo = kwargs.get("sigbar_min_logp", style.get("sigbar_min_logp", 2.0))
            hi = kwargs.get("sigbar_max_logp", style.get("sigbar_max_logp", 10.0))
        # Draw legend
        legend_axes = kwargs.get("axes", [0.92, 0.20, 0.015, 0.25])
        ax_leg = fig.add_axes(legend_axes, frameon=False)
        grad = np.linspace(0, 1, 256).reshape(-1, 1)
        ax_leg.imshow(grad, aspect="auto", cmap=cmap, origin="lower")
        ax_leg.set_yticks([0, 255])
        ax_leg.set_yticklabels([f"{lo:g}", f"{hi:g}"])
        ax_leg.set_ylabel("-log10(p)", fontsize=8)
        ax_leg.set_xticks([])
        ax_leg.tick_params(axis="y", labelsize=7)
        for spine in ax_leg.spines.values():
            spine.set_visible(False)
