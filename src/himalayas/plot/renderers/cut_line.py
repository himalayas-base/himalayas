"""
himalayas/plot/renderers/cut_line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, AuxTransformBox
from matplotlib.patches import Circle
from matplotlib.transforms import Affine2D

# Scissors icon drawn on a 24-unit grid with y pointing down: two rings and three strokes.
_ICON_UNITS = 24.0
_ICON_RINGS = ((6.0, 6.0), (6.0, 18.0))
_ICON_RING_RADIUS = 3.0
_ICON_STROKES = (
    ((20.0, 4.0), (8.12, 15.88)),
    ((14.47, 14.48), (20.0, 20.0)),
    ((8.12, 8.12), (12.0, 12.0)),
)
_ICON_STROKE_UNITS = 2.2


def _build_scissors(
    fig: plt.Figure,
    *,
    color: str,
    size: float,
) -> AuxTransformBox:
    """
    Builds the scissors icon as a box of artists scaled to points.

    Args:
        fig (plt.Figure): Target figure (its dpi sets the icon scale).

    Kwargs:
        color (str): Icon color.
        size (float): Icon height and width in points.

    Returns:
        AuxTransformBox: Box holding the icon artists.
    """
    # Map icon units to inches (y flipped so the icon hangs downward), then to pixels.
    unit = size / 72.0 / _ICON_UNITS
    box = AuxTransformBox(Affine2D().scale(unit, -unit) + fig.dpi_scale_trans)
    linewidth = _ICON_STROKE_UNITS / _ICON_UNITS * size
    # Draw the two rings, then the blade and handle strokes.
    for center in _ICON_RINGS:
        box.add_artist(
            Circle(center, _ICON_RING_RADIUS, fill=False, edgecolor=color, linewidth=linewidth)
        )
    for (x0, y0), (x1, y1) in _ICON_STROKES:
        box.add_artist(
            Line2D(
                [x0, x1],
                [y0, y1],
                color=color,
                linewidth=linewidth,
                solid_capstyle="round",
            )
        )
    return box


def render_cut_line(
    ax_dend: plt.Axes,
    threshold: float,
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    scissors: bool,
    scissors_size: float,
) -> None:
    """
    Renders a vertical cut line at a dendrogram height, with optional scissors at its foot.

    Args:
        ax_dend (plt.Axes): Dendrogram axis.
        threshold (float): Dendrogram height at which to draw the line.

    Kwargs:
        color (str): Line and scissors color.
        linestyle (str): Matplotlib line style of the cut line.
        linewidth (float): Cut line width.
        scissors (bool): Whether to draw the scissors icon at the foot of the line.
        scissors_size (float): Scissors size in points.
    """
    # Span the full panel height: x in data units, y in axes fraction.
    transform = ax_dend.get_xaxis_transform()
    ax_dend.add_line(
        Line2D(
            [threshold, threshold],
            [0.0, 1.0],
            transform=transform,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            clip_on=False,
        )
    )
    if not scissors:
        return
    # Hang the scissors just below the foot of the line.
    ax_dend.add_artist(
        AnnotationBbox(
            _build_scissors(ax_dend.figure, color=color, size=scissors_size),
            (threshold, 0.0),
            xycoords=transform,
            box_alignment=(0.5, 1.0),
            frameon=False,
            annotation_clip=False,
            pad=0.0,
        )
    )
