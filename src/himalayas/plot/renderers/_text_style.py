"""
himalayas/plot/renderers/_text_style
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt


def apply_text_style(
    text_obj: plt.Text,
    *,
    font: Optional[str] = None,
    fontsize: Optional[float] = None,
    color: Optional[str] = None,
    alpha: Optional[float] = None,
    fontweight: Optional[str] = None,
) -> None:
    """
    Applies text styling directly to a Matplotlib Text object.

    Using direct set_* calls rather than passing text properties as kwargs to
    the originating function (e.g. ax.set_title, ax.text) ensures reliable
    dispatch across all backends, including inline Jupyter renderers.

    Args:
        text_obj (plt.Text): Matplotlib Text object to style.

    Kwargs:
        font (Optional[str]): Font family or name. Defaults to None.
        fontsize (Optional[float]): Font size in points. Defaults to None.
        color (Optional[str]): Text color. Defaults to None.
        alpha (Optional[float]): Text transparency. Defaults to None.
        fontweight (Optional[str]): Font weight. Defaults to None.
    """
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
