"""Plot layer renderers."""

from .base import BoundaryRegistry, Renderer
from .axes import AxesRenderer
from .dendrogram import DendrogramRenderer
from .matrix import MatrixRenderer

__all__ = [
    "AxesRenderer",
    "BoundaryRegistry",
    "DendrogramRenderer",
    "MatrixRenderer",
    "Renderer",
]
