"""
himalayas/plot/renderers
~~~~~~~~~~~~~~~~~~~~~~~~
"""

from .axes import AxesRenderer
from .base import BoundaryRegistry, Renderer
from .cluster_bar import render_cluster_bar_track
from .cluster_labels import ClusterLabelsRenderer
from .colorbar import ColorbarRenderer
from .dendrogram import DendrogramRenderer
from .matrix import MatrixRenderer
from .sigbar_legend import SigbarLegendRenderer

__all__ = [
    "AxesRenderer",
    "BoundaryRegistry",
    "render_cluster_bar_track",
    "ClusterLabelsRenderer",
    "ColorbarRenderer",
    "DendrogramRenderer",
    "MatrixRenderer",
    "SigbarLegendRenderer",
    "Renderer",
]
