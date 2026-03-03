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
from .label_legend import LabelLegendRenderer
from .matrix import MatrixRenderer

__all__ = [
    "AxesRenderer",
    "BoundaryRegistry",
    "render_cluster_bar_track",
    "ClusterLabelsRenderer",
    "ColorbarRenderer",
    "DendrogramRenderer",
    "LabelLegendRenderer",
    "MatrixRenderer",
    "Renderer",
]
