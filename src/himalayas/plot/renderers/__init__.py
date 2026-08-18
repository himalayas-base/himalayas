"""
himalayas/plot/renderers
~~~~~~~~~~~~~~~~~~~~~~~~
"""

from .axes import AxesRenderer
from .base import BoundaryRegistry
from .cluster_bar import render_cluster_bar_track
from .cluster_labels import ClusterLabelsRenderer
from .colorbar import ColorbarRenderer
from .compact_labels import CompactLabelsRenderer
from .dendrogram import DendrogramRenderer
from .label_legend import LabelLegendRenderer
from .matrix import MatrixRenderer

__all__ = [
    "AxesRenderer",
    "BoundaryRegistry",
    "render_cluster_bar_track",
    "ClusterLabelsRenderer",
    "ColorbarRenderer",
    "CompactLabelsRenderer",
    "DendrogramRenderer",
    "LabelLegendRenderer",
    "MatrixRenderer",
]
