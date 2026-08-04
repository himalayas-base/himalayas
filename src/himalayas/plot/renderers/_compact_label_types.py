"""
himalayas/plot/renderers/_compact_label_types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

# Leader-line shape, shared by Plotter.plot_cluster_labels_compact() and CompactLabelsRenderer.
LINE_SHAPES = {"straight", "curved", "elbow"}

# Leader-line style, shared by Plotter.plot_cluster_labels_compact() and CompactLabelsRenderer.
LINE_STYLES = {"solid", "dashed", "dotted"}

# Matrix-side (source) endpoint decoration for a compact-label leader line.
SOURCE_ENDS = {"tick", "span", "round", "none"}

# Table-side (target) endpoint decoration for a compact-label leader line.
TARGET_ENDS = {"tick", "arrow", "round", "none"}

# Cluster-abreast span/bracket kind, shared by Plotter.plot_cluster_labels() and
# _render_cluster_text_and_separators().
CLUSTER_SPANS = {"line", "bracket"}
