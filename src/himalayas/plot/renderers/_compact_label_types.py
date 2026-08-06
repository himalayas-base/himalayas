"""
himalayas/plot/renderers/_compact_label_types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

# Leader-line shape, shared by Plotter.plot_cluster_labels_compact() and CompactLabelsRenderer.
LINE_SHAPES = {"straight", "curved", "elbow"}

# Leader-line style, shared by Plotter.plot_cluster_labels_compact() and CompactLabelsRenderer.
LINE_STYLES = {"solid", "dashed", "dotted"}

# Cluster-side identity marker text mode for a compact-label leader line.
CLUSTER_MARKERS = {"alpha", "cid"}

# Connector-start point decoration for a compact-label leader line, used when
# cluster_span is None (i.e. there is no cluster-span to anchor to).
LINE_STARTS = {"tick", "round", "none"}

# Table-side (end) endpoint decoration for a compact-label leader line.
LINE_ENDS = {"tick", "arrow", "round", "none"}

# Cluster-abreast span kind, shared by Plotter.plot_cluster_labels() and
# _render_cluster_text_and_separators(). Optional end caps are controlled by
# cluster_span_cap_width, not a separate span kind.
CLUSTER_SPANS = {"line"}
