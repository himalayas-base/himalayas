"""
himalayas/plot/renderers/_cluster_label_types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Optional, Tuple

# (label, pval, qval, score, fe)
ClusterLabelStats = Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float]]
