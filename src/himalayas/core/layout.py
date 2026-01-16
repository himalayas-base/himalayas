"""Layout metadata derived from clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage

if TYPE_CHECKING:
    from .matrix import Matrix


@dataclass(frozen=True)
class ClusterLayout:
    """Read-only layout/geometry derived from a clustering."""

    leaf_order: np.ndarray
    ordered_labels: np.ndarray
    ordered_cluster_ids: np.ndarray
    cluster_spans: List[Tuple[int, int, int]]
    cluster_sizes: Dict[int, int]
    col_order: Optional[np.ndarray] = None


def compute_col_order(
    matrix: "Matrix",
    *,
    linkage_method: str = "ward",
    linkage_metric: str = "euclidean",
) -> np.ndarray:
    """Compute a dendrogram leaf order for matrix columns (visual grouping only)."""
    Z = linkage(
        matrix.values.T,
        method=linkage_method,
        metric=linkage_metric,
        optimal_ordering=True,
    )
    return leaves_list(Z)
