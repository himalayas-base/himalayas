"""
himalayas/core/layout
~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage

if TYPE_CHECKING:
    from .matrix import Matrix


@dataclass(frozen=True)
class ClusterLayout:
    """
    Data class for storing dendrogram-driven order and cluster span metadata for plotting.
    """

    leaf_order: np.ndarray
    ordered_labels: np.ndarray
    ordered_cluster_ids: np.ndarray
    cluster_spans: List[Tuple[int, int, int]]
    cluster_sizes: Dict[int, int]
    col_order: Optional[np.ndarray] = None
    col_linkage: Optional[np.ndarray] = None


def _resolve_fastcluster_linkage():
    """
    Resolves fastcluster.linkage when available.

    Returns:
        Optional[callable]: fastcluster linkage callable, or None when unavailable.
    """
    try:
        from fastcluster import linkage as fastcluster_linkage
    except ImportError:
        return None
    return fastcluster_linkage


def compute_col_linkage(
    matrix: Matrix,
    *,
    linkage_method: str = "ward",
    linkage_metric: str = "euclidean",
    optimal_ordering: bool = False,
) -> Optional[np.ndarray]:
    """
    Computes a linkage matrix for the columns of a matrix (visual grouping only).

    Args:
        matrix (Matrix): Matrix providing column values.

    Kwargs:
        linkage_method (str): Linkage method for hierarchical clustering. Defaults to "ward".
        linkage_metric (str): Distance metric for hierarchical clustering. Defaults to "euclidean".
        optimal_ordering (bool): Whether to optimize leaf ordering in the linkage output.
            Defaults to False.

    Returns:
        Optional[np.ndarray]: Column linkage matrix, or None when the matrix has fewer than two
            columns.
    """
    if int(matrix.df.shape[1]) <= 1:
        return None
    if not bool(optimal_ordering):
        fastcluster_linkage = _resolve_fastcluster_linkage()
        if fastcluster_linkage is not None:
            return fastcluster_linkage(
                matrix.values.T,
                method=linkage_method,
                metric=linkage_metric,
            )
    return linkage(
        matrix.values.T,
        method=linkage_method,
        metric=linkage_metric,
        optimal_ordering=bool(optimal_ordering),
    )


def col_order_from_linkage(col_linkage: Optional[np.ndarray], n_cols: int) -> np.ndarray:
    """
    Resolves the dendrogram leaf order of columns from a column linkage matrix.

    Args:
        col_linkage (Optional[np.ndarray]): Column linkage matrix, or None for fewer than two
            columns.
        n_cols (int): Number of matrix columns.

    Returns:
        np.ndarray: Column order indices in dendrogram order.
    """
    if col_linkage is None:
        return np.arange(n_cols, dtype=int)
    return leaves_list(col_linkage)


def compute_col_order(
    matrix: Matrix,
    *,
    linkage_method: str = "ward",
    linkage_metric: str = "euclidean",
    optimal_ordering: bool = False,
) -> np.ndarray:
    """
    Computes a dendrogram leaf order for matrix columns (visual grouping only).

    Args:
        matrix (Matrix): Matrix providing column values.

    Kwargs:
        linkage_method (str): Linkage method for hierarchical clustering. Defaults to "ward".
        linkage_metric (str): Distance metric for hierarchical clustering. Defaults to "euclidean".
        optimal_ordering (bool): Whether to optimize leaf ordering in the linkage output.
            Defaults to False.

    Returns:
        np.ndarray: Column order indices in dendrogram order.
    """
    col_linkage = compute_col_linkage(
        matrix,
        linkage_method=linkage_method,
        linkage_metric=linkage_metric,
        optimal_ordering=optimal_ordering,
    )
    return col_order_from_linkage(col_linkage, int(matrix.df.shape[1]))
