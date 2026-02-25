"""
himalayas/core/analysis
~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Optional

from .annotations import Annotations
from .clustering import cut_linkage, compute_linkage
from .enrichment import run_cluster_hypergeom
from .layout import compute_col_order
from .matrix import Matrix
from .results import Results


class Analysis:
    """
    Class for orchestrating clustering, enrichment, and layout over a matrix and annotations.
    """

    def __init__(self, matrix: Matrix, annotations: Annotations) -> None:
        """
        Initializes the Analysis instance.

        Args:
            matrix (Matrix): Matrix to analyze.
            annotations (Annotations): Annotations aligned to the matrix.
        """
        self.matrix = matrix
        self.annotations = annotations
        self.clusters = None
        self.layout = None
        self.results = None
        self._cluster_linkage_method = "ward"
        self._cluster_linkage_metric = "euclidean"
        self._cluster_optimal_ordering = False
        self._col_order_cache = {}
        self._row_linkage_cache = {}

    def cluster(
        self,
        linkage_method: str = "ward",
        linkage_metric: str = "euclidean",
        linkage_threshold: float = 0.7,
        *,
        optimal_ordering: bool = False,
        min_cluster_size: int = 1,
    ) -> Analysis:
        """
        Performs clustering on the analysis matrix.

        Args:
            linkage_method (str): Linkage method for hierarchical clustering. Defaults to "ward".
            linkage_metric (str): Distance metric for hierarchical clustering. Defaults to "euclidean".
            linkage_threshold (float): Distance threshold for cutting the dendrogram. Defaults to 0.7.

        Kwargs:
            optimal_ordering (bool): Whether to optimize leaf ordering in the linkage output.
                Defaults to False.
            min_cluster_size (int): Enforces a minimum cluster size by merging smaller clusters
                upward along the dendrogram. Values <= 1 disable enforcement. Defaults to 1.

        Returns:
            Analysis: The Analysis instance (for method chaining).
        """
        self.results = None
        self.layout = None
        self._cluster_linkage_method = linkage_method
        self._cluster_linkage_metric = linkage_metric
        self._cluster_optimal_ordering = bool(optimal_ordering)
        row_cache_key = (
            self._cluster_linkage_method,
            self._cluster_linkage_metric,
            self._cluster_optimal_ordering,
        )
        linkage_matrix = self._row_linkage_cache.get(row_cache_key)
        if linkage_matrix is None:
            linkage_matrix = compute_linkage(
                self.matrix,
                linkage_method=self._cluster_linkage_method,
                linkage_metric=self._cluster_linkage_metric,
                optimal_ordering=self._cluster_optimal_ordering,
            )
            self._row_linkage_cache[row_cache_key] = linkage_matrix
        self.clusters = cut_linkage(
            linkage_matrix,
            self.matrix.labels,
            linkage_threshold=linkage_threshold,
            min_cluster_size=min_cluster_size,
        )
        return self

    def enrich(
        self,
        *,
        min_overlap: int = 1,
        background: Optional[Matrix] = None,
    ) -> Analysis:
        """
        Performs enrichment analysis on the clustered matrix.

        Kwargs:
            min_overlap (int): Minimum overlap (k) to report. Defaults to 1.
            background (Optional[Matrix]): Background matrix defining enrichment universe.
                Defaults to None.

        Returns:
            Analysis: The Analysis instance (for method chaining).

        Raises:
            RuntimeError: If cluster() has not been called.
        """
        # Validation
        if self.clusters is None:
            raise RuntimeError("cluster() must be called before enrich()")
        self.results = run_cluster_hypergeom(
            self.matrix,
            self.clusters,
            self.annotations,
            min_overlap=min_overlap,
            background=background,
        )
        return self

    def finalize(
        self,
        *,
        col_cluster: bool = False,
    ) -> Analysis:
        """
        Finalizes the analysis by computing a layout, attaching context, and adding effect sizes
        and q-values to produce a presentation-ready Results object. This step is required for
        plotting and downstream visualization, but may be skipped if only raw enrichment statistics
        are needed.

        Kwargs:
            col_cluster (bool): Whether to compute column order by clustering. Defaults to False.

        Returns:
            Analysis: The Analysis instance (for method chaining).

        Raises:
            RuntimeError: If cluster() or enrich() has not been called.
        """
        # Validation
        if self.clusters is None or self.results is None:
            raise RuntimeError("cluster() and enrich() must be called before finalize()")

        # Resolve column order, then construct finalized Results
        col_order = None
        if col_cluster:
            cache_key = (
                self._cluster_linkage_method,
                self._cluster_linkage_metric,
                self._cluster_optimal_ordering,
            )
            cached = self._col_order_cache.get(cache_key)
            if cached is None:
                cached = compute_col_order(
                    self.matrix,
                    linkage_method=self._cluster_linkage_method,
                    linkage_metric=self._cluster_linkage_metric,
                    optimal_ordering=self._cluster_optimal_ordering,
                )
                self._col_order_cache[cache_key] = cached
            col_order = cached
        self.layout = self.clusters.layout(col_order=col_order)
        self.results = Results(
            self.results.df,
            method=self.results.method,
            matrix=self.matrix,
            clusters=self.clusters,
            layout=self.layout,
            parent=self.results.parent,
        )
        self.results = self.results.with_effect_sizes().with_qvalues()

        return self
