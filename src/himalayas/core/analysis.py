"""
himalayas/core/analysis
~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from .annotations import Annotations
from .clustering import cluster
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

    def cluster(self, **kwargs) -> Analysis:
        """
        Performs clustering on the analysis matrix.

        Kwargs:
            **kwargs: Keyword arguments forwarded to clustering. Defaults to {}.

        Returns:
            Analysis: The Analysis instance (for method chaining).
        """
        self._cluster_linkage_method = kwargs.get("linkage_method", "ward")
        self._cluster_linkage_metric = kwargs.get("linkage_metric", "euclidean")
        self.clusters = cluster(self.matrix, **kwargs)
        return self

    def enrich(self, **kwargs) -> Analysis:
        """
        Performs enrichment analysis on the clustered matrix.

        Kwargs:
            **kwargs: Keyword arguments forwarded to enrichment. Defaults to {}.

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
            **kwargs,
        )
        return self

    def finalize(
        self,
        *,
        col_cluster: bool = False,
    ) -> Analysis:
        """
        Finalizes the analysis by computing a layout, attaching context, and adding q-values to produce
        a presentation-ready Results object. This step is required for plotting and downstream
        visualization, but may be skipped if only raw enrichment statistics are needed.

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
            col_order = compute_col_order(
                self.matrix,
                linkage_method=self._cluster_linkage_method,
                linkage_metric=self._cluster_linkage_metric,
            )
        self.layout = self.clusters.layout(col_order=col_order)
        self.results = Results(
            self.results.df,
            method=self.results.method,
            matrix=self.matrix,
            clusters=self.clusters,
            layout=self.layout,
            parent=self.results.parent,
        )
        self.results = self.results.with_qvalues()

        return self
