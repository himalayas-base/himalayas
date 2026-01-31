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
        """
        self.matrix = matrix
        self.annotations = annotations
        self.clusters = None
        self.layout = None
        self.results = None

    def cluster(self, **kwargs) -> Analysis:
        """
        Performs clustering on the analysis matrix.

        Returns:
            Analysis: The Analysis instance (for method chaining).
        """
        self.clusters = cluster(self.matrix, **kwargs)
        return self

    def enrich(self, **kwargs) -> Analysis:
        """
        Performs enrichment analysis on the clustered matrix.

        Returns:
            Analysis: The Analysis instance (for method chaining).
        """
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
        add_qvalues: bool = True,
        col_cluster: bool = False,
        **kwargs,
    ) -> Analysis:
        """
        Finalizes the analysis by computing a layout and producing a presentation-ready Results object.
        This step is required for plotting and downstream visualization, but may be skipped if only raw
        enrichment statistics are needed.

        Returns:
            Analysis: The Analysis instance (for method chaining).
        """
        if self.clusters is None or self.results is None:
            raise RuntimeError("cluster() and enrich() must be called before finalize()")

        col_order = None
        if col_cluster:
            col_order = compute_col_order(self.matrix, **kwargs)

        self.layout = self.clusters.layout(col_order=col_order)
        self.results = Results(
            self.results.df,
            method=self.results.method,
            matrix=self.matrix,
            clusters=self.clusters,
            layout=self.layout,
            parent=self.results.parent,
        )
        if add_qvalues:
            self.results = self.results.with_qvalues()

        return self
