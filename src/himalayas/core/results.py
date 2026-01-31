"""
himalayas/core/results
~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .clustering import Clusters
from .layout import ClusterLayout
from .matrix import Matrix


class Results:
    """
    Class for holding analysis results and attached context for plotting or subsetting.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        method: str,
        *,
        matrix: Optional[Matrix] = None,
        clusters: Optional[Clusters] = None,
        layout: Optional[ClusterLayout] = None,
        parent: Optional[Results] = None,
    ) -> None:
        """
        Initializes the Results instance.
        """
        self.df = df
        self.method = method
        self.matrix = matrix
        self.clusters = clusters
        self.parent = parent
        self._layout = layout
        # Analysis parameters / provenance
        self.params: Dict[str, Any] = {}
        if clusters is not None:
            self.params["linkage_threshold"] = clusters.threshold

    def filter(self, expr, **kwargs) -> Results:
        """
        Return a Results object filtered by a DataFrame query expression.

        Args:
            expr (str): DataFrame query expression.
            **kwargs: Additional keyword arguments for pd.DataFrame.query.

        Returns:
            Results: New Results object with filtered DataFrame.
        """
        filtered_df = self.df.query(expr, **kwargs)
        return Results(
            filtered_df,
            method=self.method,
            matrix=self.matrix,
            clusters=self.clusters,
            layout=self._layout,
            parent=self,
        )

    def subset(
        self,
        cluster: int,
    ) -> Results:
        """
        Return a Results object restricted to a single cluster.

        The returned Results contains:
          - a subset Matrix (rows restricted to the cluster)
          - no clusters attached (user must explicitly re-cluster)
          - parent pointer preserved for provenance
        """
        if self.matrix is None or self.clusters is None:
            raise ValueError("Results must have matrix and clusters to subset")

        cid = int(cluster)

        if cid not in self.clusters.cluster_to_labels:
            raise KeyError(f"Cluster {cid} not found")

        # --------------------------------------------
        # Subset matrix (rows only)
        # --------------------------------------------
        labels = sorted(self.clusters.cluster_to_labels[cid])
        df_sub = self.matrix.df.loc[labels]

        sub_matrix = Matrix(
            df_sub,
            matrix_semantics=self.matrix.matrix_semantics,
            axis=self.matrix.axis,
        )

        # --------------------------------------------
        # Return new Results view (no clustering yet)
        # --------------------------------------------
        return Results(
            df=pd.DataFrame(),
            method="subset",
            matrix=sub_matrix,
            clusters=None,
            layout=None,
            parent=self,
        )

    @staticmethod
    def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
        """
        Benjamini–Hochberg FDR correction.
        Returns q-values aligned to the input order.
        """
        pvals = np.asarray(pvals, dtype=float)

        if pvals.size == 0:
            return pvals

        if np.any((pvals < 0) | (pvals > 1) | ~np.isfinite(pvals)):
            raise ValueError("p-values must be finite and in [0, 1]")

        m = pvals.size
        order = np.argsort(pvals)
        ranked = pvals[order]

        q = ranked * (m / (np.arange(1, m + 1)))
        q = np.minimum.accumulate(q[::-1])[::-1]
        q = np.clip(q, 0.0, 1.0)

        out = np.empty_like(q)
        out[order] = q
        return out

    def with_qvalues(self, pval_col: str = "pval", qval_col: str = "qval") -> Results:
        """
        Return a new Results with BH-FDR q-values added as `qval_col`.
        Does not mutate the original Results.
        """
        if pval_col not in self.df.columns:
            raise KeyError(f"Missing p-value column: {pval_col!r}")

        df2 = self.df.copy()
        pvals = df2[pval_col].to_numpy(dtype=float)
        # Preserve row alignment: NaN p-values yield NaN q-values
        qvals = np.full_like(pvals, np.nan, dtype=float)
        mask = np.isfinite(pvals)
        qvals[mask] = self._bh_fdr(pvals[mask])
        df2[qval_col] = qvals
        return Results(
            df2,
            method=self.method,
            matrix=self.matrix,
            clusters=self.clusters,
            layout=self._layout,
            parent=self,
        )

    def cluster_layout(self, *, strict: bool = True) -> ClusterLayout:
        """Return the authoritative clustering layout for downstream plotting."""
        if self._layout is None:
            raise ValueError("Results has no attached ClusterLayout")
        return self._layout

    def cluster_spans(self, *, strict: bool = True) -> List[Tuple[int, int, int]]:
        return self.cluster_layout(strict=strict).cluster_spans
