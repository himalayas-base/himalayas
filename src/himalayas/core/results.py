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

        Args:
            df (pd.DataFrame): Results table.
            method (str): Analysis method identifier.

        Kwargs:
            matrix (Optional[Matrix]): Matrix associated with these results. Defaults to None.
            clusters (Optional[Clusters]): Clusters associated with these results. Defaults to None.
            layout (Optional[ClusterLayout]): Cached layout for plotting. Defaults to None.
            parent (Optional[Results]): Parent results for provenance. Defaults to None.
        """
        self.df = df
        self.method = method
        self.matrix = matrix
        self.clusters = clusters
        self.parent = parent
        self._layout = layout
        self.params: Dict[str, Any] = {}
        if clusters is not None:
            self.params["linkage_threshold"] = clusters.threshold

    def filter(self, expr, **kwargs) -> Results:
        """
        Returns a Results object filtered by a DataFrame query expression.

        Args:
            expr (str): DataFrame query expression.

        Kwargs:
            **kwargs: Additional keyword arguments for pd.DataFrame.query. Defaults to {}.

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
        Returns a Results object restricted to a single cluster.

        Args:
            cluster (int): Cluster id to subset.

        Returns:
            Results: New Results object restricted to the cluster.

        Raises:
            ValueError: If matrix or clusters are not attached.
            KeyError: If the cluster id is not found.
        """
        # Validation
        if self.matrix is None or self.clusters is None:
            raise ValueError("Results must have matrix and clusters to subset")
        cid = int(cluster)
        if cid not in self.clusters.cluster_to_labels:
            raise KeyError(f"Cluster {cid} not found")

        # Subset matrix (rows only)
        labels = sorted(self.clusters.cluster_to_labels[cid])
        df_sub = self.matrix.df.loc[labels]
        sub_matrix = Matrix(
            df_sub,
            axis=self.matrix.axis,
        )
        # Return new Results view
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
        Performs Benjamini–Hochberg FDR correction.

        Args:
            pvals (np.ndarray): Array of p-values.

        Returns:
            np.ndarray: FDR-adjusted q-values aligned to input order.

        Raises:
            ValueError: If p-values are non-finite or outside [0, 1].
        """
        pvals = np.asarray(pvals, dtype=float)
        if pvals.size == 0:
            return pvals

        # Validation
        if np.any((pvals < 0) | (pvals > 1) | ~np.isfinite(pvals)):
            raise ValueError("p-values must be finite and in [0, 1]")

        # Compute BH q-values and restore original order
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
        Returns a new Results with BH-FDR q-values added as `qval_col`. Does not mutate
        the original Results.

        Args:
            pval_col (str): Column name for p-values. Defaults to "pval".
            qval_col (str): Column name for q-values. Defaults to "qval".

        Returns:
            Results: New Results object with q-values added.

        Raises:
            KeyError: If the p-value column is missing.
        """
        # Validation
        if pval_col not in self.df.columns:
            raise KeyError(f"Missing p-value column: {pval_col!r}")

        # Compute q-values while preserving row alignment
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
        """
        Returns the authoritative clustering layout for downstream plotting.

        Kwargs:
            strict (bool): If True, requires contiguous clusters. Defaults to True.

        Returns:
            ClusterLayout: Attached cluster layout object.

        Raises:
            ValueError: If no ClusterLayout is attached.
        """
        if self._layout is None:
            raise ValueError("Results has no attached ClusterLayout")
        return self._layout

    def cluster_spans(self, *, strict: bool = True) -> List[Tuple[int, int, int]]:
        """
        Returns cluster spans in dendrogram order.

        Kwargs:
            strict (bool): If True, requires contiguous clusters. Defaults to True.

        Returns:
            List[Tuple[int, int, int]]: List of (cluster_id, start, end) spans.
        """
        return self.cluster_layout(strict=strict).cluster_spans
