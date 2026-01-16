"""Cluster-term enrichment routines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from .annotations import Annotations
from .clustering import Clusters, cluster
from .layout import compute_col_order
from .matrix import Matrix
from .results import Results


# ------------------------------------------------------------
# Helper functions for hypergeometric enrichment
# ------------------------------------------------------------


def _encode_label_indices(labels: np.ndarray) -> Dict[Any, int]:
    """Map label -> dense integer index for fast set operations."""
    # labels are already validated unique in Matrix
    return {lab: i for i, lab in enumerate(labels.tolist())}


def _count_intersection_sorted(a: np.ndarray, b: np.ndarray) -> int:
    """Count intersection size between two sorted, unique int arrays (two-pointer, no allocations)."""
    i = 0
    j = 0
    na = int(a.size)
    nb = int(b.size)
    c = 0
    while i < na and j < nb:
        av = int(a[i])
        bv = int(b[j])
        if av == bv:
            c += 1
            i += 1
            j += 1
        elif av < bv:
            i += 1
        else:
            j += 1
    return c


# ------------------------------------------------------------
# First-pass cluster × term enrichment using hypergeometric test
# ------------------------------------------------------------


def run_cluster_hypergeom(
    matrix: Matrix,
    clusters: Clusters,
    annotations: Annotations,
    *,
    min_overlap: int = 1,
    background: Optional[Matrix] = None,
) -> Results:
    """
    First-pass cluster × term enrichment using the hypergeometric test.

    If `background` is provided, it defines the enrichment universe (N). Otherwise, the universe defaults to the analysis matrix.

    Returns a Results whose `.df` contains:
      cluster, term, k, K, n, N, pval

    Performance notes
    -----------------
    This implementation avoids Python set intersections in the inner loop.
    It encodes labels into dense integer IDs, then counts intersections via
    a two-pointer scan over sorted integer arrays (no temporary allocations).

    It also caches hypergeom.sf evaluations per cluster for repeated (K, k)
    pairs (N and n are fixed per cluster).
    """
    bg_labels = background.labels if background is not None else matrix.labels

    # Enrichment universe
    N = int(bg_labels.shape[0])

    if background is not None:
        if not set(matrix.labels).issubset(set(bg_labels)):
            raise ValueError(
                "background matrix must contain all labels present in the analysis matrix"
            )

    if N == 0:
        raise ValueError("Enrichment universe is empty (N=0)")

    label_to_idx = _encode_label_indices(bg_labels)

    # --------------------------------------------------------
    # Pre-encode term label sets -> sorted unique int arrays
    # --------------------------------------------------------
    term_items: List[Tuple[str, np.ndarray, int]] = []  # (term, idx_array, K)
    for term, term_labels in annotations.term_to_labels.items():
        # term_labels already overlaps matrix labels by construction
        idx = np.fromiter((label_to_idx[l] for l in term_labels), dtype=np.int32)
        idx.sort()
        # de-dup defensively (should already be unique)
        if idx.size > 1:
            idx = np.unique(idx)
        K = int(idx.size)
        if K < int(min_overlap):
            continue
        term_items.append((term, idx, K))

    if not term_items:
        return Results(
            pd.DataFrame(columns=["cluster", "term", "k", "K", "n", "N", "pval"]),
            method="hypergeom",
            matrix=matrix,
            clusters=clusters,
        )

    # --------------------------------------------------------
    # Pre-encode clusters -> sorted unique int arrays
    # --------------------------------------------------------
    cluster_ids = clusters.unique_clusters

    rows: List[Dict[str, Any]] = []

    for cid in cluster_ids:
        cid_int = int(cid)
        cluster_labels = clusters.cluster_to_labels[cid_int]
        n = int(clusters.cluster_sizes[cid_int])
        if n <= 0 or n < int(min_overlap):
            continue

        cidx = np.fromiter((label_to_idx[l] for l in cluster_labels), dtype=np.int32)
        cidx.sort()
        if cidx.size > 1:
            cidx = np.unique(cidx)
        if cidx.size == 0:
            raise RuntimeError("Cluster with zero label indices encountered after validation")

        # Cache hypergeom.sf for this cluster (N and n fixed)
        sf_cache: Dict[Tuple[int, int], float] = {}

        for term, tidx, K in term_items:
            # Fast intersection size
            k = _count_intersection_sorted(cidx, tidx)
            if k < int(min_overlap):
                continue

            key = (K, k)
            pval = sf_cache.get(key)
            if pval is None:
                # P(X >= k) under Hypergeom(N population, K successes, n draws)
                pval = float(hypergeom.sf(k - 1, N, K, n))
                sf_cache[key] = pval

            rows.append(
                {
                    "cluster": cid_int,
                    "term": term,
                    "k": int(k),
                    "K": int(K),
                    "n": int(n),
                    "N": int(N),
                    "pval": pval,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["pval", "cluster", "term"], kind="mergesort").reset_index(drop=True)

        # Reduce memory footprint (safe downcasts for typical biology sizes)
        # Use int32 universally if you anticipate >32767 labels/terms.
        df = df.astype(
            {
                "cluster": "int32",
                "k": "int32",
                "K": "int32",
                "n": "int32",
                "N": "int32",
                "pval": "float64",
            }
        )

    return Results(
        df,
        method="hypergeom",
        matrix=matrix,
        clusters=clusters,
    )


def run_first_pass(
    matrix: Matrix,
    annotations: Annotations,
    *,
    background: Optional[Matrix] = None,
    linkage_method: str = "ward",
    linkage_metric: str = "euclidean",
    linkage_threshold: float = 0.7,
    min_cluster_size: Optional[int] = None,
    min_overlap: int = 1,
    add_qvalues: bool = True,
    col_cluster: bool = False,
    col_linkage_method: str = "ward",
    col_linkage_metric: str = "euclidean",
) -> Results:
    """Bone-simple first-pass: cluster to hypergeom enrichment to (optional) q-values.

    Use `background` to define the enrichment universe when running on a subset / zoomed matrix.

    This keeps the end-user workflow minimal:
        results = run_first_pass(matrix, annotations, ...)

    Returns
    -------
    Results
        Enrichment results with `matrix` and `clusters` attached.
    """
    clusters = cluster(
        matrix,
        linkage_method=linkage_method,
        linkage_metric=linkage_metric,
        linkage_threshold=linkage_threshold,
        min_cluster_size=min_cluster_size,
    )
    res = run_cluster_hypergeom(
        matrix,
        clusters,
        annotations,
        min_overlap=min_overlap,
        background=background,
    )

    # Optionally compute column order for layout (visual only)
    col_order = None
    if col_cluster:
        col_order = compute_col_order(
            matrix,
            linkage_method=col_linkage_method,
            linkage_metric=col_linkage_metric,
        )

    # Attach authoritative cluster layout for plotting (single source of truth)
    layout = clusters.layout(strict=True, col_order=col_order)
    res = Results(
        res.df,
        method=res.method,
        matrix=matrix,
        clusters=clusters,
        layout=layout,
        parent=res.parent,
    )
    return res.with_qvalues() if add_qvalues else res
