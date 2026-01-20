"""
himalayas/core/enrichment
~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from .annotations import Annotations
from .clustering import Clusters
from .matrix import Matrix
from .results import Results


def _encode_label_indices(labels: np.ndarray) -> Dict[Any, int]:
    """
    Maps label to dense integer index for fast set operations.

    Args:
        labels (np.ndarray): Array of unique labels.

    Returns:
        Dict[Any, int]: Mapping from label to integer index.
    """
    # Labels are validated to be unique already
    return {lab: i for i, lab in enumerate(labels.tolist())}


def _count_intersection_sorted(a: np.ndarray, b: np.ndarray) -> int:
    """
    Counts intersection size between two sorted, unique int arrays (two-pointer, no allocations).

    Args:
        a (np.ndarray): First sorted array of integers.
        b (np.ndarray): Second sorted array of integers.

    Returns:
        int: Size of the intersection between a and b.
    """
    # Initialize pointers and count
    i = 0
    j = 0
    na = int(a.size)
    nb = int(b.size)
    c = 0

    # Two-pointer scan
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


def _validate_background(matrix: Matrix, background: Optional[Matrix]) -> Tuple[np.ndarray, int]:
    """
    Validates background matrix and determine universe labels and size.

    Args:
        matrix (Matrix): Analysis matrix.
        background (Optional[Matrix]): Background matrix defining enrichment universe.

    Returns:
        Tuple[np.ndarray, int]: Background labels and universe size N.

    Raises:
        ValueError: If background matrix does not contain all analysis matrix labels.
    """
    bg_labels = background.labels if background is not None else matrix.labels
    N = int(bg_labels.shape[0])
    if background is not None:
        if not set(matrix.labels).issubset(set(bg_labels)):
            raise ValueError(
                "background matrix must contain all labels present in the analysis matrix"
            )
    if N == 0:
        raise ValueError("Enrichment universe is empty (N=0)")
    return bg_labels, N


def _encode_terms(
    annotations: Annotations, label_to_idx: Dict[Any, int], *, min_overlap: int
) -> List[Tuple[str, np.ndarray, int]]:
    """
    Pre-encodes term label sets as sorted unique int arrays.

    Args:
        annotations (Annotations): Annotations aligned to the matrix.
        label_to_idx (Dict[Any, int]): Mapping from label to integer index.
        min_overlap (int): Minimum overlap (k) to report.

    Returns:
        List[Tuple[str, np.ndarray, int]]: List of tuples (term, idx_array, K).
    """
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
    return term_items


def _encode_clusters(
    clusters: Clusters, label_to_idx: Dict[Any, int], *, min_overlap: int
) -> Dict[int, Tuple[np.ndarray, int]]:
    """
    Pre-encodes clusters as sorted unique int arrays.

    Args:
        clusters (Clusters): Clustering results aligned to the matrix.
        label_to_idx (Dict[Any, int]): Mapping from label to integer index.
        min_overlap (int): Minimum overlap (k) to report.

    Returns:
        Dict[int, Tuple[np.ndarray, int]]: Mapping cluster_id → (cidx, n).

    Raises:
        RuntimeError: If a cluster has zero label indices after validation.
    """
    cluster_dict: Dict[int, Tuple[np.ndarray, int]] = {}
    cluster_ids = clusters.unique_clusters

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

        cluster_dict[cid_int] = (cidx, n)
    return cluster_dict


def run_cluster_hypergeom(
    matrix: Matrix,
    clusters: Clusters,
    annotations: Annotations,
    *,
    min_overlap: int = 1,
    background: Optional[Matrix] = None,
) -> Results:
    """
    Performs cluster x term enrichment using the hypergeometric test. If `background` is provided,
    it defines the enrichment universe (N). Otherwise, the universe defaults to the analysis matrix.

    Args:
        matrix (Matrix): Analysis matrix.
        clusters (Clusters): Clustering results aligned to the matrix.
        annotations (Annotations): Annotations aligned to the matrix.
        min_overlap (int, optional): Minimum overlap (k) to report. Defaults to 1.
        background (Optional[Matrix], optional): Background matrix defining enrichment universe.
            Defaults to None.

    Returns:
        Results: Enrichment results with the following columns: cluster, term, k, K, n, N, pval

    Raises:
        ValueError: If background matrix does not contain all analysis matrix labels.
    """
    # Validate background and get universe labels and size
    bg_labels, N = _validate_background(matrix, background)
    label_to_idx = _encode_label_indices(bg_labels)
    term_items = _encode_terms(annotations, label_to_idx, min_overlap=min_overlap)
    # Early exit if no terms pass filtering
    if not term_items:
        return Results(
            pd.DataFrame(columns=["cluster", "term", "k", "K", "n", "N", "pval"]),
            method="hypergeom",
            matrix=matrix,
            clusters=clusters,
        )

    # Encode clusters as index arrays
    cluster_dict = _encode_clusters(clusters, label_to_idx, min_overlap=min_overlap)
    rows: List[Dict[str, Any]] = []
    for cid_int, (cidx, n) in cluster_dict.items():
        # Cache hypergeom.sf for this cluster (N and n fixed)
        sf_cache: Dict[Tuple[int, int], float] = {}
        #  Test all terms
        for term, tidx, K in term_items:
            # Fast intersection size; skip if below min_overlap
            k = _count_intersection_sorted(cidx, tidx)
            if k < int(min_overlap):
                continue
            # Compute p-value with caching
            key = (K, k)
            pval = sf_cache.get(key)
            if pval is None:
                # P(X >= k) under Hypergeom(N population, K successes, n draws)
                pval = float(hypergeom.sf(k - 1, N, K, n))
                sf_cache[key] = pval
            # Record result row
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

    # Assemble results DataFrame
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
