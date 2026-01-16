"""
himalayas/core/clustering
~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage, fcluster

from .layout import ClusterLayout
from .matrix import Matrix


def _build_tree_arrays(Z: np.ndarray, n_leaves: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds explicit binary-tree arrays (left, right, parent) from a SciPy linkage matrix.
    Leaves are indexed 0..n_leaves-1. Each linkage row i creates an internal node with index
    n_leaves+i, whose children are Z[i,0] and Z[i,1].

    Args:
        Z (np.ndarray): Linkage matrix.
        n_leaves (int): Number of leaves in the tree.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Left child, right child, and parent arrays.
    """
    # Normalize linkage matrix and derive node counts
    Z = np.asarray(Z)
    m = int(Z.shape[0])
    n_nodes = int(n_leaves + m)

    # Allocate explicit binary-tree representation
    left = np.full(n_nodes, -1, dtype=np.int32)
    right = np.full(n_nodes, -1, dtype=np.int32)
    parent = np.full(n_nodes, -1, dtype=np.int32)

    # Populate child and parent pointers for each merge
    for i in range(m):
        a = int(Z[i, 0])
        b = int(Z[i, 1])
        node = int(n_leaves + i)
        left[node] = a
        right[node] = b
        parent[a] = node
        parent[b] = node

    return left, right, parent


def _compute_subtree_sizes(left: np.ndarray, right: np.ndarray, n_leaves: int) -> np.ndarray:
    """
    Computes subtree sizes (number of leaves) for every node in the linkage tree.

    Args:
        left (np.ndarray): Left child indices for each node.
        right (np.ndarray): Right child indices for each node.
        n_leaves (int): Number of leaves in the tree.

    Returns:
        np.ndarray: Array of subtree sizes for each node.
    """
    # Initialize sizes array
    n_nodes = int(left.size)
    sizes = np.zeros(n_nodes, dtype=np.int32)
    sizes[:n_leaves] = 1

    # Internal nodes are laid out in increasing index order
    for node in range(n_leaves, n_nodes):
        a = int(left[node])
        b = int(right[node])
        # Defensive: linkage format guarantees these exist for internal nodes
        sizes[node] = int(sizes[a] + sizes[b])

    return sizes


def _lca_pair(a: int, b: int, parent: np.ndarray) -> int:
    """
    Returns the lowest common ancestor of two nodes using parent pointers.

    Args:
        a (int): Index of the first node.
        b (int): Index of the second node.
        parent (np.ndarray): Array of parent indices for each node.

    Returns:
        int: Index of the lowest common ancestor node.

    Raises:
        RuntimeError: If no common ancestor is found (should not happen in a valid tree).
    """
    # Collect ancestors of `a`, then walk `b`'s ancestors until we find a match
    seen = set()
    x = int(a)
    while x != -1:
        seen.add(x)
        x = int(parent[x])

    # Now walk up from `b`, looking for the first ancestor in `seen`
    y = int(b)
    while y not in seen:
        y = int(parent[y])
        if y == -1:
            raise RuntimeError(
                "LCA computation failed: reached root without finding common ancestor"
            )

    return int(y)


def _lca_many(nodes: Sequence[int], parent: np.ndarray) -> int:
    """
    Returns the lowest common ancestor of multiple nodes (pairwise reduction).

    Args:
        nodes (Sequence[int]): Indices of the nodes.
        parent (np.ndarray): Array of parent indices for each node.

    Returns:
        int: Index of the lowest common ancestor node.

    Raises:
        ValueError: If the input node list is empty.
    """
    if not nodes:
        raise ValueError("LCA requested for empty node set")
    # Iteratively compute LCA pairwise
    cur = int(nodes[0])
    for x in nodes[1:]:
        cur = _lca_pair(cur, int(x), parent)
    return int(cur)


def _node_leaves(
    node: int,
    left: np.ndarray,
    right: np.ndarray,
    n_leaves: int,
    cache: Dict[int, np.ndarray],
) -> np.ndarray:
    """
    Returns sorted leaf indices under a node (iterative DFS, memoized).

    Args:
        node (int): Index of the node.
        left (np.ndarray): Left child indices for each node.
        right (np.ndarray): Right child indices for each node.
        n_leaves (int): Number of leaves in the tree.
        cache (Dict[int, np.ndarray]): Cache mapping node indices to leaf index arrays.

    Returns:
        np.ndarray: Sorted array of leaf indices under the specified node.
    """
    # Check cache first
    node = int(node)
    got = cache.get(node)
    if got is not None:
        return got

    # Iterative DFS to collect leaves
    out: List[int] = []
    stack: List[int] = [node]
    while stack:
        u = int(stack.pop())
        if u < n_leaves:
            out.append(u)
            continue
        a = int(left[u])
        b = int(right[u])
        # Push both children
        stack.append(a)
        stack.append(b)

    # Sort leaves for deterministic output
    arr = np.asarray(out, dtype=np.int32)
    if arr.size > 1:
        arr.sort()
    cache[node] = arr

    return arr


def _group_leaves_by_cluster(cluster_ids: np.ndarray) -> Dict[int, List[int]]:
    """
    Groups leaf indices by cluster id in a single pass.

    Args:
        cluster_ids (np.ndarray): Array of cluster IDs for each leaf.

    Returns:
        Dict[int, List[int]]: Mapping from cluster ID to list of leaf indices.
    """
    groups: Dict[int, List[int]] = {}
    for i, cid in enumerate(cluster_ids.tolist()):
        groups.setdefault(int(cid), []).append(int(i))

    return groups


def _relabel_by_dendrogram_order(cluster_ids: np.ndarray, leaf_order: np.ndarray) -> np.ndarray:
    """
    Relabels cluster IDs to 1..K by first appearance in dendrogram order.

    Args:
        cluster_ids (np.ndarray): Original cluster IDs for each leaf.
        leaf_order (np.ndarray): Leaf order from the dendrogram.

    Returns:
        np.ndarray: Relabeled cluster IDs aligned to original leaf order.
    """
    cluster_ids = np.asarray(cluster_ids)
    leaf_order = np.asarray(leaf_order, dtype=int)
    # Map cluster ID -> first position in dendrogram order
    first_pos: Dict[int, int] = {}
    for pos, leaf in enumerate(leaf_order.tolist()):
        cid = int(cluster_ids[int(leaf)])
        if cid not in first_pos:
            first_pos[cid] = int(pos)

    # Order cluster IDs by first appearance
    ordered = [cid for cid, _ in sorted(first_pos.items(), key=lambda kv: kv[1])]
    mapping = {cid: i + 1 for i, cid in enumerate(ordered)}
    # Apply mapping to original cluster IDs
    out = np.empty_like(cluster_ids, dtype=np.int32)
    for i, cid in enumerate(cluster_ids.tolist()):
        out[i] = int(mapping[int(cid)])

    return out


def _enforce_min_cluster_size(
    Z: np.ndarray,
    labels: np.ndarray,
    cluster_ids: np.ndarray,
    min_cluster_size: int,
) -> np.ndarray:
    """Post-processes a dendrogram cut to enforce a minimum cluster size.
    Strategy:
    1. Cut the dendrogram using fcluster (already done).
    2. For each cluster, find the subtree root spanning its leaves (LCA).
    3. If too small, climb upward until the subtree meets min_cluster_size.
    4. Resolve overlaps by keeping only the highest selected subtrees.

    Args:
        Z (np.ndarray): Linkage matrix.
        labels (np.ndarray): Labels aligned to leaves.
        cluster_ids (np.ndarray): Initial cluster IDs from fcluster.
        min_cluster_size (int): Minimum desired cluster size.

    Returns:
        np.ndarray: Adjusted cluster IDs satisfying minimum size.

    Raises:
        ValueError: If min_cluster_size exceeds the number of leaves.
    """
    cluster_ids = np.asarray(cluster_ids, dtype=np.int32)
    # Number of leaves
    n = int(labels.shape[0])
    # Return early if no enforcement needed
    if min_cluster_size <= 1:
        return cluster_ids
    if min_cluster_size > n:
        raise ValueError(
            f"min_cluster_size={min_cluster_size} exceeds N={n}. "
            "Decrease min_cluster_size or cluster fewer items."
        )

    # Build tree structure
    left, right, parent = _build_tree_arrays(Z, n)
    sizes = _compute_subtree_sizes(left, right, n)
    # Deterministic ordering uses the dendrogram leaf order.
    leaf_order = leaves_list(Z)
    # Group leaves by initial cluster IDs.
    groups = _group_leaves_by_cluster(cluster_ids)

    # Map each initial cluster -> target subtree node.
    target_for_cluster: Dict[int, int] = {}
    for cid, leaves in groups.items():
        lca = _lca_many(leaves, parent)
        node = int(lca)
        # Climb until minimum size satisfied.
        while sizes[node] < int(min_cluster_size):
            p = int(parent[node])
            if p == -1:
                break
            node = p

        target_for_cluster[int(cid)] = int(node)

    # Collect unique target nodes.
    target_nodes: Set[int] = set(target_for_cluster.values())
    # Prune overlaps: if a node has an ancestor also selected, drop the node.
    keep: Set[int] = set()
    for node in target_nodes:
        cur = int(node)
        covered = False
        p = int(parent[cur])
        # Walk up to root
        while p != -1:
            if p in target_nodes:
                covered = True
                break
            p = int(parent[p])

        if not covered:
            keep.add(cur)

    # Assign leaves by kept nodes.
    assigned = np.full(n, -1, dtype=np.int32)
    leaf_cache: Dict[int, np.ndarray] = {}

    # Deterministic: assign kept nodes by first appearance in dendrogram order.
    keep_list = list(keep)
    keep_first_pos: Dict[int, int] = {}
    pos_map = {int(leaf): int(pos) for pos, leaf in enumerate(leaf_order.tolist())}
    # Compute first positions
    for node in keep_list:
        leaves = _node_leaves(node, left, right, n, leaf_cache)
        keep_first_pos[int(node)] = min(pos_map[int(x)] for x in leaves.tolist())
    # Assign leaves
    for node in sorted(keep_list, key=lambda nd: keep_first_pos[int(nd)]):
        leaves = _node_leaves(node, left, right, n, leaf_cache)
        for li in leaves.tolist():
            assigned[int(li)] = int(node)

    if np.any(assigned < 0):
        raise RuntimeError("Minimum cluster size enforcement failed to assign all leaves")

    # Relabel to 1..K in dendrogram order for stable downstream presentation.
    return _relabel_by_dendrogram_order(assigned, leaf_order)


class Clusters:
    """
    Contains dendrogram structure and cluster assignments. Owns all cluster-level
    metadata needed for downstream analysis and plotting, avoiding recomputation.
    """

    def __init__(
        self,
        linkage_matrix: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        *,
        min_cluster_size: int = 1,
    ):
        """
        Constructs Clusters from a linkage matrix and label set.

        Args:
            linkage_matrix (np.ndarray): Linkage matrix from hierarchical clustering.
            labels (np.ndarray): Labels aligned to the rows/columns of the matrix.
            threshold (float): Distance threshold for cutting the dendrogram.
            min_cluster_size (int): Enforces a minimum cluster size by merging smaller clusters upward
                along the dendrogram. Values ≤1 disable enforcement.

        Raises:
            ValueError: If labels and cluster IDs length mismatch.
        """
        self.linkage_matrix = linkage_matrix
        self.labels = np.asarray(labels, dtype=object)
        self.threshold = float(threshold)
        self.cluster_ids = fcluster(
            linkage_matrix,
            threshold,
            criterion="distance",
        ).astype(int)

        if self.labels.shape[0] != self.cluster_ids.shape[0]:
            raise ValueError("labels and cluster_ids length mismatch")

        # Optional post-process: enforce a minimum cluster size by merging upward
        # along the dendrogram (nearest neighbor defined by lowest merge height).
        if min_cluster_size > 1:
            self.cluster_ids = _enforce_min_cluster_size(
                self.linkage_matrix,
                self.labels,
                self.cluster_ids,
                min_cluster_size,
            ).astype(int)

        # Lazy caches for computed properties
        self._leaf_order: Optional[np.ndarray] = None
        self._label_to_cluster: Optional[Dict[Any, int]] = None
        self._cluster_to_labels: Optional[Dict[int, Set[Any]]] = None
        self._cluster_sizes: Optional[Dict[int, int]] = None
        self._unique_clusters: Optional[np.ndarray] = None
        self._layout: Optional[ClusterLayout] = None

    @property
    def leaf_order(self) -> np.ndarray:
        """
        Returns leaf order implied by the hierarchical clustering.

        Returns:
            np.ndarray: Array of leaf indices in dendrogram order.
        """
        if self._leaf_order is None:
            self._leaf_order = leaves_list(self.linkage_matrix)
        return self._leaf_order

    @property
    def label_to_cluster(self) -> Dict[Any, int]:
        """
        Maps matrix labels to integer cluster IDs.

        Returns:
            Dict[Any, int]: Mapping from label to cluster ID.
        """
        if self._label_to_cluster is None:
            self._label_to_cluster = dict(zip(self.labels.tolist(), self.cluster_ids.tolist()))
        return self._label_to_cluster

    @property
    def unique_clusters(self) -> np.ndarray:
        """
        Returns unique cluster IDs (sorted).

        Returns:
            np.ndarray: Array of unique cluster IDs.
        """
        if self._unique_clusters is None:
            self._unique_clusters = np.unique(self.cluster_ids)
        return self._unique_clusters

    @property
    def cluster_to_labels(self) -> Dict[int, Set[Any]]:
        """
        Maps cluster ID to set of labels (e.g., genes).

        Returns:
            Dict[int, Set[Any]]: Mapping from cluster ID to set of labels.
        """
        if self._cluster_to_labels is None:
            out: Dict[int, Set[Any]] = {}
            for lab, cid in zip(self.labels, self.cluster_ids):
                out.setdefault(int(cid), set()).add(lab)
            self._cluster_to_labels = out

        return self._cluster_to_labels

    @property
    def cluster_sizes(self) -> Dict[int, int]:
        """
        Maps cluster ID to number of labels in that cluster.

        Returns:
            Dict[int, int]: Mapping from cluster ID to cluster size.
        """
        if self._cluster_sizes is None:
            self._cluster_sizes = {cid: len(gs) for cid, gs in self.cluster_to_labels.items()}
        return self._cluster_sizes

    def ordered_cluster_ids(self, order: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns cluster IDs aligned to dendrogram order (or a provided order).

        Args:
            order (Optional[np.ndarray]): Optional order of leaf indices. If None, uses dendrogram

        Returns:
            np.ndarray: Array of cluster IDs in the specified order.
        """
        if order is None:
            order = self.leaf_order
        return self.cluster_ids[np.asarray(order, dtype=int)]

    def ordered_labels(self, order: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Labels aligned to dendrogram order (or a provided order).

        Args:
            order (Optional[np.ndarray]): Optional order of leaf indices. If None, uses dendrogram

        Returns:
            np.ndarray: Array of labels in the specified order.
        """
        if order is None:
            order = self.leaf_order
        return self.labels[np.asarray(order, dtype=int)]

    def cluster_spans(
        self,
        order: Optional[np.ndarray] = None,
        *,
        strict: bool = True,
    ) -> List[Tuple[int, int, int]]:
        """Returns contiguous spans for each cluster in dendrogram order. If `strict` is True,
        raises an error if any cluster is non-contiguous in the requested order.

        Args:
            order (Optional[np.ndarray]): Optional order of leaf indices. If None, uses dendrogram
                order.
            strict (bool): If True, raises an error if any cluster is non-contiguous in the order.

        Returns:
            List[Tuple[int, int, int]]: List of (cluster_id, start_index, end_index) spans in the order.

        Raises:
            ValueError: If `strict` is True and any cluster is non-contiguous in the requested order.
        """
        # Get cluster IDs in the requested order
        cids = self.ordered_cluster_ids(order)
        spans: List[Tuple[int, int, int]] = []
        if cids.size == 0:
            return spans

        # Identify contiguous spans
        start = 0
        cur = int(cids[0])
        for i in range(1, cids.size):
            nxt = int(cids[i])
            if nxt != cur:
                spans.append((cur, start, i - 1))
                start = i
                cur = nxt
        spans.append((cur, start, int(cids.size - 1)))

        if strict:
            # Verify each cluster appears in a single span
            seen = {}
            for cid, s, e in spans:
                if cid in seen:
                    raise ValueError(
                        f"Cluster {cid} is non-contiguous in the requested order: "
                        f"spans {seen[cid]} and {(s, e)}"
                    )
                seen[cid] = (s, e)

        return spans

    def layout(
        self,
        *,
        strict: bool = True,
        col_order: Optional[np.ndarray] = None,
    ) -> ClusterLayout:
        """
        Returns a frozen, read-only layout object for downstream plotting.

        Args:
            strict (bool): If True, raises an error if any cluster is non-contiguous in the order.
            col_order (Optional[np.ndarray]): Optional order of leaf indices for columns. If None,
                uses dendrogram order.

        Returns:
            ClusterLayout: Cached cluster layout object.
        """
        if self._layout is None:
            order = self.leaf_order
            ordered_labels = self.ordered_labels(order)
            ordered_cids = self.ordered_cluster_ids(order)
            spans = self.cluster_spans(order, strict=strict)
            self._layout = ClusterLayout(
                leaf_order=order,
                ordered_labels=ordered_labels,
                ordered_cluster_ids=ordered_cids,
                cluster_spans=spans,
                cluster_sizes=dict(self.cluster_sizes),
                col_order=col_order,
            )
        return self._layout


def cluster(
    matrix: Matrix,
    linkage_method: str = "ward",
    linkage_metric: str = "euclidean",
    linkage_threshold: float = 0.7,
    *,
    min_cluster_size: int = 1,
) -> Clusters:
    """
    Performs hierarchical clustering and returns a `Clusters` object with cached metadata.
    If `min_cluster_size` is set to a value greater than 1, clusters smaller than this value are merged
    upward along the dendrogram until the size constraint is satisfied. Values ≤1 disable enforcement.

    Args:
        matrix (Matrix): Matrix to cluster.
        linkage_method (str): Linkage method for hierarchical clustering.
        linkage_metric (str): Distance metric for hierarchical clustering.
        linkage_threshold (float): Distance threshold for cutting the dendrogram.
        min_cluster_size (int): Enforces a minimum cluster size by merging smaller clusters
            upward along the dendrogram. Values ≤1 disable enforcement.

    Returns:
        Clusters: Clusters object containing dendrogram and cluster assignments.
    """
    Z = linkage(
        matrix.values,
        method=linkage_method,
        metric=linkage_metric,
        optimal_ordering=True,
    )
    return Clusters(
        Z,
        matrix.labels,
        linkage_threshold,
        min_cluster_size=min_cluster_size,
    )
