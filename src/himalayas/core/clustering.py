"""
himalayas/core/clustering
~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage, fcluster

from .layout import ClusterLayout


# ------------------------------------------------------------
# Branch cutting post-process: enforce minimum cluster size
# ------------------------------------------------------------

def _build_tree_arrays(Z: np.ndarray, n_leaves: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (left_child, right_child, parent) arrays for a SciPy linkage tree.

    Parameters
    ----------
    Z : np.ndarray
        SciPy linkage matrix of shape (n_leaves-1, 4).
    n_leaves : int
        Number of original leaves.

    Returns
    -------
    (left, right, parent) : tuple of np.ndarray
        Arrays of length (n_leaves + Z.shape[0]) where internal node indices are
        `n_leaves + i` for merge row i.

    Notes
    -----
    This is an O(n) conversion from linkage format to an explicit binary tree.
    """
    Z = np.asarray(Z)
    m = int(Z.shape[0])
    n_nodes = int(n_leaves + m)

    left = np.full(n_nodes, -1, dtype=np.int32)
    right = np.full(n_nodes, -1, dtype=np.int32)
    parent = np.full(n_nodes, -1, dtype=np.int32)

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
    """Compute subtree sizes (# of leaves) for each node in the linkage tree."""
    n_nodes = int(left.size)
    sizes = np.zeros(n_nodes, dtype=np.int32)
    sizes[:n_leaves] = 1

    # Internal nodes are laid out in increasing index order.
    for node in range(n_leaves, n_nodes):
        a = int(left[node])
        b = int(right[node])
        # Defensive: linkage format guarantees these exist for internal nodes.
        sizes[node] = int(sizes[a] + sizes[b])

    return sizes


def _lca_pair(a: int, b: int, parent: np.ndarray) -> int:
    """Lowest common ancestor for two nodes using parent pointers."""
    seen = set()
    x = int(a)
    while x != -1:
        seen.add(x)
        x = int(parent[x])

    y = int(b)
    while y not in seen:
        y = int(parent[y])
        if y == -1:
            raise RuntimeError("LCA computation failed: reached root without finding common ancestor")

    return int(y)


def _lca_many(nodes: Sequence[int], parent: np.ndarray) -> int:
    """Lowest common ancestor for a set of leaf indices (iterated pairwise LCA)."""
    if not nodes:
        raise ValueError("LCA requested for empty node set")
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
    """Return sorted leaf indices under `node` (memoized).

    Uses an explicit stack (no recursion) to avoid recursion depth issues.
    """
    node = int(node)
    got = cache.get(node)
    if got is not None:
        return got

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

    arr = np.asarray(out, dtype=np.int32)
    if arr.size > 1:
        arr.sort()
    cache[node] = arr
    return arr


def _group_leaves_by_cluster(cluster_ids: np.ndarray) -> Dict[int, List[int]]:
    """Group leaf indices by cluster id (single pass, no np.where loops)."""
    groups: Dict[int, List[int]] = {}
    for i, cid in enumerate(cluster_ids.tolist()):
        groups.setdefault(int(cid), []).append(int(i))
    return groups


def _relabel_by_dendrogram_order(cluster_ids: np.ndarray, leaf_order: np.ndarray) -> np.ndarray:
    """Relabel cluster IDs to consecutive 1..K by first appearance in dendrogram order."""
    cluster_ids = np.asarray(cluster_ids)
    leaf_order = np.asarray(leaf_order, dtype=int)

    first_pos: Dict[int, int] = {}
    for pos, leaf in enumerate(leaf_order.tolist()):
        cid = int(cluster_ids[int(leaf)])
        if cid not in first_pos:
            first_pos[cid] = int(pos)

    ordered = [cid for cid, _ in sorted(first_pos.items(), key=lambda kv: kv[1])]
    mapping = {cid: i + 1 for i, cid in enumerate(ordered)}

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
    """Post-process a dendrogram cut to enforce a minimum cluster size.

    Strategy
    --------
    1) Perform a standard cut (already done by the caller via `fcluster`).
    2) For each cut cluster, find its corresponding subtree root (LCA of its leaves).
    3) If the cluster is smaller than `min_cluster_size`, climb to parents until the
       subtree contains at least `min_cluster_size` leaves.
    4) Convert the resulting (possibly overlapping) target subtrees into a valid
       non-overlapping partition by keeping only the highest selected nodes.

    The notion of "nearest neighbor" is unambiguous here: merging happens by moving
    upward along the linkage tree, which merges with the sibling at the lowest
    available merge height.

    Returns
    -------
    np.ndarray
        New integer cluster IDs (length = n_leaves), relabeled in dendrogram order.
    """
    cluster_ids = np.asarray(cluster_ids, dtype=np.int32)
    n = int(labels.shape[0])

    if min_cluster_size <= 1:
        return cluster_ids
    if min_cluster_size > n:
        raise ValueError(
            f"min_cluster_size={min_cluster_size} exceeds N={n}. "
            "Decrease min_cluster_size or cluster fewer items."
        )

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
    for node in keep_list:
        leaves = _node_leaves(node, left, right, n, leaf_cache)
        keep_first_pos[int(node)] = min(pos_map[int(x)] for x in leaves.tolist())

    for node in sorted(keep_list, key=lambda nd: keep_first_pos[int(nd)]):
        leaves = _node_leaves(node, left, right, n, leaf_cache)
        for li in leaves.tolist():
            assigned[int(li)] = int(node)

    if np.any(assigned < 0):
        raise RuntimeError("Minimum cluster size enforcement failed to assign all leaves")

    # Relabel to 1..K in dendrogram order for stable downstream presentation.
    return _relabel_by_dendrogram_order(assigned, leaf_order)


class Clusters:
    """Holds dendrogram and cluster assignments.

    This object is the single source of truth for:
      - leaf order
      - observation (label) → cluster mapping
      - cluster → labels mapping
      - cluster spans in dendrogram order (contiguity checked)

    The goal is to avoid recomputation in downstream analysis and plotting.
    """

    def __init__(
        self,
        linkage_matrix: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        *,
        min_cluster_size: Optional[int] = None,
    ):
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
        if min_cluster_size is not None:
            mcs = int(min_cluster_size)
            if mcs > 1:
                self.cluster_ids = _enforce_min_cluster_size(
                    self.linkage_matrix,
                    self.labels,
                    self.cluster_ids,
                    mcs,
                ).astype(int)

        # Lazy caches
        self._leaf_order: Optional[np.ndarray] = None
        self._label_to_cluster: Optional[Dict[Any, int]] = None
        self._cluster_to_labels: Optional[Dict[int, Set[Any]]] = None
        self._cluster_sizes: Optional[Dict[int, int]] = None
        self._unique_clusters: Optional[np.ndarray] = None
        self._layout: Optional[ClusterLayout] = None

    @property
    def leaf_order(self) -> np.ndarray:
        """Leaf order implied by the hierarchical clustering."""
        if self._leaf_order is None:
            self._leaf_order = leaves_list(self.linkage_matrix)
        return self._leaf_order

    @property
    def label_to_cluster(self) -> Dict[Any, int]:
        """Map matrix labels to integer cluster IDs."""
        if self._label_to_cluster is None:
            self._label_to_cluster = dict(zip(self.labels.tolist(), self.cluster_ids.tolist()))
        return self._label_to_cluster

    @property
    def unique_clusters(self) -> np.ndarray:
        """Unique cluster IDs (sorted)."""
        if self._unique_clusters is None:
            self._unique_clusters = np.unique(self.cluster_ids)
        return self._unique_clusters

    @property
    def cluster_to_labels(self) -> Dict[int, Set[Any]]:
        """Map cluster ID → set of labels (e.g., genes)."""
        if self._cluster_to_labels is None:
            out: Dict[int, Set[Any]] = {}
            for lab, cid in zip(self.labels, self.cluster_ids):
                out.setdefault(int(cid), set()).add(lab)
            self._cluster_to_labels = out
        return self._cluster_to_labels

    @property
    def cluster_sizes(self) -> Dict[int, int]:
        """Map cluster ID → number of labels in that cluster."""
        if self._cluster_sizes is None:
            self._cluster_sizes = {cid: len(gs) for cid, gs in self.cluster_to_labels.items()}
        return self._cluster_sizes

    def ordered_cluster_ids(self, order: Optional[np.ndarray] = None) -> np.ndarray:
        """Cluster IDs aligned to dendrogram order (or a provided order)."""
        if order is None:
            order = self.leaf_order
        return self.cluster_ids[np.asarray(order, dtype=int)]

    def ordered_labels(self, order: Optional[np.ndarray] = None) -> np.ndarray:
        """Labels aligned to dendrogram order (or a provided order)."""
        if order is None:
            order = self.leaf_order
        return self.labels[np.asarray(order, dtype=int)]

    def cluster_spans(
        self,
        order: Optional[np.ndarray] = None,
        *,
        strict: bool = True,
    ) -> List[Tuple[int, int, int]]:
        """Return contiguous spans for each cluster in dendrogram order.

        Returns
        -------
        list of (cluster_id, start, end)
            start/end are indices in the *ordered* (leaf) space.

        Notes
        -----
        If `strict=True`, raises if any cluster is non-contiguous in the requested order.
        Non-contiguity indicates a mismatch between the dendrogram order used for cutting
        and the cluster labels used for drawing boundaries.
        """
        cids = self.ordered_cluster_ids(order)
        spans: List[Tuple[int, int, int]] = []
        if cids.size == 0:
            return spans

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
        """Return a frozen, read-only layout object for downstream plotting."""
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
    matrix: "Matrix",
    linkage_method: str = "ward",
    linkage_metric: str = "euclidean",
    linkage_threshold: float = 0.7,
    *,
    min_cluster_size: Optional[int] = None,
) -> Clusters:
    """Perform hierarchical clustering and return a `Clusters` object with cached metadata.

    If `min_cluster_size` is provided, clusters smaller than this value are merged
    upward along the dendrogram until the size constraint is satisfied.
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

