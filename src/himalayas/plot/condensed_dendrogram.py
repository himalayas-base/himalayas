"""
himalayas/plot/condensed_dendrogram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Hashable
from os import PathLike
from typing import (
    Any,
    Collection,
    Dict,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    TypedDict,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap, Normalize
from scipy.cluster.hierarchy import dendrogram, leaves_list

from .renderers._label_format import (
    apply_label_text_policy,
    collect_label_stats,
    compose_label_text,
)
from .renderers.cluster_labels import _parse_label_overrides

if TYPE_CHECKING:
    from ..core.clustering import Clusters
    from ..core.results import Results


class DendrogramData(TypedDict):
    """
    Typed dictionary for condensed dendrogram data.
    """

    icoord: Sequence[Sequence[float]]
    dcoord: Sequence[Sequence[float]]
    leaves: Sequence[int]


class ClusterLabelInfo(NamedTuple):
    """
    Canonical per-cluster label payload used by condensed plotting.
    """

    label: str
    pval: float
    qval: float
    n_members: Optional[float]
    score: float


@dataclass(frozen=True)
class CondensedDendrogramSpec:
    """
    Immutable render specification for condensed dendrogram plots.
    """

    results: "Results"
    rank_by: str = "p"
    label_mode: str = "top_term"
    figsize: Tuple[float, float] = (10, 10)
    sigbar_cmap: Union[str, Colormap] = "YlOrBr"
    sigbar_min_logp: float = 2.0
    sigbar_max_logp: float = 10.0
    sigbar_norm: Optional[Normalize] = None
    sigbar_width: float = 0.06
    sigbar_height: float = 0.8
    sigbar_alpha: float = 1.0
    font: str = "Helvetica"
    fontsize: float = 9
    max_words: Optional[int] = None
    wrap_text: bool = True
    wrap_width: Optional[int] = None
    overflow: str = "wrap"
    omit_words: Optional[Tuple[str, ...]] = None
    label_fields: Tuple[str, ...] = ("label", "n", "p")
    label_prefix: Optional[str] = None
    label_overrides: Optional[Dict[int, str]] = None
    label_color: str = "black"
    label_alpha: float = 1.0
    placeholder_text: str = "—"
    placeholder_color: Optional[str] = None
    placeholder_alpha: Optional[float] = None
    skip_unlabeled: bool = False
    label_fontweight: str = "normal"
    dendrogram_color: str = "black"
    dendrogram_lw: float = 1.0
    label_left_pad: float = 0.02
    background_color: Optional[str] = None


class CondensedDendrogramPlot:
    """
    Class for a rendered condensed dendrogram figure.
    """

    def __init__(
        self,
        *,
        fig: plt.Figure,
        ax_den: plt.Axes,
        ax_sig: plt.Axes,
        ax_txt: plt.Axes,
        spec: Optional[CondensedDendrogramSpec] = None,
    ) -> None:
        """
        Initializes the CondensedDendrogramPlot handle.

        Kwargs:
            fig (plt.Figure): Rendered figure.
            ax_den (plt.Axes): Dendrogram axis.
            ax_sig (plt.Axes): Significance bar axis.
            ax_txt (plt.Axes): Label text axis.
            spec (Optional[CondensedDendrogramSpec]): Render specification used to
                rebuild if the figure is later closed. Defaults to None.
        """
        self.fig = fig
        self.ax_den = ax_den
        self.ax_sig = ax_sig
        self.ax_txt = ax_txt
        self._spec = spec

    def _ensure_open(self) -> None:
        """
        Ensures the backing figure is open, rebuilding it when possible.

        Raises:
            RuntimeError: If the figure is closed and no render spec is available.
        """
        if self._figure_is_open():
            return
        if self._spec is None:
            raise RuntimeError("Cannot reopen condensed dendrogram: render spec is unavailable.")
        rebuilt = _render_condensed(self._spec)
        self.fig = rebuilt.fig
        self.ax_den = rebuilt.ax_den
        self.ax_sig = rebuilt.ax_sig
        self.ax_txt = rebuilt.ax_txt

    def _figure_is_open(self) -> bool:
        """
        Checks whether the figure handle is still open.

        Returns:
            bool: True if the figure exists and is open, False otherwise.
        """
        try:
            return self.fig.number in plt.get_fignums()
        except (AttributeError, RuntimeError, ValueError):
            return False

    def save(self, path: Union[str, PathLike[str]], **kwargs: Any) -> None:
        """
        Saves the rendered condensed dendrogram figure.

        Args:
            path (Union[str, PathLike[str]]): Output path for the figure.

        Kwargs:
            **kwargs: Additional matplotlib savefig options. Defaults to {}.

        Raises:
            RuntimeError: If the figure has been closed and cannot be rebuilt.
        """
        self._ensure_open()
        self.fig.savefig(
            path,
            facecolor=self.fig.get_facecolor(),
            **kwargs,
        )

    def show(self) -> None:
        """
        Shows the rendered condensed dendrogram figure.

        Raises:
            RuntimeError: If the figure has been closed and cannot be rebuilt.
        """
        self._ensure_open()
        plt.show()


def _validate_condensed_inputs(
    results: Results,
    label_fields: Sequence[str],
    label_prefix: Optional[str],
) -> None:
    """
    Validates inputs for condensed dendrogram plotting.

    Args:
        results (Results): Enrichment results exposing cluster_layout() and clusters.
        label_fields (Sequence[str]): Fields to include in labels ("label", "n", "p", "q").
        label_prefix (Optional[str]): Label prefix mode.

    Raises:
        AttributeError: If required attributes are missing from results.
        ValueError: If no clusters are available or label_fields is invalid.
    """
    # Validation
    allowed = {"label", "n", "p", "q"}
    if not set(label_fields).issubset(allowed):
        raise ValueError(f"label_fields must be a subset of {allowed}")
    if label_prefix not in {None, "cid"}:
        raise ValueError("label_prefix must be one of {None, 'cid'}")
    if not list(results.cluster_layout().cluster_spans):
        raise ValueError("No clusters found")


def _get_master_linkage(results: Results) -> Tuple[np.ndarray, Clusters]:
    """
    Resolves the master linkage matrix and cluster container.

    Args:
        results (Results): Enrichment results exposing clusters.

    Returns:
        Tuple[np.ndarray, Clusters]: Master linkage matrix and Clusters instance.

    Raises:
        AttributeError: If required attributes are missing from results.
    """
    clusters = results.clusters
    if clusters is None or clusters.linkage_matrix is None:
        raise AttributeError(
            "results.clusters.linkage_matrix is required to preserve the master dendrogram "
            "order/heights"
        )
    return clusters.linkage_matrix, clusters


def _resolve_cluster_order(
    Z_master: np.ndarray,
    results: Results,
    clusters: Clusters,
) -> Tuple[list[int], Dict[Hashable, int]]:
    """
    Resolves cluster order from master dendrogram leaf order.

    Args:
        Z_master (np.ndarray): Master linkage matrix.
        results (Results): Enrichment results exposing matrix and clusters.
        clusters (Clusters): Clusters instance.

    Returns:
        Tuple[list[int], Dict[Hashable, int]]: Ordered list of cluster ids and
            mapping row label -> cluster id.

    Raises:
        ValueError: If no cluster ids could be mapped from master leaf order.
    """
    # Map master leaf order to cluster ids
    row_labels = results.matrix.labels
    cluster_to_rows = clusters.cluster_to_labels
    row_to_cluster = {row_id: int(cid) for cid, rows in cluster_to_rows.items() for row_id in rows}
    ordered_cluster_ids: list[int] = []
    seen: set[int] = set()
    # Scan master leaf order and collect cluster ids
    for i in leaves_list(Z_master):
        cid = row_to_cluster.get(row_labels[int(i)], None)
        if cid is None or cid in seen:
            continue
        ordered_cluster_ids.append(cid)
        seen.add(cid)
    # Validation
    if not ordered_cluster_ids:
        raise ValueError("No cluster ids could be mapped from master leaf order")

    return ordered_cluster_ids, row_to_cluster


def _prepare_cluster_labels(
    cluster_ids: Sequence[int],
    cluster_labels: pd.DataFrame,
    clusters: Clusters,
    *,
    label_prefix: Optional[str] = None,
    label_overrides: Optional[Dict[int, str]] = None,
    omit_words: Optional[Sequence[str]] = None,
    max_words: Optional[int] = None,
    placeholder_text: str = "—",
    wrap_text: bool = True,
    wrap_width: Optional[int] = None,
    overflow: str = "wrap",
) -> Tuple[
    list[str],
    np.ndarray,
    Dict[int, ClusterLabelInfo],
    Optional[Dict[int, int]],
    np.ndarray,
]:
    """
    Prepares cluster labels and ranking scores for condensed dendrogram plotting.

    Args:
        cluster_ids (Sequence[int]): Ordered list of cluster ids.
        cluster_labels (pd.DataFrame): DataFrame with columns: cluster, label,
            and optional pval/qval/score/n.
        clusters (Clusters): Clusters instance.

    Kwargs:
        label_prefix (Optional[str]): Label prefix mode. Defaults to None.
        label_overrides (Optional[Dict[int, str]]): Normalized mapping cluster_id -> custom label.
            Defaults to None.
        omit_words (Optional[Sequence[str]]): Words to omit from cluster labels. Defaults to None.
        max_words (Optional[int]): Maximum words in cluster labels. Defaults to None.
        placeholder_text (str): Text for clusters without generated labels. Defaults to "—".
        wrap_text (bool): Whether to wrap cluster labels. Defaults to True.
        wrap_width (Optional[int]): Maximum characters per line when wrapping. Defaults to None.
        overflow (str): Overflow handling when truncating ("wrap" or "ellipsis"). Defaults to "wrap".

    Returns:
        Tuple[
            list[str],
            np.ndarray,
            Dict[int, ClusterLabelInfo],
            Optional[Dict[int, int]],
            np.ndarray,
        ]: (
            List of formatted cluster labels,
            Array of ranking scores per cluster,
            Mapping cluster_id -> ClusterLabelInfo,
            Optional mapping cluster_id -> size,
            Y positions for cluster labels,
        )
    """
    if label_overrides is None:
        label_overrides = {}
    # Build label map from DataFrame
    lab_map: Dict[int, ClusterLabelInfo] = {}
    for _, row in cluster_labels.iterrows():
        pval_raw = row.get("pval", np.nan)
        qval_raw = row.get("qval", np.nan)
        score_raw = row.get("score", pval_raw if not pd.isna(pval_raw) else qval_raw)
        pval = float(pval_raw) if pval_raw is not None and not pd.isna(pval_raw) else np.nan
        qval = float(qval_raw) if qval_raw is not None and not pd.isna(qval_raw) else np.nan
        score = float(score_raw) if score_raw is not None and not pd.isna(score_raw) else np.nan
        lab_map[int(row["cluster"])] = ClusterLabelInfo(
            label=str(row["label"]),
            pval=pval,
            qval=qval,
            n_members=row.get("n", None),
            score=score,
        )
    cluster_sizes = getattr(clusters, "cluster_sizes", None) if clusters is not None else None
    cluster_sizes = dict(cluster_sizes) if cluster_sizes is not None else None

    # Prepare labels and ranking scores in order
    labels: list[str] = []
    scores: list[float] = []
    for cid in cluster_ids:
        if cid not in lab_map:
            labels.append(placeholder_text)
            scores.append(np.nan)
            continue
        # Apply overrides and formatting, then collect
        lab_info = lab_map[cid]
        lab = lab_info.label
        if label_prefix == "cid":
            lab = f"{cid}. {lab}"
        if cid in label_overrides:
            lab = label_overrides[cid]
        labels.append(
            apply_label_text_policy(
                lab,
                omit_words=omit_words,
                max_words=max_words,
                overflow=overflow,
                wrap_text=wrap_text,
                wrap_width=wrap_width,
            )
        )
        scores.append(lab_info.score)
    # Convert ranking scores to array
    score_arr = np.asarray(scores, float)
    y = np.arange(len(cluster_ids)) * 10.0 + 5.0

    return labels, score_arr, lab_map, cluster_sizes, y


def _resolve_condensed_label_style(
    *,
    is_placeholder: bool,
    label_color: str,
    label_alpha: float,
    placeholder_color: Optional[str],
    placeholder_alpha: Optional[float],
) -> Tuple[str, float]:
    """
    Resolves color/alpha for condensed label text, with placeholder-specific overrides.

    Args:
        is_placeholder (bool): Whether the current cluster label is a placeholder.
        label_color (str): Base label color.
        label_alpha (float): Base label alpha.
        placeholder_color (Optional[str]): Optional placeholder color override.
        placeholder_alpha (Optional[float]): Optional placeholder alpha override.

    Returns:
        Tuple[str, float]: Resolved (color, alpha) pair.
    """
    if is_placeholder:
        return (
            placeholder_color if placeholder_color is not None else label_color,
            placeholder_alpha if placeholder_alpha is not None else label_alpha,
        )

    return label_color, label_alpha


def _compose_condensed_cluster_text(
    *,
    label: str,
    is_placeholder: bool,
    label_fields: Sequence[str],
    n_members: Optional[int],
    pval: Optional[float],
    qval: Optional[float],
    wrap_text: bool,
    wrap_width: Optional[int],
) -> str:
    """
    Composes rendered text for one condensed cluster label.

    Args:
        label (str): Base text label.
        is_placeholder (bool): Whether the current cluster label is a placeholder.
        label_fields (Sequence[str]): Fields to include in non-placeholder labels.
        n_members (Optional[int]): Cluster size.
        pval (Optional[float]): Cluster p-value.
        qval (Optional[float]): Cluster q-value.
        wrap_text (bool): Whether to wrap label text.
        wrap_width (Optional[int]): Maximum characters per wrapped line.

    Returns:
        str: Final rendered label text.
    """
    if is_placeholder:
        return label

    has_label, stats = collect_label_stats(
        label_fields,
        n_members=n_members,
        pval=pval,
        qval=qval,
    )
    return compose_label_text(
        label,
        has_label=has_label,
        stats=stats,
        wrap_text=wrap_text,
        wrap_width=wrap_width,
    )


def _compute_condensed_dendrogram(
    Z_master: np.ndarray,
    row_to_cluster: Dict[Hashable, int],
    cluster_ids: Sequence[int],
    row_labels: Sequence[Hashable],
) -> DendrogramData:
    """
    Computes condensed dendrogram data from master linkage.

    Args:
        Z_master (np.ndarray): Master linkage matrix.
        row_to_cluster (Dict[Hashable, int]): Mapping row label -> cluster id.
        cluster_ids (Sequence[int]): Ordered list of cluster ids.
        row_labels (Sequence[Hashable]): Row labels corresponding to master linkage.

    Returns:
        DendrogramData: Condensed dendrogram data.

    Raises:
        ValueError: If fewer than two clusters are available for branching.
    """
    if len(cluster_ids) < 2:
        raise ValueError(
            "Condensed dendrogram requires at least two clusters to draw branches "
            f"(found {len(cluster_ids)}). "
            "This often occurs when the current matrix resolves to one cluster. "
            "Skip condensed plotting for this run or adjust clustering settings."
        )

    # Build condensed linkage matrix
    n_master = Z_master.shape[0] + 1
    row_index_to_cluster = {
        i: row_to_cluster[row_labels[int(i)]]
        for i in range(n_master)
        if row_labels[int(i)] in row_to_cluster
    }
    Zc = _condense_linkage_to_clusters(Z_master, row_index_to_cluster, cluster_ids)
    return dendrogram(Zc, orientation="left", no_labels=True, no_plot=True)


def _setup_condensed_axes(
    figsize: Sequence[float],
    sigbar_width: float,
    label_left_pad: float,
    background_color: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes]:
    """
    Sets up the condensed dendrogram axes.

    Args:
        figsize (Sequence[float]): Figure size (width, height).
        sigbar_width (float): Width of significance bar (axes fraction).
        label_left_pad (float): Left padding for labels (axes fraction).
        background_color (Optional[str]): Background color for figure and axes. Defaults to None.
    """
    # Build axes layout for dendrogram, sigbar, and labels
    fig = plt.figure(figsize=figsize)
    ax_den = fig.add_axes([0.05, 0.05, 0.60, 0.90], frameon=False)
    ax_sig = fig.add_axes([0.66, 0.05, sigbar_width, 0.90], frameon=False)
    txt_x0 = 0.66 + sigbar_width + float(label_left_pad)
    ax_txt = fig.add_axes(
        [txt_x0, 0.05, 0.33 - sigbar_width - float(label_left_pad), 0.90],
        frameon=False,
    )
    if background_color is not None:
        fig.patch.set_facecolor(background_color)
        for ax in (ax_den, ax_sig, ax_txt):
            ax.set_facecolor(background_color)

    return fig, ax_den, ax_sig, ax_txt


def _finalize_axes(*axes: plt.Axes) -> None:
    """
    Finalizes axes by removing ticks and spines.

    Args:
        axes (plt.Axes): Matplotlib Axes to finalize.
    """
    for ax in axes:
        ax.set(xticks=[], yticks=[])
        for sp in ax.spines.values():
            sp.set_visible(False)


def _condense_linkage_to_clusters(
    Z_master: np.ndarray,
    row_index_to_cluster: Dict[int, int],
    cluster_ids: Sequence[int],
) -> np.ndarray:
    """
    Condenses master linkage matrix to cluster-level dendrogram. Preserves master
    dendrogram heights and leaf order by treating each cluster as a single leaf
    in the condensed tree.

    Args:
        Z_master (np.ndarray): Master linkage matrix (n-1, 4).
        row_index_to_cluster (Dict[int, int]): Mapping row index -> cluster id.
        cluster_ids (Sequence[int]): Ordered list of cluster ids to include.

    Returns:
        np.ndarray: Condensed linkage matrix (k-1, 4).

    Raises:
        ValueError: If the condensed linkage is incomplete.
    """
    cluster_index = {cid: i for i, cid in enumerate(cluster_ids)}
    # Build leaf group mapping
    n_master = Z_master.shape[0] + 1
    leaf_groups: list[Optional[int]] = []
    for i in range(n_master):
        cid = row_index_to_cluster.get(i, None)
        if cid is None or cid not in cluster_index:
            leaf_groups.append(None)
            continue
        leaf_groups.append(cluster_index[cid])

    # Process merges
    node_groups: Dict[int, set[int]] = {}
    for i, grp in enumerate(leaf_groups):
        node_groups[i] = set() if grp is None else {int(grp)}
    # Build condensed linkage rows by scanning master merges
    Zc_rows: list[list[float]] = []
    rep_to_id = {frozenset({i}): i for i in range(len(cluster_ids))}
    next_id = len(cluster_ids)
    for t in range(Z_master.shape[0]):
        a = int(Z_master[t, 0])
        b = int(Z_master[t, 1])
        h = float(Z_master[t, 2])
        # Merge groups
        Ga = node_groups.get(a, set())
        Gb = node_groups.get(b, set())
        G = Ga | Gb
        node_groups[n_master + t] = G
        # Only record merge if it connects two distinct cluster groups
        if len(G) <= 1 or Ga == Gb:
            continue
        # Map group representatives to ids
        ra = frozenset(Ga)
        rb = frozenset(Gb)
        rg = frozenset(G)
        if ra not in rep_to_id:
            rep_to_id[ra] = next_id
            next_id += 1
        if rb not in rep_to_id:
            rep_to_id[rb] = next_id
            next_id += 1
        # Create condensed linkage row
        ida = rep_to_id[ra]
        idb = rep_to_id[rb]
        if rg in rep_to_id:
            continue
        rep_to_id[rg] = next_id
        Zc_rows.append([ida, idb, h, float(len(G))])
        next_id += 1

    # Validate completeness
    expected = len(cluster_ids) - 1
    if len(Zc_rows) != expected:
        raise ValueError(
            "Condensed linkage is incomplete (expected "
            f"{expected} merges, got {len(Zc_rows)}). "
            "This usually means clusters are not all connected under the provided "
            "master linkage."
        )

    return np.asarray(Zc_rows, dtype=float)


def _get_cluster_size(
    cluster_id: int,
    *,
    label_map: Dict[int, ClusterLabelInfo],
    cluster_sizes: Optional[Dict[int, int]] = None,
    cluster_to_labels: Dict[int, Collection[Hashable]],
) -> Optional[int]:
    """
    Resolves cluster size from available sources.

    Args:
        cluster_id (int): Cluster id.

    Kwargs:
        label_map (Dict[int, ClusterLabelInfo]): Mapping cluster_id -> ClusterLabelInfo.
        cluster_sizes (Optional[Dict[int, int]]): Pre-computed cluster sizes. Defaults to None.
        cluster_to_labels (Dict[int, Collection[Hashable]]): Mapping cluster_id -> labels

    Returns:
        Optional[int]: Cluster size, or None if not found.
    """
    # Check label map first, then cluster_sizes, then compute from cluster_to_labels
    if cluster_id in label_map:
        n0 = label_map[cluster_id].n_members
        if n0 is not None:
            try:
                if np.isfinite(n0):
                    return int(n0)
            except TypeError:
                pass
    if cluster_sizes is not None and cluster_id in cluster_sizes:
        return int(cluster_sizes[cluster_id])
    cluster_members = cluster_to_labels.get(cluster_id, None)
    if cluster_members is not None:
        return int(len(cluster_members))
    return None


def _render_condensed(spec: CondensedDendrogramSpec) -> CondensedDendrogramPlot:
    """
    Renders a condensed dendrogram from an immutable render specification.

    Args:
        spec (CondensedDendrogramSpec): Normalized rendering specification.

    Raises:
        AttributeError: If required attributes are missing from results.
        ValueError: If no clusters are found or plotting options are invalid.
        TypeError: If label_overrides is not a dict.

    Returns:
        CondensedDendrogramPlot: Rendered condensed dendrogram figure handle.
    """
    # Validation
    _validate_condensed_inputs(spec.results, spec.label_fields, spec.label_prefix)
    cluster_labels = spec.results.cluster_labels(
        rank_by=spec.rank_by,
        label_mode=spec.label_mode,
        max_words=spec.max_words,
    )
    # Get master linkage and clusters
    Z_master, clusters = _get_master_linkage(spec.results)
    cluster_ids, row_to_cluster = _resolve_cluster_order(Z_master, spec.results, clusters)
    labels, scores, lab_map, cluster_sizes, y = _prepare_cluster_labels(
        cluster_ids,
        cluster_labels,
        clusters,
        label_prefix=spec.label_prefix,
        label_overrides=spec.label_overrides,
        omit_words=spec.omit_words,
        max_words=spec.max_words,
        placeholder_text=spec.placeholder_text,
        wrap_text=spec.wrap_text,
        wrap_width=spec.wrap_width,
        overflow=spec.overflow,
    )
    d = _compute_condensed_dendrogram(
        Z_master,
        row_to_cluster,
        cluster_ids,
        spec.results.matrix.labels,
    )
    _fig, ax_den, ax_sig, ax_txt = _setup_condensed_axes(
        spec.figsize,
        spec.sigbar_width,
        spec.label_left_pad,
        spec.background_color,
    )

    # Render dendrogram
    k = len(cluster_ids)
    cluster_y = {i: y[i] for i in range(k)}
    leaf_order = d["leaves"]
    max_h = max(map(max, d["dcoord"]))
    x_pad = max_h * 0.05
    for xs, ys in zip(d["dcoord"], d["icoord"]):
        mapped = []
        for yval in ys:
            slot = int(round((yval - 5.0) / 10.0))
            slot = max(0, min(k - 1, slot))
            mapped.append(cluster_y[int(leaf_order[slot])])
        ax_den.plot(xs, mapped, color=spec.dendrogram_color, lw=spec.dendrogram_lw)
    ax_den.set(xlim=(max_h + x_pad, 0.0), ylim=(k * 10, 0))

    # Render significance bar
    cmap = plt.get_cmap(spec.sigbar_cmap)
    ax_sig.set(xlim=(0, 1), ylim=(k * 10, 0))
    if not np.isfinite(spec.sigbar_height) or spec.sigbar_height <= 0 or spec.sigbar_height > 1:
        raise ValueError("sigbar_height must be in the range (0, 1]")

    # Normalize scores to [0, 1] based on log-transformed p-values, then render bars
    norm = spec.sigbar_norm
    denom = spec.sigbar_max_logp - spec.sigbar_min_logp
    row_pitch = float(np.mean(np.diff(y))) if len(y) > 1 else 10.0
    bar_height = row_pitch * float(spec.sigbar_height)
    for i, score in enumerate(scores):
        if not np.isfinite(score) or score <= 0:
            val = 0.0
        else:
            lp = -np.log10(score)
            if norm is None:
                val = np.clip((lp - spec.sigbar_min_logp) / denom, 0, 1)
            else:
                try:
                    val = float(norm(lp))
                except TypeError:
                    val = float(norm([lp])[0])
        ax_sig.add_patch(
            plt.Rectangle(
                (0, y[i] - (bar_height / 2.0)),
                1,
                bar_height,
                facecolor=cmap(val),
                edgecolor="none",
                alpha=spec.sigbar_alpha,
            )
        )

    # Render text labels
    ax_txt.set(xlim=(0, 1), ylim=(k * 10, 0))
    # Mapping cluster -> row ids is needed for _get_cluster_size.
    cluster_to_rows = getattr(clusters, "cluster_to_labels", None) or {}
    # Format and place per-cluster label text
    for cid, yi, lab in zip(cluster_ids, y, labels):
        is_placeholder = int(cid) not in lab_map
        if is_placeholder and spec.skip_unlabeled:
            continue

        n = None
        pval_value = None
        qval_value = None
        if not is_placeholder and "n" in spec.label_fields:
            n = _get_cluster_size(
                int(cid),
                label_map=lab_map,
                cluster_sizes=cluster_sizes,
                cluster_to_labels=cluster_to_rows,
            )
        if not is_placeholder:
            label_info = lab_map[int(cid)]
            pval_value = label_info.pval if np.isfinite(label_info.pval) else None
            qval_value = label_info.qval if np.isfinite(label_info.qval) else None
        txt = _compose_condensed_cluster_text(
            label=lab,
            is_placeholder=is_placeholder,
            label_fields=spec.label_fields,
            n_members=n,
            pval=pval_value,
            qval=qval_value,
            wrap_text=spec.wrap_text,
            wrap_width=spec.wrap_width,
        )
        text_color, text_alpha = _resolve_condensed_label_style(
            is_placeholder=is_placeholder,
            label_color=spec.label_color,
            label_alpha=spec.label_alpha,
            placeholder_color=spec.placeholder_color,
            placeholder_alpha=spec.placeholder_alpha,
        )
        ax_txt.text(
            0.0,
            yi,
            txt,
            va="center",
            ha="left",
            font=spec.font,
            fontsize=spec.fontsize,
            color=text_color,
            alpha=text_alpha,
            fontweight=spec.label_fontweight,
        )

    # Clean axes and return rendered figure handle
    _finalize_axes(ax_den, ax_sig, ax_txt)
    return CondensedDendrogramPlot(
        fig=_fig,
        ax_den=ax_den,
        ax_sig=ax_sig,
        ax_txt=ax_txt,
        spec=spec,
    )


def plot_dendrogram_condensed(
    results: Results,
    *,
    rank_by: str = "p",
    label_mode: str = "top_term",
    figsize: Sequence[float] = (10, 10),
    sigbar_cmap: Union[str, Colormap] = "YlOrBr",
    sigbar_min_logp: float = 2.0,
    sigbar_max_logp: float = 10.0,
    sigbar_norm: Optional[Normalize] = None,
    sigbar_width: float = 0.06,
    sigbar_height: float = 0.8,
    sigbar_alpha: float = 1.0,
    font: str = "Helvetica",
    fontsize: float = 9,
    max_words: Optional[int] = None,
    wrap_text: bool = True,
    wrap_width: Optional[int] = None,
    overflow: str = "wrap",
    omit_words: Optional[Sequence[str]] = None,
    label_fields: Sequence[str] = ("label", "n", "p"),
    label_prefix: Optional[str] = None,
    label_overrides: Optional[Dict[int, str]] = None,
    label_color: str = "black",
    label_alpha: float = 1.0,
    placeholder_text: str = "—",
    placeholder_color: Optional[str] = None,
    placeholder_alpha: Optional[float] = None,
    skip_unlabeled: bool = False,
    label_fontweight: str = "normal",
    dendrogram_color: str = "black",
    dendrogram_lw: float = 1.0,
    label_left_pad: float = 0.02,
    background_color: Optional[str] = None,
) -> CondensedDendrogramPlot:
    """
    Builds and returns a cluster-level condensed dendrogram plot.
    Leaf order and branch heights are preserved from the master dendrogram by condensing
    the master linkage matrix at the cluster level.

    Args:
        results (Results): Enrichment results exposing cluster_layout() and clusters.

    Kwargs:
        rank_by (str): Ranking statistic for representative terms, one of {"p", "q"}.
            Defaults to "p".
        label_mode (str): Label mode, one of {"top_term", "compressed"}. Defaults to "top_term".
        figsize (Sequence[float]): Figure size (width, height). Defaults to (10, 10).
        sigbar_cmap (Union[str, Colormap]): Colormap for significance bar. Defaults to "YlOrBr".
        sigbar_min_logp (float): Minimum -log10(score) for significance bar scaling.
            Defaults to 2.0.
        sigbar_max_logp (float): Maximum -log10(score) for significance bar scaling.
            Defaults to 10.0.
        sigbar_norm (Optional[Normalize]): Optional normalization for significance bar. Defaults to None.
        sigbar_width (float): Width of significance bar (axes fraction). Defaults to 0.06.
        sigbar_height (float): Height of each significance bar as a fraction of row pitch.
            Defaults to 0.8.
        sigbar_alpha (float): Alpha for significance bar. Defaults to 1.0.
        font (str): Font family for labels. Defaults to "Helvetica".
        fontsize (float): Font size for labels. Defaults to 9.
        max_words (Optional[int]): Maximum words in cluster labels. Defaults to None.
        wrap_text (bool): Whether to wrap cluster labels. Defaults to True.
        wrap_width (Optional[int]): Maximum characters per line when wrapping. Defaults to None.
        overflow (str): Overflow handling when truncating ("wrap" or "ellipsis"). Defaults to "wrap".
        omit_words (Optional[Sequence[str]]): Words to omit from cluster labels. Defaults to None.
        label_fields (Sequence[str]): Fields to include in labels ("label", "n", "p", "q").
            Defaults to ("label", "n", "p").
        label_prefix (Optional[str]): Optional prefix mode for display labels.
            Supported values are None and "cid". Defaults to None.
        label_overrides (Optional[Dict[int, str]]): Mapping cluster_id -> custom label.
            Defaults to None.
        label_color (str): Color for cluster labels. Defaults to "black".
        label_alpha (float): Alpha for cluster labels. Defaults to 1.0.
        placeholder_text (str): Text for unlabeled clusters. Defaults to "—".
        placeholder_color (Optional[str]): Color override for placeholder labels.
            If None, falls back to `label_color`.
        placeholder_alpha (Optional[float]): Alpha override for placeholder labels.
            If None, falls back to `label_alpha`.
        skip_unlabeled (bool): Whether placeholder labels should be skipped.
            Defaults to False.
        label_fontweight (str): Font weight for cluster labels. Defaults to "normal".
        dendrogram_color (str): Color for dendrogram lines. Defaults to "black".
        dendrogram_lw (float): Line width for dendrogram lines. Defaults to 1.0.
        label_left_pad (float): Left padding for labels (axes fraction). Defaults to 0.02.
        background_color (Optional[str]): Background color for figure and axes. Defaults to None.

    Raises:
        AttributeError: If required attributes are missing from results.
        ValueError: If no clusters are found or plotting options are invalid.
        TypeError: If label_overrides is not a dict.

    Returns:
        CondensedDendrogramPlot: Rendered condensed dendrogram figure handle.
    """
    override_map = _parse_label_overrides(label_overrides)
    spec = CondensedDendrogramSpec(
        results=results,
        rank_by=rank_by,
        label_mode=label_mode,
        figsize=tuple(figsize),
        sigbar_cmap=sigbar_cmap,
        sigbar_min_logp=sigbar_min_logp,
        sigbar_max_logp=sigbar_max_logp,
        sigbar_norm=sigbar_norm,
        sigbar_width=sigbar_width,
        sigbar_height=sigbar_height,
        sigbar_alpha=sigbar_alpha,
        font=font,
        fontsize=fontsize,
        max_words=max_words,
        wrap_text=wrap_text,
        wrap_width=wrap_width,
        overflow=overflow,
        omit_words=None if omit_words is None else tuple(omit_words),
        label_fields=tuple(label_fields),
        label_prefix=label_prefix,
        label_overrides=override_map,
        label_color=label_color,
        label_alpha=label_alpha,
        placeholder_text=placeholder_text,
        placeholder_color=placeholder_color,
        placeholder_alpha=placeholder_alpha,
        skip_unlabeled=skip_unlabeled,
        label_fontweight=label_fontweight,
        dendrogram_color=dendrogram_color,
        dendrogram_lw=dendrogram_lw,
        label_left_pad=label_left_pad,
        background_color=background_color,
    )
    return _render_condensed(spec)
