"""
himalayas/plot/condensed_dendrogram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import (
    Collection,
    Dict,
    Hashable,
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

from .renderers._label_format import collect_label_stats

if TYPE_CHECKING:
    from ..core.clustering import Clusters
    from ..core.results import Results


class DendrogramData(TypedDict):
    """
    Type class for condensed dendrogram data.
    """

    icoord: Sequence[Sequence[float]]
    dcoord: Sequence[Sequence[float]]
    leaves: Sequence[int]


def _validate_condensed_inputs(
    results: Results,
    cluster_labels: pd.DataFrame,
    label_fields: Sequence[str],
    label_overrides: Optional[Dict[int, str]] = None,
) -> None:
    """
    Validates inputs for condensed dendrogram plotting.

    Args:
        results (Results): Enrichment results exposing cluster_layout() and clusters.
        cluster_labels (pd.DataFrame): DataFrame with columns: cluster, label, pval.
        label_fields (Sequence[str]): Fields to include in labels ("label", "n", "p").
        label_overrides (Optional[Dict[int, str]]): Mapping cluster_id -> custom label. Defaults to None.

    Raises:
        AttributeError: If required attributes are missing from results.
        ValueError: If cluster_labels is missing required columns or no clusters found.
        TypeError: If label_overrides is not a dict.
    """
    # Validation
    if not hasattr(results, "cluster_layout"):
        raise AttributeError("results must expose cluster_layout()")
    required = {"cluster", "label", "pval"}
    if not required.issubset(cluster_labels.columns):
        raise ValueError(f"cluster_labels must contain {required}")
    allowed = {"label", "n", "p"}
    if not set(label_fields).issubset(allowed):
        raise ValueError(f"label_fields must be a subset of {allowed}")
    if label_overrides is not None and not isinstance(label_overrides, dict):
        raise TypeError("label_overrides must be a dict mapping cluster_id -> label string")
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
    clusters = getattr(results, "clusters", None)
    Z_master = getattr(clusters, "linkage_matrix", None) if clusters is not None else None
    if Z_master is None or clusters is None:
        raise AttributeError(
            "results.clusters.linkage_matrix is required to preserve the master dendrogram "
            "order/heights"
        )
    return Z_master, clusters


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
            mapping gene label -> cluster id.

    Raises:
        ValueError: If no cluster ids could be mapped from master leaf order.
    """
    # Map master leaf order to cluster ids
    row_labels = results.matrix.labels
    c2g = getattr(clusters, "cluster_to_labels", None) or {}
    gene_to_cluster = {g: int(cid) for cid, genes in c2g.items() for g in genes}
    ordered_cluster_ids: list[int] = []
    seen: set[int] = set()
    # Scan master leaf order and collect cluster ids
    for i in leaves_list(Z_master):
        cid = gene_to_cluster.get(row_labels[int(i)], None)
        if cid is None or cid in seen:
            continue
        ordered_cluster_ids.append(cid)
        seen.add(cid)
    # Validation
    if not ordered_cluster_ids:
        raise ValueError("No cluster ids could be mapped from master leaf order")

    return ordered_cluster_ids, gene_to_cluster


def _prepare_cluster_labels(
    cluster_ids: Sequence[int],
    cluster_labels: pd.DataFrame,
    clusters: Clusters,
    *,
    label_overrides: Optional[Dict[int, str]] = None,
    omit_words: Optional[Sequence[str]] = None,
    max_words: Optional[int] = None,
    wrap_text: bool = True,
    wrap_width: Optional[int] = None,
    overflow: str = "wrap",
) -> Tuple[
    list[str],
    np.ndarray,
    Dict[int, Tuple[str, float, Optional[float]]],
    Optional[Dict[int, int]],
    np.ndarray,
]:
    """
    Prepares cluster labels and p-values for condensed dendrogram plotting.

    Args:
        cluster_ids (Sequence[int]): Ordered list of cluster ids.
        cluster_labels (pd.DataFrame): DataFrame with columns: cluster, label, pval.
        clusters (Clusters): Clusters instance.
        label_overrides (Optional[Dict[int, str]]): Mapping cluster_id -> custom label. Defaults to None.
        omit_words (Optional[Sequence[str]]): Words to omit from cluster labels. Defaults to None.
        max_words (Optional[int]): Maximum words in cluster labels. Defaults to None.
        wrap_text (bool): Whether to wrap cluster labels. Defaults to True.
        wrap_width (Optional[int]): Maximum characters per line when wrapping. Defaults to None.
        overflow (str): Overflow handling when truncating ("wrap" or "ellipsis"). Defaults to "wrap".

    Returns:
        Tuple[
            list[str],
            np.ndarray,
            Dict[int, Tuple[str, float, Optional[float]]],
            Optional[Dict[int, int]],
            np.ndarray,
        ]: (
            List of formatted cluster labels,
            Array of p-values per cluster,
            Mapping cluster_id -> (label, pval, n),
            Optional mapping cluster_id -> size,
            Y positions for cluster labels,
        )
    """
    if label_overrides is None:
        label_overrides = {}
    # Build label map from DataFrame
    lab_map: Dict[int, Tuple[str, float, Optional[float]]] = {
        int(r["cluster"]): (str(r["label"]), float(r["pval"]), r.get("n", None))
        for _, r in cluster_labels.iterrows()
    }
    cluster_sizes = getattr(clusters, "cluster_sizes", None) if clusters is not None else None
    cluster_sizes = dict(cluster_sizes) if cluster_sizes is not None else None

    # Prepare labels and p-values in order
    labels: list[str] = []
    pvals: list[float] = []
    for cid in cluster_ids:
        if cid not in lab_map:
            labels.append("—")
            pvals.append(np.nan)
            continue
        # Apply overrides and formatting, then collect
        lab, p, _ = lab_map[cid]
        if cid in label_overrides:
            lab = str(label_overrides[cid])
        labels.append(
            _format_cluster_label(
                lab,
                omit_words=omit_words,
                max_words=max_words,
                overflow=overflow,
                wrap_text=wrap_text,
                wrap_width=wrap_width,
            )
        )
        pvals.append(p)
    # Convert p-values to array
    pvals_arr = np.asarray(pvals, float)
    y = np.arange(len(cluster_ids)) * 10.0 + 5.0

    return labels, pvals_arr, lab_map, cluster_sizes, y


def _compute_condensed_dendrogram(
    Z_master: np.ndarray,
    gene_to_cluster: Dict[Hashable, int],
    cluster_ids: Sequence[int],
    row_labels: Sequence[Hashable],
) -> DendrogramData:
    """
    Computes condensed dendrogram data from master linkage.

    Args:
        Z_master (np.ndarray): Master linkage matrix.
        gene_to_cluster (Dict[Hashable, int]): Mapping gene label -> cluster id.
        cluster_ids (Sequence[int]): Ordered list of cluster ids.
        row_labels (Sequence[Hashable]): Gene labels corresponding to master linkage.

    Returns:
        DendrogramData: Condensed dendrogram data.
    """
    # Build condensed linkage matrix
    n_master = Z_master.shape[0] + 1
    gene_to_cluster_index = {
        i: gene_to_cluster[row_labels[int(i)]]
        for i in range(n_master)
        if row_labels[int(i)] in gene_to_cluster
    }
    Zc = _condense_linkage_to_clusters(Z_master, gene_to_cluster_index, cluster_ids)
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
    plt.show()


def _condense_linkage_to_clusters(
    Z_master: np.ndarray,
    gene_to_cluster: Dict[int, int],
    cluster_ids: Sequence[int],
) -> np.ndarray:
    """
    Condenses master linkage matrix to cluster-level dendrogram. Preserves master
    dendrogram heights and leaf order by treating each cluster as a single leaf
    in the condensed tree.

    Args:
        Z_master (np.ndarray): Master linkage matrix (n-1, 4).
        gene_to_cluster (Dict[int, int]): Mapping gene index -> cluster id.
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
        cid = gene_to_cluster.get(i, None)
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


def _format_cluster_label(
    raw_label: str,
    *,
    omit_words: Optional[Sequence[str]] = None,
    max_words: Optional[int] = None,
    overflow: str = "wrap",
    wrap_text: bool = True,
    wrap_width: Optional[int] = None,
) -> str:
    """
    Applies text policy to cluster label in the following order: omit words -> truncate -> wrap.

    Args:
        raw_label (str): Original cluster label.
        omit_words (Optional[Sequence[str]]): Words to omit (case-insensitive). Defaults to None.
        max_words (Optional[int]): Maximum number of words to keep. Defaults to None.
        overflow (str): Overflow handling when truncating ("wrap" or "ellipsis"). Defaults to "wrap".
        wrap_text (bool): Whether to apply text wrapping. Defaults to True.
        wrap_width (Optional[int]): Maximum characters per line when wrapping. Defaults to None.

    Returns:
        str: Formatted cluster label.
    """
    label = str(raw_label)
    # Omit words
    if omit_words:
        omit = {w.lower() for w in omit_words}
        words = [w for w in label.split() if w.lower() not in omit]
        label = " ".join(words) if words else label
    # Truncate to max words
    if max_words is not None:
        words = label.split()
        if len(words) > max_words:
            if overflow == "ellipsis" and max_words > 0:
                label = " ".join(words[: max_words - 1]) + "…"
            else:
                label = " ".join(words[:max_words])
    # Wrap text
    if wrap_text and wrap_width is not None and wrap_width > 0:
        wrapped_lines = []
        line = ""
        for word in label.split():
            if len(line) + len(word) + (1 if line else 0) <= wrap_width:
                line = f"{line} {word}".strip()
            else:
                if line:
                    wrapped_lines.append(line)
                line = word
        if line:
            wrapped_lines.append(line)
        label = "\n".join(wrapped_lines)

    return label


def _get_cluster_size(
    cluster_id: int,
    *,
    label_map: Dict[int, Tuple[str, float, Optional[float]]],
    cluster_sizes: Optional[Dict[int, int]] = None,
    cluster_to_labels: Dict[int, Collection[Hashable]],
) -> Optional[int]:
    """
    Resolves cluster size from available sources.

    Args:
        cluster_id (int): Cluster id.
        label_map (Dict[int, Tuple[str, float, Optional[float]]]): Mapping cluster_id ->
            (label, pval, n).
        cluster_sizes (Optional[Dict[int, int]]): Pre-computed cluster sizes. Defaults to None.
        cluster_to_labels (Dict[int, Collection[Hashable]]): Mapping cluster_id -> labels

    Returns:
        Optional[int]: Cluster size, or None if not found.
    """
    # Check label map first, then cluster_sizes, then compute from cluster_to_labels
    if cluster_id in label_map:
        n0 = label_map[cluster_id][2]
        if n0 is not None and np.isfinite(n0):
            return int(n0)
    if cluster_sizes is not None and cluster_id in cluster_sizes:
        return int(cluster_sizes[cluster_id])
    genes = cluster_to_labels.get(cluster_id, None)
    if genes is not None:
        return int(len(genes))
    return None


def plot_dendrogram_condensed(
    results: Results,
    cluster_labels: pd.DataFrame,
    *,
    figsize: Sequence[float] = (10, 10),
    sigbar_cmap: Union[str, Colormap] = "YlOrBr",
    sigbar_min_logp: float = 2.0,
    sigbar_max_logp: float = 10.0,
    sigbar_norm: Optional[Normalize] = None,
    sigbar_width: float = 0.06,
    sigbar_alpha: float = 0.9,
    font: str = "Helvetica",
    fontsize: float = 9,
    max_words: Optional[int] = None,
    wrap_text: bool = True,
    wrap_width: Optional[int] = None,
    overflow: str = "wrap",
    omit_words: Optional[Sequence[str]] = None,
    label_fields: Sequence[str] = ("label", "n", "p"),
    label_overrides: Optional[Dict[int, str]] = None,
    label_color: str = "black",
    label_alpha: float = 1.0,
    label_fontweight: str = "normal",
    dendrogram_color: str = "black",
    dendrogram_lw: float = 1.0,
    label_left_pad: float = 0.02,
    background_color: Optional[str] = None,
) -> None:
    """
    Plots cluster-level dendrogram using condensed master linkage. Leaf order and branch
    heights are preserved from the master dendrogram by condensing the master linkage
    matrix at the cluster level.

    Args:
        results (Results): Enrichment results exposing cluster_layout() and clusters.
        cluster_labels (pd.DataFrame): DataFrame with columns: cluster, label, pval.
        figsize (Sequence[float]): Figure size (width, height). Defaults to (10, 10).
        sigbar_cmap (Union[str, Colormap]): Colormap for significance bar. Defaults to "YlOrBr".
        sigbar_min_logp (float): Minimum -log10(p) for significance bar scaling. Defaults to 2.0.
        sigbar_max_logp (float): Maximum -log10(p) for significance bar scaling. Defaults to 10.0.
        sigbar_norm (Optional[Normalize]): Optional normalization for significance bar. Defaults to None.
        sigbar_width (float): Width of significance bar (axes fraction). Defaults to 0.06.
        sigbar_alpha (float): Alpha for significance bar. Defaults to 0.9.
        font (str): Font family for labels. Defaults to "Helvetica".
        fontsize (float): Font size for labels. Defaults to 9.
        max_words (Optional[int]): Maximum words in cluster labels. Defaults to None.
        wrap_text (bool): Whether to wrap cluster labels. Defaults to True.
        wrap_width (Optional[int]): Maximum characters per line when wrapping. Defaults to None.
        overflow (str): Overflow handling when truncating ("wrap" or "ellipsis"). Defaults to "wrap".
        omit_words (Optional[Sequence[str]]): Words to omit from cluster labels. Defaults to None.
        label_fields (Sequence[str]): Fields to include in labels ("label", "n", "p").
            Defaults to ("label", "n", "p").
        label_overrides (Optional[Dict[int, str]]): Mapping cluster_id -> custom label. Defaults to None.
        label_color (str): Color for cluster labels. Defaults to "black".
        label_alpha (float): Alpha for cluster labels. Defaults to 1.0.
        label_fontweight (str): Font weight for cluster labels. Defaults to "normal".
        dendrogram_color (str): Color for dendrogram lines. Defaults to "black".
        dendrogram_lw (float): Line width for dendrogram lines. Defaults to 1.0.
        label_left_pad (float): Left padding for labels (axes fraction). Defaults to 0.02.
        background_color (Optional[str]): Background color for figure and axes. Defaults to None.

    Raises:
        AttributeError: If required attributes are missing from results.
        ValueError: If cluster_labels is missing required columns or no clusters found.
        TypeError: If label_overrides is not a dict.
    """
    # Validation
    _validate_condensed_inputs(results, cluster_labels, label_fields, label_overrides)
    # Get master linkage and clusters
    Z_master, clusters = _get_master_linkage(results)
    cluster_ids, gene_to_cluster = _resolve_cluster_order(Z_master, results, clusters)
    label_overrides = label_overrides or {}
    labels, pvals, lab_map, cluster_sizes, y = _prepare_cluster_labels(
        cluster_ids,
        cluster_labels,
        clusters,
        label_overrides=label_overrides,
        omit_words=omit_words,
        max_words=max_words,
        wrap_text=wrap_text,
        wrap_width=wrap_width,
        overflow=overflow,
    )
    d = _compute_condensed_dendrogram(
        Z_master,
        gene_to_cluster,
        cluster_ids,
        results.matrix.labels,
    )
    _fig, ax_den, ax_sig, ax_txt = _setup_condensed_axes(
        figsize, sigbar_width, label_left_pad, background_color
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
        ax_den.plot(xs, mapped, color=dendrogram_color, lw=dendrogram_lw)
    ax_den.set(xlim=(max_h + x_pad, 0.0), ylim=(k * 10, 0))
    # Render significance bar
    cmap = plt.get_cmap(sigbar_cmap)
    ax_sig.set(xlim=(0, 1), ylim=(k * 10, 0))
    norm = sigbar_norm
    denom = sigbar_max_logp - sigbar_min_logp
    for i, p in enumerate(pvals):
        if not np.isfinite(p) or p <= 0:
            val = 0.0
        else:
            lp = -np.log10(p)
            if norm is None:
                val = np.clip((lp - sigbar_min_logp) / denom, 0, 1)
            else:
                try:
                    val = float(norm(lp))
                except TypeError:
                    val = float(norm([lp])[0])
        ax_sig.add_patch(
            plt.Rectangle(
                (0, y[i] - 4),
                1,
                8.0,
                facecolor=cmap(val),
                edgecolor="none",
                alpha=sigbar_alpha,
            )
        )
    # Render text labels
    ax_txt.set(xlim=(0, 1), ylim=(k * 10, 0))
    # c2g is needed for _get_cluster_size
    c2g = getattr(clusters, "cluster_to_labels", None) or {}
    # Format and place per-cluster label text
    for cid, yi, lab, p in zip(cluster_ids, y, labels, pvals):
        n = None
        if "n" in label_fields:
            n = _get_cluster_size(
                int(cid),
                label_map=lab_map,
                cluster_sizes=cluster_sizes,
                cluster_to_labels=c2g,
            )
        pval_value = p if np.isfinite(p) else None
        has_label, stats = collect_label_stats(
            label_fields,
            n_members=n,
            pval=pval_value,
        )
        if has_label:
            if stats:
                stat_tail = "(" + ", ".join(stats) + ")"
                if wrap_text and wrap_width is not None and wrap_width > 0:
                    lines = lab.split("\n") if lab else [""]
                    if len(lines[-1]) + 1 + len(stat_tail) <= wrap_width:
                        lines[-1] = (lines[-1] + " " + stat_tail).strip()
                        txt = "\n".join(lines)
                    else:
                        txt = (lab + "\n" + stat_tail) if lab else stat_tail
                else:
                    txt = f"{lab} {stat_tail}"
            else:
                txt = lab
        else:
            if stats:
                txt = "(" + ", ".join(stats) + ")"
            else:
                txt = ""
        ax_txt.text(
            0.0,
            yi,
            txt,
            va="center",
            ha="left",
            font=font,
            fontsize=fontsize,
            color=label_color,
            alpha=label_alpha,
            fontweight=label_fontweight,
        )
    # Clean axes and display the figure
    _finalize_axes(ax_den, ax_sig, ax_txt)
