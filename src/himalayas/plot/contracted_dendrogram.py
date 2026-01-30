"""
himalayas/plot/contracted_dendrogram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list


# ============================================================================
# PRIVATE HELPERS
# ============================================================================


def _contract_linkage_to_clusters(
    Z_master: np.ndarray,
    gene_to_cluster: dict[Any, int],
    cluster_ids: list[int],
) -> np.ndarray:
    """
    Contract master linkage matrix to cluster-level dendrogram.

    Preserves master dendrogram heights and leaf order by treating each
    cluster as a single leaf in the contracted tree.
    """
    cluster_index = {cid: i for i, cid in enumerate(cluster_ids)}

    n_master = Z_master.shape[0] + 1
    leaf_groups: list[Optional[int]] = []
    for i in range(n_master):
        cid = gene_to_cluster.get(i, None)
        if cid is None or cid not in cluster_index:
            leaf_groups.append(None)
            continue
        leaf_groups.append(cluster_index[cid])

    node_groups: Dict[int, set[int]] = {}
    for i, grp in enumerate(leaf_groups):
        node_groups[i] = set() if grp is None else {int(grp)}

    Zc_rows: list[list[float]] = []
    rep_to_id = {frozenset({i}): i for i in range(len(cluster_ids))}
    next_id = len(cluster_ids)

    for t in range(Z_master.shape[0]):
        a = int(Z_master[t, 0])
        b = int(Z_master[t, 1])
        h = float(Z_master[t, 2])

        Ga = node_groups.get(a, set())
        Gb = node_groups.get(b, set())
        G = Ga | Gb
        node_groups[n_master + t] = G

        if len(G) <= 1 or Ga == Gb:
            continue

        ra = frozenset(Ga)
        rb = frozenset(Gb)
        rg = frozenset(G)

        if ra not in rep_to_id:
            rep_to_id[ra] = next_id
            next_id += 1
        if rb not in rep_to_id:
            rep_to_id[rb] = next_id
            next_id += 1

        ida = rep_to_id[ra]
        idb = rep_to_id[rb]

        if rg in rep_to_id:
            continue

        rep_to_id[rg] = next_id
        Zc_rows.append([ida, idb, h, float(len(G))])
        next_id += 1

    expected = len(cluster_ids) - 1
    if len(Zc_rows) != expected:
        raise ValueError(
            "Contracted linkage is incomplete (expected "
            f"{expected} merges, got {len(Zc_rows)}). "
            "This usually means clusters are not all connected under the provided "
            "master linkage."
        )

    return np.asarray(Zc_rows, dtype=float)


def _format_cluster_label(
    raw_label: str,
    *,
    omit_words: Optional[list[str]] = None,
    max_words: Optional[int] = None,
    overflow: str = "ellipsis",
    wrap_text: bool = False,
    wrap_width: Optional[int] = None,
) -> str:
    """
    Apply text policy to cluster label.

    Order: omit words -> truncate -> wrap.
    """
    label = str(raw_label)

    if omit_words:
        omit = {w.lower() for w in omit_words}
        words = [w for w in label.split() if w.lower() not in omit]
        label = " ".join(words) if words else label

    if max_words is not None:
        words = label.split()
        if len(words) > max_words:
            if overflow == "ellipsis" and max_words > 0:
                label = " ".join(words[: max_words - 1]) + " …"
            else:
                label = " ".join(words[:max_words])

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
    label_map: dict[int, tuple],
    cluster_sizes: Optional[dict[int, int]],
    cluster_to_labels: dict[int, list],
) -> Optional[int]:
    """Resolve cluster size from available sources."""
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


# ============================================================================
# PUBLIC API
# ============================================================================


def plot_term_hierarchy_contracted(
    results: Any,
    cluster_labels: pd.DataFrame,
    *,
    figsize: Sequence[float] = (10, 10),
    sigbar_cmap="YlOrBr",
    sigbar_min_logp=2.0,
    sigbar_max_logp=12.0,
    sigbar_width=0.06,
    sigbar_alpha=0.9,
    font="Helvetica",
    fontsize=9,
    max_words=None,
    wrap_text: bool = False,
    wrap_width: Optional[int] = None,
    overflow: str = "ellipsis",
    omit_words=None,
    label_fields: Sequence[str] = ("label", "n", "p"),
    label_overrides: Optional[Dict[int, str]] = None,
    label_color="black",
    label_alpha=1.0,
    label_fontweight="normal",
    dendrogram_color="black",
    dendrogram_lw=1.0,
    label_left_pad=0.02,
    background_color=None,
) -> None:
    """
    Cluster-level dendrogram using contracted master linkage.

    Leaf order and branch heights are preserved from the master dendrogram
    by contracting the master linkage matrix at the cluster level.
    """
    # === VALIDATION ===
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
    clusters = getattr(results, "clusters", None)
    Z_master = getattr(clusters, "linkage_matrix", None) if clusters is not None else None
    if Z_master is None:
        raise AttributeError(
            "results.clusters.linkage_matrix is required to preserve the master dendrogram "
            "order/heights"
        )
    # === DATA PREP ===
    row_labels = results.matrix.labels
    c2g = getattr(clusters, "cluster_to_labels", None) or {}
    gene_to_cluster = {g: int(cid) for cid, genes in c2g.items() for g in genes}
    ordered_cluster_ids, seen = [], set()
    for i in leaves_list(Z_master):
        cid = gene_to_cluster.get(row_labels[int(i)], None)
        if cid is None or cid in seen:
            continue
        ordered_cluster_ids.append(cid)
        seen.add(cid)
    if not ordered_cluster_ids:
        raise ValueError("No cluster ids could be mapped from master leaf order")
    cluster_ids = ordered_cluster_ids
    k = len(cluster_ids)
    y = np.arange(k) * 10.0 + 5.0
    lab_map: Dict[int, Tuple[str, float, Optional[float]]] = {
        int(r["cluster"]): (str(r["label"]), float(r["pval"]), r.get("n", None))
        for _, r in cluster_labels.iterrows()
    }
    cluster_sizes = getattr(clusters, "cluster_sizes", None) if clusters is not None else None
    cluster_sizes = dict(cluster_sizes) if cluster_sizes is not None else None
    label_overrides = label_overrides or {}
    labels, pvals = [], []
    for cid in cluster_ids:
        if cid not in lab_map:
            labels.append("—")
            pvals.append(np.nan)
            continue
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
    pvals = np.asarray(pvals, float)
    n_master = Z_master.shape[0] + 1
    gene_to_cluster_index = {
        i: gene_to_cluster[row_labels[int(i)]]
        for i in range(n_master)
        if row_labels[int(i)] in gene_to_cluster
    }
    # === LINKAGE CONTRACTION ===
    Zc = _contract_linkage_to_clusters(Z_master, gene_to_cluster_index, cluster_ids)
    # === DENDROGRAM COMPUTATION ===
    d = dendrogram(Zc, orientation="left", no_labels=True, no_plot=True)
    # === FIGURE SETUP ===
    fig = plt.figure(figsize=figsize)
    ax_den = fig.add_axes([0.05, 0.05, 0.60, 0.90], frameon=False)
    ax_sig = fig.add_axes([0.66, 0.05, sigbar_width, 0.90], frameon=False)
    txt_x0 = 0.66 + sigbar_width + float(label_left_pad)
    ax_txt = fig.add_axes(
        [txt_x0, 0.05, 0.33 - sigbar_width - float(label_left_pad), 0.90],
        frameon=False,
    )
    axes = (ax_den, ax_sig, ax_txt)
    if background_color is not None:
        fig.patch.set_facecolor(background_color)
        for ax in axes:
            ax.set_facecolor(background_color)
    # === RENDERING: DENDROGRAM ===
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
    # === RENDERING: SIGNIFICANCE BAR ===
    cmap = plt.get_cmap(sigbar_cmap)
    ax_sig.set(xlim=(0, 1), ylim=(k * 10, 0))
    denom = sigbar_max_logp - sigbar_min_logp
    for i, p in enumerate(pvals):
        if not np.isfinite(p) or p <= 0:
            val = 0.0
        else:
            lp = -np.log10(p)
            val = np.clip((lp - sigbar_min_logp) / denom, 0, 1)
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
    # === RENDERING: LABELS ===
    ax_txt.set(xlim=(0, 1), ylim=(k * 10, 0))
    for cid, yi, lab, p in zip(cluster_ids, y, labels, pvals):
        parts = []
        if "label" in label_fields:
            parts.append(lab)
        if "n" in label_fields:
            n = _get_cluster_size(
                int(cid),
                label_map=lab_map,
                cluster_sizes=cluster_sizes,
                cluster_to_labels=c2g,
            )
            if n is not None:
                parts.append(f"n={n}")
        if "p" in label_fields and np.isfinite(p):
            ptxt = f"p={p:.1e}" if p < 1e-3 else f"p={p:.3f}"
            parts.append(ptxt)

        if not parts:
            txt = ""
        elif len(parts) == 1:
            txt = parts[0]
        else:
            label_head = parts[0]
            stat_tail = "(" + ", ".join(parts[1:]) + ")"
            if wrap_text and wrap_width is not None and wrap_width > 0:
                lines = label_head.split("\n") if label_head else [""]
                if len(lines[-1]) + 1 + len(stat_tail) <= wrap_width:
                    lines[-1] = (lines[-1] + " " + stat_tail).strip()
                    txt = "\n".join(lines)
                else:
                    txt = (label_head + "\n" + stat_tail) if label_head else stat_tail
            else:
                txt = f"{label_head} {stat_tail}"

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
    for ax in axes:
        ax.set(xticks=[], yticks=[])
        for sp in ax.spines.values():
            sp.set_visible(False)
    plt.show()
