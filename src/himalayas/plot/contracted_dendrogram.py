# ============================================================
# HiMaLAYAS — Cluster-Level Dendrogram (Contracted Master Linkage)
# ============================================================
#
# GOAL:
#   Reconstruct a cluster-level hierarchy that closely mimics
#   the master dendrogram in BOTH ordering and relative heights,
#   without fragile assumptions.
#
# STRATEGY:
#   - Contract the MASTER linkage matrix at the chosen cluster cut
#   - Treat clusters as leaves; preserve master topology and leaf order
#
# This is simple, stable, and honest.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from typing import Any, Dict, Optional, Sequence


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

    This intentionally trades exact topology for stability
    and interpretability.

    Label verbosity is controlled at plot time via `label_fields`.
    """

    # ------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------
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

    layout = results.cluster_layout()
    spans = list(layout.cluster_spans)
    if not spans:
        raise ValueError("No clusters found")

    # ------------------------------------------------------------
    # Master linkage matrix (authoritative ordering + heights)
    # ------------------------------------------------------------
    Z_master = getattr(getattr(results, "clusters", None), "linkage_matrix", None)
    if Z_master is None:
        raise AttributeError(
            "results.clusters.linkage_matrix is required to preserve the master dendrogram order/heights"
        )

    # ------------------------------------------------------------
    # Build a leaf->cluster assignment using the *existing* clusters
    # ------------------------------------------------------------
    row_labels = results.matrix.labels

    # gene -> cluster id
    gene_to_cluster = {}
    c2g = getattr(getattr(results, "clusters", None), "cluster_to_labels", None) or {}
    for cid, genes in c2g.items():
        for g in genes:
            gene_to_cluster[g] = int(cid)

    # leaves_list gives the master dendrogram leaf order by row index
    from scipy.cluster.hierarchy import leaves_list

    master_leaf_order = leaves_list(Z_master)

    # ordered unique cluster ids as they appear in the master leaf order
    ordered_cluster_ids = []
    seen = set()
    for i in master_leaf_order:
        g = row_labels[int(i)]
        cid = gene_to_cluster.get(g, None)
        if cid is None:
            continue
        if cid not in seen:
            ordered_cluster_ids.append(cid)
            seen.add(cid)

    if not ordered_cluster_ids:
        raise ValueError("No cluster ids could be mapped from master leaf order")

    cluster_ids = ordered_cluster_ids
    k = len(cluster_ids)
    y = np.arange(k) * 10.0 + 5.0

    # ------------------------------------------------------------
    # Build label map
    # ------------------------------------------------------------
    lab_map = {}
    for _, r in cluster_labels.iterrows():
        lab_map[int(r["cluster"])] = (
            str(r["label"]),
            float(r["pval"]),
            r.get("n", None),
        )

    # ------------------------------------------------------------
    # Cluster size fallback
    # ------------------------------------------------------------
    # Prefer an explicit 'n' column in cluster_labels if present, otherwise
    # fall back to results.clusters.cluster_sizes or len(cluster_to_labels[cid]).
    cluster_sizes = None
    if (
        hasattr(results, "clusters")
        and getattr(results.clusters, "cluster_sizes", None) is not None
    ):
        cluster_sizes = dict(results.clusters.cluster_sizes)

    def get_cluster_size(cid: int) -> Optional[int]:
        # 1) explicit in label map
        if cid in lab_map:
            n0 = lab_map[cid][2]
            if n0 is not None and np.isfinite(n0):
                return int(n0)
        # 2) results.clusters.cluster_sizes
        if cluster_sizes is not None and cid in cluster_sizes:
            return int(cluster_sizes[cid])
        # 3) len(cluster_to_labels)
        genes = c2g.get(cid, None)
        if genes is not None:
            return int(len(genes))
        return None

    labels, pvals = [], []
    for cid in cluster_ids:
        if cid not in lab_map:
            labels.append("—")
            pvals.append(np.nan)
            continue

        lab, p, _ = lab_map[cid]

        if label_overrides is not None and cid in label_overrides:
            lab = str(label_overrides[cid])

        if omit_words:
            omit = {w.lower() for w in omit_words}
            words = [w for w in lab.split() if w.lower() not in omit]
            lab = " ".join(words) if words else lab

        # Centralized Plotter-style text policy: word truncation then visual wrapping
        if max_words is not None:
            words = lab.split()
            if len(words) > max_words:
                if overflow == "ellipsis" and max_words > 0:
                    lab = " ".join(words[: max_words - 1]) + " …"
                else:
                    lab = " ".join(words[:max_words])

        if wrap_text and wrap_width is not None and wrap_width > 0:
            # Wrap by inserting newlines at appropriate spaces without breaking words
            wrapped_lines = []
            line = ""
            for word in lab.split():
                if len(line) + len(word) + (1 if line else 0) <= wrap_width:
                    line = f"{line} {word}".strip()
                else:
                    if line:
                        wrapped_lines.append(line)
                    line = word
            if line:
                wrapped_lines.append(line)
            lab = "\n".join(wrapped_lines)

        labels.append(lab)
        pvals.append(p)

    pvals = np.asarray(pvals, float)

    # ------------------------------------------------------------
    # Contract the master dendrogram so clusters become leaves
    # ------------------------------------------------------------
    cluster_index = {cid: i for i, cid in enumerate(cluster_ids)}

    # Map each master leaf (row index) -> contracted leaf (cluster index)
    n_master = Z_master.shape[0] + 1
    leaf_groups = []
    for i in range(n_master):
        g = row_labels[int(i)]
        cid = gene_to_cluster.get(g, None)
        if cid is None:
            leaf_groups.append(None)
            continue
        if cid not in cluster_index:
            leaf_groups.append(None)
            continue
        leaf_groups.append(cluster_index[cid])

    # For each master node id (0..n_master-1 leaves, n_master.. internal), track which contracted leaves it contains
    node_groups = {}
    for i, grp in enumerate(leaf_groups):
        node_groups[i] = set() if grp is None else {int(grp)}

    # Contracted linkage rows
    Zc_rows = []

    # Representative set -> contracted node id
    rep_to_id = {frozenset({i}): i for i in range(k)}
    next_id = k

    for t in range(Z_master.shape[0]):
        a = int(Z_master[t, 0])
        b = int(Z_master[t, 1])
        h = float(Z_master[t, 2])

        Ga = node_groups.get(a, set())
        Gb = node_groups.get(b, set())
        G = Ga | Gb

        # Always propagate membership upward
        node_groups[n_master + t] = G

        # Ignore merges that don't combine different clusters
        if len(G) <= 1 or Ga == Gb:
            continue

        ra = frozenset(Ga)
        rb = frozenset(Gb)
        rg = frozenset(G)

        # Defensive: if a representative node was never materialized (rare), materialize it now
        if ra not in rep_to_id:
            rep_to_id[ra] = next_id
            next_id += 1
        if rb not in rep_to_id:
            rep_to_id[rb] = next_id
            next_id += 1

        ida = rep_to_id[ra]
        idb = rep_to_id[rb]

        # If we've already merged into this union set, skip
        if rg in rep_to_id:
            continue

        rep_to_id[rg] = next_id
        Zc_rows.append([ida, idb, h, float(len(G))])
        next_id += 1

    if len(Zc_rows) != k - 1:
        raise ValueError(
            f"Contracted linkage is incomplete (expected {k-1} merges, got {len(Zc_rows)}). "
            "This usually means clusters are not all connected under the provided master linkage."
        )

    Zc = np.asarray(Zc_rows, dtype=float)

    # ------------------------------------------------------------
    # Dendrogram from contracted linkage
    # ------------------------------------------------------------
    d = dendrogram(
        Zc,
        orientation="left",
        no_labels=True,
        no_plot=True,
    )

    # ------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    if background_color is not None:
        fig.patch.set_facecolor(background_color)
    ax_den = fig.add_axes([0.05, 0.05, 0.60, 0.90], frameon=False)
    if background_color is not None:
        ax_den.set_facecolor(background_color)
    ax_sig = fig.add_axes([0.66, 0.05, sigbar_width, 0.90], frameon=False)
    if background_color is not None:
        ax_sig.set_facecolor(background_color)
    txt_x0 = 0.66 + sigbar_width + float(label_left_pad)
    ax_txt = fig.add_axes(
        [txt_x0, 0.05, 0.33 - sigbar_width - float(label_left_pad), 0.90], frameon=False
    )
    if background_color is not None:
        ax_txt.set_facecolor(background_color)

    # ------------------------------------------------------------
    # Map dendrogram's internal leaf y positions -> fixed cluster y positions
    # ------------------------------------------------------------
    # Authoritative y positions (already in master-cluster order)
    cluster_y = {i: y[i] for i in range(k)}

    def remap_y(yval: float) -> float:
        # nearest leaf slot (5,15,25,...) -> authoritative y
        slot = int(round((yval - 5.0) / 10.0))
        slot = max(0, min(k - 1, slot))
        leaf_id = d["leaves"][slot]
        return cluster_y[int(leaf_id)]

    # Determine a compact x-scale (avoid oversized tree)
    max_h = max(map(max, d["dcoord"]))
    x_pad = max_h * 0.05

    for xs, ys in zip(d["dcoord"], d["icoord"]):
        ax_den.plot(xs, [remap_y(y) for y in ys], color=dendrogram_color, lw=dendrogram_lw)

    # Proper orientation: tree grows leftward toward leaves
    ax_den.set_xlim(max_h + x_pad, 0.0)
    ax_den.set_ylim(k * 10, 0)

    ax_den.set_xticks([])
    ax_den.set_yticks([])
    for sp in ax_den.spines.values():
        sp.set_visible(False)

    # ------------------------------------------------------------
    # Significance bar
    # ------------------------------------------------------------
    cmap = plt.get_cmap(sigbar_cmap)

    def norm_logp(p: float) -> float:
        if not np.isfinite(p) or p <= 0:
            return 0.0
        lp = -np.log10(p)
        return np.clip((lp - sigbar_min_logp) / (sigbar_max_logp - sigbar_min_logp), 0, 1)

    ax_sig.set_xlim(0, 1)
    ax_sig.set_ylim(k * 10, 0)

    ax_sig.set_xticks([])
    ax_sig.set_yticks([])
    for sp in ax_sig.spines.values():
        sp.set_visible(False)

    for i, p in enumerate(pvals):
        ax_sig.add_patch(
            plt.Rectangle(
                (0, y[i] - 4),
                1,
                8.0,
                facecolor=cmap(norm_logp(p)),
                edgecolor="none",
                alpha=sigbar_alpha,
            )
        )

    # ------------------------------------------------------------
    # Text panel
    # ------------------------------------------------------------
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(k * 10, 0)

    ax_txt.set_xticks([])
    ax_txt.set_yticks([])
    for sp in ax_txt.spines.values():
        sp.set_visible(False)

    for cid, yi, lab, p in zip(cluster_ids, y, labels, pvals):
        parts = []

        if "label" in label_fields:
            parts.append(lab)

        if "n" in label_fields:
            n = get_cluster_size(int(cid))
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

            # Try to append stat_tail to the last line if it fits; otherwise put it on its own line.
            if wrap_text and wrap_width is not None and wrap_width > 0:
                lines = label_head.split("\n") if label_head else [""]
                if len(lines[-1]) + 1 + len(stat_tail) <= wrap_width:
                    lines[-1] = (lines[-1] + " " + stat_tail).strip()
                    txt = "\n".join(lines)
                else:
                    txt = (label_head + "\n" + stat_tail) if label_head else stat_tail
            else:
                # No wrapping policy active: keep stats inline
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

    plt.show()
