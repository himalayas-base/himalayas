"""
himalayas/plot/renderers/_label_format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

import textwrap
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._cluster_label_types import ClusterLabelStats


def compute_equal_slots(n: int, *, pitch: float = 1.0) -> np.ndarray:
    """
    Computes equal-pitch slot centers for `n` items, independent of any other sizing.

    Args:
        n (int): Number of slots.

    Kwargs:
        pitch (float): Spacing between adjacent slot centers. Defaults to 1.0.

    Returns:
        np.ndarray: Slot center positions, i.e. `[pitch/2, 3*pitch/2, ...]`.
    """
    return np.arange(int(n)) * float(pitch) + float(pitch) / 2.0


def collect_label_stats(
    label_fields: Optional[Sequence[str]],
    *,
    n_members: Optional[int] = None,
    pval: Optional[float] = None,
    qval: Optional[float] = None,
    fe: Optional[float] = None,
    force_label: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Collects ordered label stats based on label_fields.

    Args:
        label_fields (Optional[Sequence[str]]): Fields to include in labels.

    Kwargs:
        n_members (Optional[int]): Cluster size for "n". Defaults to None.
        pval (Optional[float]): P-value for "p". Defaults to None.
        qval (Optional[float]): Q-value for "q". Defaults to None.
        fe (Optional[float]): Fold enrichment for "fe". Defaults to None.
        force_label (bool): Force inclusion of label text even when "label" is absent
            from label_fields. Defaults to False.

    Returns:
        Tuple[bool, List[str]]: (has_label, ordered_stats_list).
    """
    if label_fields is None:
        return bool(force_label), []

    has_label = "label" in label_fields or bool(force_label)
    stats: List[str] = []
    for field in label_fields:
        if field == "label":
            continue
        if field == "n" and n_members is not None:
            stats.append(f"n={n_members}")
        elif field == "p" and pval is not None:
            stats.append(rf"$p$={pval:.2e}")
        elif field == "q" and qval is not None:
            stats.append(rf"$q$={qval:.2e}")
        elif field == "fe" and fe is not None:
            stats.append(f"FE={fe:.2f}")
    return has_label, stats


def format_label_prefix(label_prefix: Optional[str], cluster_id: int) -> str:
    """
    Formats a cluster prefix token for display.

    Args:
        label_prefix (Optional[str]): Prefix mode.
        cluster_id (int): Cluster id.

    Returns:
        str: Prefix token including trailing period (for example "3." or "C.").
    """
    if label_prefix == "cid":
        return f"{cluster_id}."
    if label_prefix != "alpha":
        return ""

    # Excel-style alpha indexing: 1 -> A, 26 -> Z, 27 -> AA.
    n = int(cluster_id)
    if n <= 0:
        return f"{cluster_id}."

    chars: List[str] = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        chars.append(chr(ord("A") + rem))
    return "".join(reversed(chars)) + "."


def apply_label_text_policy(
    raw_label: str,
    *,
    omit_words: Optional[Sequence[str]] = None,
    max_words: Optional[int] = None,
    overflow: str = "wrap",
    wrap_text: bool = True,
    wrap_width: Optional[int] = None,
) -> str:
    """
    Applies label text policy in this order: omit words -> truncate -> wrap.

    Args:
        raw_label (str): Source label text.

    Kwargs:
        omit_words (Optional[Sequence[str]]): Words to omit (case-insensitive). Defaults to None.
        max_words (Optional[int]): Maximum words to keep. Defaults to None.
        overflow (str): Truncation mode, one of {"wrap", "ellipsis"}. Defaults to "wrap".
        wrap_text (bool): Whether to wrap text. Defaults to True.
        wrap_width (Optional[int]): Characters per wrapped line. Defaults to None.

    Returns:
        str: Policy-transformed label text.
    """
    label = str(raw_label)

    if omit_words:
        omit = {word.lower() for word in omit_words}
        kept = [word for word in label.split() if word.lower() not in omit]
        if kept:
            label = " ".join(kept)

    if max_words is not None:
        words = label.split()
        if len(words) > max_words:
            if overflow == "ellipsis" and max_words > 0:
                label = " ".join(words[: max_words - 1]) + "…"
            else:
                label = " ".join(words[:max_words])

    if wrap_text and wrap_width is not None and wrap_width > 0:
        label = "\n".join(textwrap.wrap(label, width=wrap_width))

    return label


def compose_label_text(
    label: str,
    *,
    has_label: bool,
    stats: Sequence[str],
    wrap_text: bool = True,
    wrap_width: Optional[int] = None,
) -> str:
    """
    Composes final display text from label + ordered stats.

    Args:
        label (str): Pre-formatted label text.

    Kwargs:
        has_label (bool): Whether the label field is enabled.
        stats (Sequence[str]): Ordered stat fragments.
        wrap_text (bool): Whether wrapping is enabled. Defaults to True.
        wrap_width (Optional[int]): Wrap width in characters. Defaults to None.

    Returns:
        str: Composed display text.
    """
    if not has_label:
        if stats:
            return "(" + ", ".join(stats) + ")"
        return ""

    if not stats:
        return label

    stat_tail = "(" + ", ".join(stats) + ")"
    if wrap_text and wrap_width is not None and wrap_width > 0:
        lines = label.split("\n") if label else [""]
        last = lines[-1]
        sep = 1 if last else 0
        if len(last) + sep + len(stat_tail) <= wrap_width:
            lines[-1] = (last + " " + stat_tail).strip()
            return "\n".join(lines)
        if label:
            return label + "\n" + stat_tail
        return stat_tail

    return f"{label} {stat_tail}".strip()


class ResolvedClusterLabel(NamedTuple):
    """
    Resolved display text for one cluster, plus whether it fell back to a placeholder.
    """

    text: str
    is_placeholder: bool


def resolve_cluster_label_content(
    cluster_id: int,
    label_map: Dict[int, ClusterLabelStats],
    n_members: Optional[int],
    *,
    label_fields: Optional[Sequence[str]],
    label_prefix: Optional[str],
    is_override: bool,
    placeholder_text: str,
    max_words: Optional[int] = None,
    omit_words: Optional[Sequence[str]] = None,
    wrap_text: bool = True,
    wrap_width: Optional[int] = None,
    overflow: str = "wrap",
) -> ResolvedClusterLabel:
    """
    Resolves the display text for one cluster: placeholder vs. real label, field
    selection, prefix, and p/q/fe/n stat formatting. Shared by every renderer that
    shows per-cluster label text so label_fields/label_prefix behave identically
    everywhere.

    Args:
        cluster_id (int): Cluster id.
        label_map (Dict[int, ClusterLabelStats]): Mapping cluster_id -> (label, pval,
            qval, score, fe), as built by _build_label_map.
        n_members (Optional[int]): Cluster size for the "n" field.

    Kwargs:
        label_fields (Optional[Sequence[str]]): Fields to display, e.g.
            ("label", "n", "p"). If None, suppresses base label/stat text.
        label_prefix (Optional[str]): Prefix mode, one of {None, "cid", "alpha"}.
        is_override (bool): Whether this cluster's label came from an explicit override.
        placeholder_text (str): Text to use when the cluster has no label.
        max_words (Optional[int]): Maximum words to keep. Defaults to None.
        omit_words (Optional[Sequence[str]]): Words to omit (case-insensitive). Defaults to None.
        wrap_text (bool): Whether to wrap label text. Defaults to True.
        wrap_width (Optional[int]): Characters per wrapped line. Defaults to None.
        overflow (str): Truncation mode, one of {"wrap", "ellipsis"}. Defaults to "wrap".

    Returns:
        ResolvedClusterLabel: Final display text and whether it is a placeholder.
    """
    if cluster_id not in label_map:
        return ResolvedClusterLabel(placeholder_text, True)

    label, pval, qval, _score, fe = label_map[cluster_id]
    prefix_active = label_prefix in {"cid", "alpha"} and not is_override
    force_label = prefix_active or is_override
    if (label_fields is None or "label" not in label_fields) and not is_override:
        label = ""
    if prefix_active:
        prefix = format_label_prefix(label_prefix, cluster_id)
        label = f"{prefix} {label}" if label else prefix

    pval_value = pval if pval is not None and not pd.isna(pval) else None
    qval_value = qval if qval is not None and not pd.isna(qval) else None
    fe_value = fe if fe is not None and not pd.isna(fe) else None
    has_label, stats = collect_label_stats(
        label_fields,
        n_members=n_members,
        pval=pval_value,
        qval=qval_value,
        fe=fe_value,
        force_label=force_label,
    )

    label_text = apply_label_text_policy(
        label,
        omit_words=omit_words,
        max_words=max_words,
        overflow=overflow,
        wrap_text=wrap_text,
        wrap_width=wrap_width,
    )
    if not has_label and not stats:
        return ResolvedClusterLabel(label_text, False)
    text = compose_label_text(
        label_text,
        has_label=has_label,
        stats=stats,
        wrap_text=wrap_text,
        wrap_width=wrap_width,
    )
    return ResolvedClusterLabel(text, False)
