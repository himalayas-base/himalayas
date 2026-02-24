"""
himalayas/plot/renderers/_label_format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

import textwrap
from typing import List, Optional, Sequence, Tuple


def collect_label_stats(
    label_fields: Optional[Sequence[str]],
    *,
    n_members: Optional[int] = None,
    pval: Optional[float] = None,
    qval: Optional[float] = None,
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
    return has_label, stats


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
