"""
himalayas/plot/renderers/_label_format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, List


def collect_label_stats(
    label_fields: Sequence[str],
    *,
    n_members: Optional[int] = None,
    pval: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    """
    Collects ordered label stats based on label_fields.

    Args:
        label_fields (Sequence[str]): Fields to include in labels.
        n_members (Optional[int]): Cluster size for "n". Defaults to None.
        pval (Optional[float]): P-value for "p". Defaults to None.

    Returns:
        Tuple[bool, List[str]]: (has_label, ordered_stats_list).
    """
    has_label = "label" in label_fields
    stats: List[str] = []
    for field in label_fields:
        if field == "label":
            continue
        if field == "n" and n_members is not None:
            stats.append(f"n={n_members}")
        elif field == "p" and pval is not None:
            stats.append(rf"$p$={pval:.2e}")
    return has_label, stats
