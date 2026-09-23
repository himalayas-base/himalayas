"""
himalayas/plot/renderers/_cluster_label_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from ._cluster_label_types import ClusterLabelStats


def _parse_label_overrides(
    overrides: Optional[Dict[int, str]] = None,
) -> Dict[int, str]:
    """
    Normalizes and validates per-cluster label overrides.

    Args:
        overrides (Dict[int, str] | None): Mapping cluster_id -> label string.
            Defaults to None.

    Returns:
        Dict[int, str]: Normalized override map keyed by cluster id.

    Raises:
        TypeError: If overrides or entries have invalid types.
    """
    if overrides is None:
        return {}
    # Validation
    if not isinstance(overrides, dict):
        raise TypeError("overrides must be a dict mapping cluster_id -> label string")

    # Normalize cluster ids and validate label strings.
    # Empty strings are allowed so callers can intentionally suppress label text.
    # for selected clusters without affecting bar tracks.
    override_map: Dict[int, str] = {}
    for key, value in overrides.items():
        cid = int(key)
        if not isinstance(value, str):
            raise TypeError("override values must be strings")
        override_map[cid] = value

    return override_map


def _build_label_map(
    df: pd.DataFrame,
    override_map: Dict[int, str],
) -> Dict[int, ClusterLabelStats]:
    """
    Resolves final label and p-value per cluster. Combines base labels from the DataFrame
    with any validated overrides.

    Args:
        df (pd.DataFrame): Cluster label table with 'cluster', 'label', and optional
            'pval', 'qval', 'score', and 'fe'.
        override_map (Dict[int, str]): Normalized overrides keyed by cluster id.

    Returns:
        Dict[int, ClusterLabelStats]: Mapping cluster id to (label, pval, qval, score, fe).

    Raises:
        ValueError: If overrides reference unknown cluster ids.
    """
    # Build base label map; overrides are label-only and must not alter stats.
    label_map: Dict[int, ClusterLabelStats] = {}
    for _, row in df.iterrows():
        cid = int(row["cluster"])
        base_label = str(row["label"])
        base_pval = row.get("pval", None)
        base_qval = row.get("qval", None)
        base_fe = row.get("fe", None)
        if "score" in row:
            base_score = row.get("score", None)
        elif "pval" in row:
            base_score = base_pval
        elif "qval" in row:
            base_score = base_qval
        else:
            base_score = None
        if cid in override_map:
            label = override_map[cid]
        else:
            label = base_label
        label_map[cid] = (label, base_pval, base_qval, base_score, base_fe)
    # Reject overrides that do not match any cluster id.
    if override_map:
        unknown = set(override_map) - set(label_map)
        if unknown:
            raise ValueError(
                "overrides contain cluster ids not present in cluster_labels: " f"{sorted(unknown)}"
            )

    return label_map
