"""
tests/test_contracted_dendrogram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pandas as pd
import pytest

from himalayas.plot import plot_term_hierarchy_contracted


@pytest.mark.api
def test_contracted_dendrogram_missing_columns_raises(toy_results):
    """
    Ensures missing columns in cluster_labels raises a ValueError.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        ValueError: If required cluster label columns are missing.
    """
    cluster_labels = pd.DataFrame({"cluster": [1], "label": ["X"]})
    with pytest.raises(ValueError):
        plot_term_hierarchy_contracted(toy_results, cluster_labels)


@pytest.mark.api
def test_contracted_dendrogram_invalid_label_fields_raises(toy_results, toy_cluster_labels):
    """
    Ensures invalid label_fields values raise a ValueError.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
        toy_cluster_labels (pd.DataFrame): Cluster labels fixture.

    Raises:
        ValueError: If label_fields contains unsupported values.
    """
    with pytest.raises(ValueError):
        plot_term_hierarchy_contracted(
            toy_results,
            toy_cluster_labels,
            label_fields=("label", "bad"),
        )


@pytest.mark.api
def test_contracted_dendrogram_bad_label_overrides_type_raises(
    toy_results, toy_cluster_labels
):
    """
    Ensures label_overrides must be a dict when provided.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
        toy_cluster_labels (pd.DataFrame): Cluster labels fixture.

    Raises:
        TypeError: If label_overrides is not a dict.
    """
    with pytest.raises(TypeError):
        plot_term_hierarchy_contracted(
            toy_results,
            toy_cluster_labels,
            label_overrides=["not-a-dict"],
        )
