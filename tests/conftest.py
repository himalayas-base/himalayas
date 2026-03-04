"""
tests/conftest
~~~~~~~~~~~~~~
"""

import pandas as pd
import pytest

from himalayas import Analysis, Annotations, Matrix


def use_agg_backend():
    """
    Configures Matplotlib to use the Agg backend and returns pyplot.

    Returns:
        Any: Matplotlib pyplot module with Agg backend active.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def extract_figure_text(fig, *, strip=False, nonempty=False):
    """
    Collects rendered text strings from all text artists in a Matplotlib figure.

    Args:
        fig (matplotlib.figure.Figure): Figure to inspect.

    Kwargs:
        strip (bool): Whether to strip leading/trailing whitespace. Defaults to False.
        nonempty (bool): Whether to drop empty strings after optional stripping. Defaults to False.

    Returns:
        list[str]: Text content from all axes text artists.
    """
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    if strip:
        texts = [txt.strip() for txt in texts]
    if nonempty:
        texts = [txt for txt in texts if txt]
    return texts


@pytest.fixture(scope="session")
def toy_df():
    """
    Returns a small toy DataFrame for test fixtures.

    Returns:
        pd.DataFrame: Toy matrix DataFrame.
    """
    return pd.DataFrame(
        [[0.0, 0.1], [0.0, 0.2], [5.0, 5.1], [5.0, 5.2]],
        index=["a", "b", "c", "d"],
        columns=["f1", "f2"],
    )


@pytest.fixture(scope="session")
def toy_matrix(toy_df):
    """
    Returns a Matrix built from the toy DataFrame.

    Args:
        toy_df (pd.DataFrame): Toy input DataFrame.

    Returns:
        Matrix: Matrix wrapper for the toy data.
    """
    return Matrix(toy_df)


@pytest.fixture(scope="session")
def toy_annotations(toy_matrix):
    """
    Returns toy annotations aligned to the toy matrix.

    Args:
        toy_matrix (Matrix): Matrix providing label universe.

    Returns:
        Annotations: Annotations for two toy terms.
    """
    return Annotations({"t1": ["a", "b"], "t2": ["c", "d"]}, toy_matrix)


@pytest.fixture(scope="session")
def toy_analysis(toy_matrix, toy_annotations):
    """
    Returns a completed Analysis pipeline for toy inputs.

    Args:
        toy_matrix (Matrix): Matrix for clustering.
        toy_annotations (Annotations): Annotations for enrichment.

    Returns:
        Analysis: Analysis instance with clustering, enrichment, and layout.
    """
    return (
        Analysis(toy_matrix, toy_annotations)
        .cluster(linkage_threshold=1.0)
        .enrich()
        .finalize(col_cluster=False)
    )


@pytest.fixture(scope="session")
def toy_results(toy_analysis):
    """
    Returns Results from the toy analysis pipeline.

    Args:
        toy_analysis (Analysis): Analysis instance with results.

    Returns:
        Results: Results object from the toy analysis.
    """
    return toy_analysis.results


@pytest.fixture(scope="session")
def toy_cluster_labels(toy_results):
    """
    Returns a simple cluster label DataFrame for plotting tests.

    Args:
        toy_results (Results): Results with clusters attached.

    Returns:
        pd.DataFrame: Cluster label table.
    """
    cids = [int(c) for c in toy_results.clusters.unique_clusters]
    return pd.DataFrame(
        {
            "cluster": cids,
            "label": [f"Cluster {c}" for c in cids],
            "pval": [1.0] * len(cids),
        }
    )
