"""
tests/conftest
~~~~~~~~~~~~~~
"""

import pandas as pd
import pytest

from himalayas import Analysis, Annotations, Matrix


@pytest.fixture(scope="session")
def toy_df():
    return pd.DataFrame(
        [[0.0, 0.1], [0.0, 0.2], [5.0, 5.1], [5.0, 5.2]],
        index=["a", "b", "c", "d"],
        columns=["f1", "f2"],
    )


@pytest.fixture(scope="session")
def toy_matrix(toy_df):
    return Matrix(toy_df)


@pytest.fixture(scope="session")
def toy_annotations(toy_matrix):
    return Annotations({"t1": ["a", "b"], "t2": ["c", "d"]}, toy_matrix)


@pytest.fixture(scope="session")
def toy_analysis(toy_matrix, toy_annotations):
    return (
        Analysis(toy_matrix, toy_annotations)
        .cluster(linkage_threshold=1.0)
        .enrich()
        .finalize(add_qvalues=False, col_cluster=False)
    )


@pytest.fixture(scope="session")
def toy_results(toy_analysis):
    return toy_analysis.results


@pytest.fixture(scope="session")
def toy_cluster_labels(toy_results):
    cids = [int(c) for c in toy_results.clusters.unique_clusters]
    return pd.DataFrame(
        {
            "cluster": cids,
            "label": [f"Cluster {c}" for c in cids],
            "pval": [1.0] * len(cids),
        }
    )
