"""
tests/test_enrichment
~~~~~~~~~~~~~~~~~~~~~
"""

import pandas as pd
import pytest

from himalayas import Analysis, Annotations, Matrix
from himalayas.core import run_cluster_hypergeom
from himalayas.core.clustering import cluster


@pytest.mark.api
def test_run_cluster_hypergeom_basic():
    """
    Ensures hypergeometric enrichment returns expected counts.
    """
    # Create a minimal matrix/annotation setup
    df = pd.DataFrame(
        [[0.0], [1.0], [2.0]],
        index=["a", "b", "c"],
        columns=["x"],
    )
    matrix = Matrix(df)
    clusters = cluster(matrix, linkage_threshold=100.0)
    annotations = Annotations({"t1": ["a", "b"], "t2": ["c"]}, matrix)
    results = run_cluster_hypergeom(matrix, clusters, annotations)

    assert set(results.df["term"].tolist()) == {"t1", "t2"}
    row_t1 = results.df[results.df["term"] == "t1"].iloc[0]
    assert row_t1["K"] == 2
    assert row_t1["k"] == 2
    assert row_t1["n"] == 3
    assert row_t1["N"] == 3


@pytest.mark.api
def test_run_cluster_hypergeom_empty_rows_schema(toy_matrix):
    """
    Ensures empty enrichments return the expected schema.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.
    """
    clusters = cluster(toy_matrix, linkage_threshold=1.0)
    annotations = Annotations({"t1": ["a", "b", "c", "d"]}, toy_matrix)
    results = run_cluster_hypergeom(
        toy_matrix,
        clusters,
        annotations,
        min_overlap=3,
    )

    assert results.df.empty
    assert results.df.columns.tolist() == ["cluster", "term", "k", "K", "n", "N", "pval"]


@pytest.mark.api
def test_run_cluster_hypergeom_background_mismatch_raises(toy_matrix):
    """
    Ensures background matrix must contain all analysis labels.

    Args:
        toy_matrix (Matrix): Toy matrix fixture.

    Raises:
        ValueError: If background is missing labels from the analysis matrix.
    """
    # Construct a background missing analysis labels to trigger universe validation.
    clusters = cluster(toy_matrix, linkage_threshold=1.0)
    annotations = Annotations({"t1": ["a", "b"]}, toy_matrix)
    background_df = toy_matrix.df.loc[["a", "b"]]
    background = Matrix(background_df)
    with pytest.raises(ValueError):
        run_cluster_hypergeom(
            toy_matrix,
            clusters,
            annotations,
            background=background,
        )


@pytest.mark.api
def test_subset_rebind_respects_global_and_local_universe_counts():
    """
    Ensures subset enrichment keeps global counts with background and local counts
    without background when annotations were rebound to the subset matrix.
    """
    df = pd.DataFrame(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [10.1, 10.1],
        ],
        index=["a", "b", "c", "d", "e", "f", "g", "h"],
        columns=["x", "y"],
    )
    full = Matrix(df)
    annotations = Annotations({"termX": ["a", "b", "e", "f"]}, full)

    parent = Analysis(full, annotations).cluster(linkage_threshold=1.0).enrich().finalize()
    cid = int(parent.clusters.label_to_cluster["a"])
    parent_row = parent.results.df.query("cluster == @cid and term == 'termX'").iloc[0]

    subset_matrix = parent.results.subset(cluster=cid).matrix
    subset_annotations = annotations.rebind(subset_matrix)
    subset = (
        Analysis(subset_matrix, subset_annotations)
        .cluster(linkage_threshold=10.0)
        .enrich(background=full)
        .finalize()
    )
    subset_row = subset.results.df[subset.results.df["term"] == "termX"].iloc[0]

    assert int(subset_row["k"]) == int(parent_row["k"])
    assert int(subset_row["K"]) == int(parent_row["K"])
    assert int(subset_row["n"]) == int(parent_row["n"])
    assert int(subset_row["N"]) == int(parent_row["N"])
    assert float(subset_row["pval"]) == pytest.approx(float(parent_row["pval"]))
    assert float(subset_row["qval"]) == pytest.approx(float(parent_row["qval"]))
    subset_local = (
        Analysis(subset_matrix, subset_annotations).cluster(linkage_threshold=10.0).enrich()
    )
    subset_local_row = subset_local.results.df[subset_local.results.df["term"] == "termX"].iloc[0]
    assert int(subset_local_row["K"]) == 2
    assert int(subset_local_row["N"]) == 4


@pytest.mark.api
def test_subset_background_must_cover_all_subset_labels():
    """
    Ensures subset + background enrichment validates that the background covers
    the full subset matrix label universe.
    """
    df = pd.DataFrame(
        [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]],
        index=["a", "b", "c", "d"],
        columns=["x", "y"],
    )
    full = Matrix(df)
    annotations = Annotations({"termX": ["a", "b", "c"]}, full)

    subset_matrix = Matrix(full.df.loc[["a", "b"]])
    subset_annotations = annotations.rebind(subset_matrix)
    background_missing_label = Matrix(full.df.loc[["a", "c", "d"]])

    with pytest.raises(ValueError):
        (
            Analysis(subset_matrix, subset_annotations)
            .cluster(linkage_threshold=1.0)
            .enrich(background=background_missing_label)
        )
