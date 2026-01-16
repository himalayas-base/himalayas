import pandas as pd

from himalayas import Matrix, cluster
from himalayas.core import Annotations, run_cluster_hypergeom


def test_run_cluster_hypergeom_basic():
    df = pd.DataFrame(
        [[0.0], [1.0], [2.0]],
        index=["a", "b", "c"],
        columns=["x"],
    )
    matrix = Matrix(df, matrix_semantics="distance")
    clusters = cluster(matrix, linkage_threshold=100.0)
    annotations = Annotations({"t1": ["a", "b"], "t2": ["c"]}, matrix)

    results = run_cluster_hypergeom(matrix, clusters, annotations)

    assert set(results.df["term"].tolist()) == {"t1", "t2"}
    row_t1 = results.df[results.df["term"] == "t1"].iloc[0]
    assert row_t1["K"] == 2
    assert row_t1["k"] == 2
    assert row_t1["n"] == 3
    assert row_t1["N"] == 3
