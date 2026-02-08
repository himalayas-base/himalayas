"""
tests/test_dendrogram_condensed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import matplotlib.pyplot as plt
import pytest

from himalayas.core.clustering import Clusters
from himalayas.core.results import Results
from himalayas.plot import plot_dendrogram_condensed


@pytest.mark.api
def test_dendrogram_condensed_missing_term_column_raises(toy_results):
    """
    Ensures missing required term column raises a KeyError.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        KeyError: If required label-generation columns are missing.
    """
    bad_df = toy_results.df.drop(columns=["term"])
    results = Results(
        bad_df,
        method=toy_results.method,
        matrix=toy_results.matrix,
        clusters=toy_results.clusters,
        layout=toy_results.cluster_layout(),
        parent=toy_results,
    )
    with pytest.raises(KeyError):
        plot_dendrogram_condensed(results)


@pytest.mark.api
def test_dendrogram_condensed_invalid_label_fields_raises(toy_results):
    """
    Ensures invalid label_fields values raise a ValueError.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        ValueError: If label_fields contains unsupported values.
    """
    with pytest.raises(ValueError):
        plot_dendrogram_condensed(
            toy_results,
            label_fields=("label", "bad"),
        )


@pytest.mark.api
def test_dendrogram_condensed_bad_label_overrides_type_raises(
    toy_results
):
    """
    Ensures label_overrides must be a dict when provided.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        TypeError: If label_overrides is not a dict.
    """
    with pytest.raises(TypeError):
        plot_dendrogram_condensed(
            toy_results,
            label_overrides=["not-a-dict"],
        )


@pytest.mark.api
def test_dendrogram_condensed_smoke(toy_results):
    """
    Ensures condensed dendrogram renders without error for valid inputs.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    # Suppress GUI rendering during tests
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plot_dendrogram_condensed(toy_results)
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_dendrogram_condensed_label_fields_respect_np_order(toy_results):
    """
    Ensures condensed dendrogram labels keep label first while respecting n/p order
    from label_fields and emitting a single stats block.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    # Suppress GUI rendering during tests
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plot_dendrogram_condensed(
            toy_results,
            label_fields=("label", "p", "n"),
        )
        fig = plt.gcf()
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        label_texts = [t for t in texts if "$p$=" in t and "n=" in t]
        assert label_texts, "Expected cluster label text to be rendered."
        for txt in label_texts:
            assert " (" in txt
            assert txt.strip().endswith(")")
            assert "$p$=" in txt
            assert "n=" in txt
            assert txt.find("$p$=") < txt.find("n=")
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_dendrogram_condensed_missing_linkage_raises(toy_results):
    """
    Ensures missing master linkage raises an AttributeError.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        AttributeError: If master linkage is missing from results.clusters.
    """
    class _DummyClusters:
        linkage_matrix = None
        threshold = 0.0

    results = Results(
        toy_results.df,
        method=toy_results.method,
        matrix=toy_results.matrix,
        clusters=_DummyClusters(),
        layout=toy_results.cluster_layout(),
        parent=toy_results,
    )
    with pytest.raises(AttributeError):
        plot_dendrogram_condensed(results)


@pytest.mark.api
def test_dendrogram_condensed_no_clusters_raises(toy_results):
    """
    Ensures empty cluster layout raises a ValueError.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        ValueError: If no clusters are present in the layout.
    """
    class _EmptyLayout:
        cluster_spans = []

    results = Results(
        toy_results.df,
        method=toy_results.method,
        matrix=toy_results.matrix,
        clusters=toy_results.clusters,
        layout=_EmptyLayout(),
        parent=toy_results,
    )
    with pytest.raises(ValueError):
        plot_dendrogram_condensed(results)


@pytest.mark.api
def test_dendrogram_condensed_unmapped_clusters_raises(toy_results):
    """
    Ensures unmapped cluster ids raise a ValueError.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        ValueError: If cluster ids cannot be mapped from master leaf order.
    """
    bad_labels = [f"x{i}" for i in range(len(toy_results.matrix.labels))]
    bad_clusters = Clusters(
        toy_results.clusters.linkage_matrix,
        labels=bad_labels,
        threshold=toy_results.clusters.threshold,
    )
    results = Results(
        toy_results.df,
        method=toy_results.method,
        matrix=toy_results.matrix,
        clusters=bad_clusters,
        layout=toy_results.cluster_layout(),
        parent=toy_results,
    )
    with pytest.raises(ValueError):
        plot_dendrogram_condensed(results)


@pytest.mark.api
def test_dendrogram_condensed_summary_max_words_controls_generation(toy_results):
    """
    Ensures summary_max_words controls compressed label generation.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        plot_dendrogram_condensed(
            toy_results,
            label_mode="compressed",
            summary_max_words=1,
            label_fields=("label",),
            wrap_text=False,
        )
        fig = plt.gcf()
        texts = [t.get_text().strip() for ax in fig.axes for t in ax.texts]
        cluster_texts = [t for t in texts if t and t != "—"]
        assert cluster_texts, "Expected generated cluster labels to be rendered."
        for txt in cluster_texts:
            assert len(txt.replace("\n", " ").split()) <= 1
    finally:
        plt.show = plt_show


@pytest.mark.api
def test_dendrogram_condensed_max_words_controls_display(toy_results):
    """
    Ensures max_words truncates rendered override labels.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        cid = int(toy_results.cluster_layout().cluster_spans[0][0])
        plot_dendrogram_condensed(
            toy_results,
            label_overrides={cid: "Alpha Beta Gamma"},
            label_fields=("label",),
            max_words=1,
            wrap_text=False,
        )
        fig = plt.gcf()
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        assert any(t.strip() == "Alpha" for t in texts)
        assert not any("Alpha Beta" in t for t in texts)
    finally:
        plt.show = plt_show
