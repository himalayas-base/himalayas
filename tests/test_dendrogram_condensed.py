"""
tests/test_dendrogram_condensed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import matplotlib.pyplot as plt
import pytest

from himalayas.core.clustering import Clusters
from himalayas.core.results import Results
from himalayas.plot import CondensedDendrogramPlot, plot_dendrogram_condensed
from himalayas.plot.condensed_dendrogram import _prepare_cluster_labels


def _use_agg_backend() -> None:
    """
    Configures Matplotlib to use the Agg backend for tests.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)


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
    _use_agg_backend()
    plot = plot_dendrogram_condensed(toy_results)
    assert isinstance(plot, CondensedDendrogramPlot)
    assert plot.fig.axes
    plt.close(plot.fig)


@pytest.mark.api
def test_dendrogram_condensed_does_not_auto_show(toy_results):
    """
    Ensures condensed dendrogram rendering has no implicit display side effect.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    _use_agg_backend()
    show_calls = []
    plot = None
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: show_calls.append((args, kwargs))
    try:
        plot = plot_dendrogram_condensed(toy_results)
        assert not show_calls
    finally:
        plt.show = plt_show
        if plot is not None:
            plt.close(plot.fig)


@pytest.mark.api
def test_dendrogram_condensed_plot_handle_save_supports_kwargs(toy_results, tmp_path):
    """
    Ensures the condensed plot handle supports explicit savefig kwargs.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
        tmp_path (Path): Temporary output directory.
    """
    _use_agg_backend()
    out = tmp_path / "condensed.png"
    plot = plot_dendrogram_condensed(toy_results)
    try:
        plot.save(out, dpi=150, bbox_inches="tight")
        assert out.exists()
        assert out.stat().st_size > 0
    finally:
        plt.close(plot.fig)


@pytest.mark.api
def test_dendrogram_condensed_plot_handle_rejects_closed_figure(toy_results):
    """
    Ensures the condensed plot handle errors after the backing figure is closed.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    _use_agg_backend()
    plot = plot_dendrogram_condensed(toy_results)
    plt.close(plot.fig)
    with pytest.raises(RuntimeError, match="figure is closed"):
        plot.show()
    with pytest.raises(RuntimeError, match="figure is closed"):
        plot.save("unused.png")


@pytest.mark.api
def test_dendrogram_condensed_label_fields_respect_np_order(toy_results):
    """
    Ensures condensed dendrogram labels keep label first while respecting n/p order
    from label_fields and emitting a single stats block.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    _use_agg_backend()
    plot = None
    try:
        plot = plot_dendrogram_condensed(
            toy_results,
            label_fields=("label", "p", "n"),
        )
        texts = [t.get_text() for ax in plot.fig.axes for t in ax.texts]
        # Cluster labels are asserted via rendered text fragments rather than data objects.
        label_texts = [t for t in texts if "$p$=" in t and "n=" in t]
        assert label_texts, "Expected cluster label text to be rendered."
        for txt in label_texts:
            assert " (" in txt
            assert txt.strip().endswith(")")
            assert "$p$=" in txt
            assert "n=" in txt
            assert txt.find("$p$=") < txt.find("n=")
    finally:
        if plot is not None:
            plt.close(plot.fig)


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
def test_dendrogram_condensed_max_words_controls_label_building_compressed(toy_results):
    """
    Ensures max_words controls compressed label building.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    _use_agg_backend()
    plot = None
    try:
        plot = plot_dendrogram_condensed(
            toy_results,
            label_mode="compressed",
            max_words=1,
            label_fields=("label",),
            wrap_text=False,
        )
        texts = [t.get_text().strip() for ax in plot.fig.axes for t in ax.texts]
        # Exclude placeholders to focus on generated cluster label text.
        cluster_texts = [t for t in texts if t and t != "—"]
        assert cluster_texts, "Expected generated cluster labels to be rendered."
        for txt in cluster_texts:
            assert len(txt.replace("\n", " ").split()) <= 1
    finally:
        if plot is not None:
            plt.close(plot.fig)


@pytest.mark.api
def test_dendrogram_condensed_max_words_controls_display(toy_results):
    """
    Ensures max_words truncates rendered override labels.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    _use_agg_backend()
    plot = None
    try:
        cid = int(toy_results.cluster_layout().cluster_spans[0][0])
        plot = plot_dendrogram_condensed(
            toy_results,
            label_overrides={cid: "Alpha Beta Gamma"},
            label_fields=("label",),
            max_words=1,
            wrap_text=False,
        )
        texts = [t.get_text() for ax in plot.fig.axes for t in ax.texts]
        assert any(t.strip() == "Alpha" for t in texts)
        assert not any("Alpha Beta" in t for t in texts)
    finally:
        if plot is not None:
            plt.close(plot.fig)


@pytest.mark.api
def test_dendrogram_condensed_override_respects_np_field_contract(toy_results):
    """
    Ensures label_fields controls displayed stats/order even when label_overrides are provided.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    _use_agg_backend()
    plot = None
    try:
        cid = int(toy_results.cluster_layout().cluster_spans[0][0])
        plot = plot_dendrogram_condensed(
            toy_results,
            label_overrides={cid: "CustomLabelIgnoredByFields"},
            label_fields=("n", "p"),
            wrap_text=False,
        )
        texts = [t.get_text() for ax in plot.fig.axes for t in ax.texts]
        # Labels are rendered as raw text; this selects n/p-only cluster entries.
        np_texts = [t for t in texts if t.startswith("(") and "n=" in t and "$p$=" in t]
        assert np_texts, "Expected n/p-only condensed cluster label text."
        assert all("CustomLabelIgnoredByFields" not in t for t in np_texts)
        for txt in np_texts:
            assert txt.find("n=") < txt.find("$p$=")
    finally:
        if plot is not None:
            plt.close(plot.fig)


@pytest.mark.unit
def test_dendrogram_condensed_override_keeps_deterministic_sigbar_source(toy_results):
    """
    Ensures label-only overrides preserve deterministic p-values used by the condensed sigbar.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    df = toy_results.cluster_labels()
    cluster_ids = [int(cid) for cid, _s, _e in toy_results.cluster_layout().cluster_spans]
    cid = cluster_ids[0]
    idx = cluster_ids.index(cid)
    base_pval = float(df.loc[df["cluster"] == cid, "pval"].iloc[0])
    labels, pvals, _lab_map, _cluster_sizes, _y = _prepare_cluster_labels(
        cluster_ids,
        df,
        toy_results.clusters,
        label_overrides={cid: f"Custom-{cid}"},
        wrap_text=False,
    )

    assert labels[idx] == f"Custom-{cid}"
    assert float(pvals[idx]) == pytest.approx(base_pval)
