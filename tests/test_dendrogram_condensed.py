"""
tests/test_dendrogram_condensed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_rgba

from himalayas import Analysis
from himalayas.core.clustering import Clusters
from himalayas.core.results import Results
from himalayas.plot import CondensedDendrogramPlot, plot_dendrogram_condensed


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
def test_dendrogram_condensed_invalid_label_prefix_raises(toy_results):
    """
    Ensures invalid label_prefix values raise a ValueError.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        ValueError: If label_prefix contains unsupported values.
    """
    with pytest.raises(ValueError):
        plot_dendrogram_condensed(
            toy_results,
            label_prefix="bad",
        )


@pytest.mark.api
def test_dendrogram_condensed_rejects_unknown_kwargs(toy_results):
    """
    Ensures condensed dendrogram does not accept unknown kwargs.

    Args:
        toy_results (Results): Results fixture with clusters and layout.

    Raises:
        TypeError: If unknown keyword arguments are provided.
    """
    with pytest.raises(TypeError, match="unknown_kwarg"):
        plot_dendrogram_condensed(toy_results, unknown_kwarg=True)


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
def test_dendrogram_condensed_placeholder_controls_match_cluster_label_parity(toy_results):
    """
    Ensures placeholder text style overrides global label style and skip_unlabeled hides placeholders.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    _use_agg_backend()
    first_cluster = int(toy_results.cluster_layout().cluster_spans[0][0])
    # Keep layout unchanged but drop one cluster from labels to force placeholder rendering.
    filtered = toy_results.filter(f"cluster == {first_cluster}")
    placeholder_text = "Nonsignificant"
    plot = None
    plot2 = None
    try:
        # Render with placeholders enabled to compare placeholder and regular label styling.
        plot = plot_dendrogram_condensed(
            filtered,
            label_fields=("label",),
            label_color="gray",
            label_alpha=0.2,
            placeholder_text=placeholder_text,
            placeholder_color="red",
            placeholder_alpha=1.0,
            skip_unlabeled=False,
        )
        texts = [t for ax in plot.fig.axes for t in ax.texts if t.get_text().strip()]
        placeholder_nodes = [t for t in texts if t.get_text() == placeholder_text]
        regular_nodes = [t for t in texts if t.get_text() != placeholder_text]
        assert placeholder_nodes, "Expected placeholder text to be rendered."
        assert regular_nodes, "Expected at least one non-placeholder cluster label."

        # Compare one placeholder label node and one regular label node for style parity checks.
        placeholder_node = placeholder_nodes[0]
        regular_node = regular_nodes[0]
        assert to_rgba(placeholder_node.get_color()) == pytest.approx(to_rgba("red"))
        assert float(placeholder_node.get_alpha()) == pytest.approx(1.0)
        assert to_rgba(regular_node.get_color()) == pytest.approx(to_rgba("gray"))
        assert float(regular_node.get_alpha()) == pytest.approx(0.2)

        # Re-render with skip_unlabeled to verify placeholders are dropped while regular labels remain.
        plot2 = plot_dendrogram_condensed(
            filtered,
            label_fields=("label",),
            placeholder_text=placeholder_text,
            skip_unlabeled=True,
        )
        texts2 = [t.get_text().strip() for ax in plot2.fig.axes for t in ax.texts if t.get_text().strip()]
        assert placeholder_text not in texts2
        assert texts2, "Expected non-placeholder labels to remain rendered."
    finally:
        if plot is not None:
            plt.close(plot.fig)
        if plot2 is not None:
            plt.close(plot2.fig)


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
def test_dendrogram_condensed_plot_handle_rebuilds_closed_figure(toy_results, tmp_path):
    """
    Ensures the condensed plot handle rebuilds after the backing figure is closed.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
        tmp_path (Path): Temporary output directory.
    """
    _use_agg_backend()
    out = tmp_path / "condensed_rebuild.png"
    plot = plot_dendrogram_condensed(toy_results)
    plt_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        # Simulate a closed backend figure between notebook cells.
        plt.close(plot.fig)
        # show() should transparently rebuild the closed figure.
        plot.show()
        assert plot._figure_is_open()

        # save() should also rebuild after close and still write the file.
        plt.close(plot.fig)
        plot.save(out, dpi=150)
        assert out.exists()
        assert out.stat().st_size > 0
    finally:
        plt.show = plt_show
        if plot._figure_is_open():
            plt.close(plot.fig)


@pytest.mark.api
def test_dendrogram_condensed_label_fields_respect_np_order(toy_results):
    """
    Ensures condensed dendrogram labels keep label first while respecting q/fe/p/n order
    from label_fields and emitting a single stats block.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    _use_agg_backend()
    plot = None
    try:
        results_q = toy_results.with_qvalues()
        plot = plot_dendrogram_condensed(
            results_q,
            label_fields=("label", "q", "fe", "p", "n"),
        )
        texts = [t.get_text() for ax in plot.fig.axes for t in ax.texts]
        # Cluster labels are asserted via rendered text fragments rather than data objects.
        label_texts = [t for t in texts if "$q$=" in t and "FE=" in t and "$p$=" in t and "n=" in t]
        assert label_texts, "Expected cluster label text to be rendered."
        for txt in label_texts:
            assert " (" in txt
            assert txt.strip().endswith(")")
            assert "$q$=" in txt
            assert "FE=" in txt
            assert "$p$=" in txt
            assert "n=" in txt
            assert txt.find("$q$=") < txt.find("FE=")
            assert txt.find("FE=") < txt.find("$p$=")
            assert txt.find("$p$=") < txt.find("n=")
    finally:
        if plot is not None:
            plt.close(plot.fig)


@pytest.mark.api
def test_dendrogram_condensed_none_label_fields_hides_text(toy_results):
    """
    Ensures label_fields=None suppresses condensed cluster text when no prefix is requested.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    _use_agg_backend()
    plot = None
    try:
        plot = plot_dendrogram_condensed(
            toy_results,
            label_fields=None,
        )
        texts = [t.get_text().strip() for ax in plot.fig.axes for t in ax.texts if t.get_text().strip()]
        assert not texts
    finally:
        if plot is not None:
            plt.close(plot.fig)


@pytest.mark.api
def test_dendrogram_condensed_label_prefix_cid_supports_compressed_labels(toy_results):
    """
    Ensures label_prefix='cid' prefixes compressed condensed labels.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    _use_agg_backend()
    plot = None
    try:
        plot = plot_dendrogram_condensed(
            toy_results,
            label_mode="compressed",
            label_fields=("label", "fe"),
            label_prefix="cid",
            wrap_text=False,
        )
        texts = [t.get_text().strip() for ax in plot.fig.axes for t in ax.texts if t.get_text().strip()]
        assert any(txt.split(". ", 1)[0].isdigit() for txt in texts if ". " in txt)
        assert any("FE=" in txt for txt in texts)
    finally:
        if plot is not None:
            plt.close(plot.fig)


@pytest.mark.api
@pytest.mark.parametrize(
    "label_fields",
    [("label",), ("n", "p"), None],
    ids=["with_label_fields", "np_only", "without_label_fields"],
)
def test_dendrogram_condensed_label_prefix_precedence_override_wins(toy_results, label_fields):
    """
    Ensures label_prefix is applied before overrides and explicit overrides win.

    Args:
        toy_results (Results): Results fixture with clusters and layout.
    """
    _use_agg_backend()
    spans = toy_results.cluster_layout().cluster_spans
    assert len(spans) >= 2
    override_cid = int(spans[0][0])
    regular_cid = int(spans[1][0])
    override_label = "Custom Prefix Override"
    plot = None
    try:
        plot = plot_dendrogram_condensed(
            toy_results,
            label_fields=label_fields,
            label_prefix="cid",
            label_overrides={override_cid: override_label},
            wrap_text=False,
        )
        texts = [t.get_text().strip() for ax in plot.fig.axes for t in ax.texts if t.get_text().strip()]
        if label_fields is None:
            assert override_label in texts
        else:
            assert any(txt.startswith(override_label) for txt in texts)
        assert all(txt != f"{override_cid}. {override_label}" for txt in texts)
        if label_fields is None:
            assert f"{regular_cid}." in texts
        else:
            assert any(txt.startswith(f"{regular_cid}. ") for txt in texts)
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
    # Replace cluster labels with unmapped names so row-to-cluster mapping fails deterministically.
    bad_labels = [f"x{i}" for i in range(len(toy_results.matrix.labels))]
    bad_clusters = Clusters(
        toy_results.clusters.linkage_matrix,
        labels=bad_labels,
        threshold=toy_results.clusters.threshold,
    )
    results = Results(
        toy_results.df,
        matrix=toy_results.matrix,
        clusters=bad_clusters,
        layout=toy_results.cluster_layout(),
        parent=toy_results,
    )
    with pytest.raises(ValueError):
        plot_dendrogram_condensed(results)


@pytest.mark.api
def test_dendrogram_condensed_single_cluster_raises_clear_error(
    toy_matrix,
    toy_annotations,
):
    """
    Ensures single-cluster inputs raise a descriptive ValueError.

    Args:
        toy_matrix (Matrix): Matrix fixture.
        toy_annotations (Annotations): Annotation fixture.

    Raises:
        ValueError: If fewer than two clusters are available for condensed branching.
    """
    analysis = (
        Analysis(toy_matrix, toy_annotations)
        .cluster(linkage_threshold=1e9)
        .enrich()
        .finalize(col_cluster=False)
    )
    results = analysis.results
    assert results is not None
    assert results.clusters is not None
    assert len(results.clusters.unique_clusters) == 1

    with pytest.raises(ValueError, match="at least two clusters"):
        plot_dendrogram_condensed(results)
