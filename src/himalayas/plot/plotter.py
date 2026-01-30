"""
himalayas/plot/plotter
~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.results import Results
from .renderers import (
    AxesRenderer,
    BoundaryRegistry,
    ClusterLabelsRenderer,
    ColorbarRenderer,
    DendrogramRenderer,
    GeneBarRenderer,
    MatrixRenderer,
    render_cluster_bar_track,
    SigbarLegendRenderer,
)
from .renderers.gene_bar import render_gene_bar_track
from .style import StyleConfig
from .track_layout import TrackLayoutManager


class Plotter:
    """
    Layered, matrix-first plotter for HiMaLAYAS. Rendering happens when `show()` or
    `save()` is called.
    """

    def __init__(self, results: Results) -> None:
        """
        Initializes Plotter.

        Args:
            results (Results): Results object to plot.

        Raises:
            AttributeError: If Results object is missing required attributes.
        """
        self.results = results

        if not hasattr(results, "matrix"):
            raise AttributeError("Plotter expects Results with a `.matrix` attribute")
        if not hasattr(results, "cluster_layout"):
            raise AttributeError("Plotter expects Results exposing `cluster_layout()`")

        self.matrix = results.matrix

        # Declarative plot plan (ordered)
        self._layers = []

        # Declarative colorbar specs (global, figure-aligned)
        self._colorbars = []
        self._colorbar_layout = None

        # Label panel track layout manager
        self._track_layout = TrackLayoutManager()

        # --------------------------------------------------------
        # Default styling/layout (user-overridable via layer kwargs)
        # --------------------------------------------------------
        self.style = StyleConfig()
        self._style = self.style

        # Figure-level configuration (explicit, opt-in)
        self._background = None
        self._fig = None

    def add_colorbar(
        self,
        *,
        name: str,
        cmap,
        norm,
        label: Optional[str] = None,
        ticks: Optional[Sequence[float]] = None,
        color: Optional[str] = None,
    ) -> Plotter:
        """
        Declare a global colorbar explaining a visual encoding.

        Colorbars are figure-aligned and rendered in a strip below the matrix.
        They are not data-aligned and do not participate in row or cluster layout.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("colorbar `name` must be a non-empty string")
        self._colorbars.append(
            {
                "name": name,
                "cmap": cmap,
                "norm": norm,
                "label": label,
                "ticks": ticks,
                "color": color,
            }
        )
        return self

    def plot_colorbars(
        self,
        *,
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        height: float = 0.05,
        hpad: float = 0.01,
        vpad: float = 0.01,
        gap: float = 0.02,
        border_color: Optional[str] = None,
        border_width: float = 0.8,
        border_alpha: float = 1.0,
        fontsize: Optional[float] = None,
        font: Optional[str] = None,
        color: Optional[str] = None,
        label_position: str = "below",
    ) -> Plotter:
        """
        Declare layout for the bottom colorbar strip.

        Parameters
        ----------
        nrows, ncols : int or None
            Grid layout. If one is None, it is inferred from the other.
            If both are None, defaults to a single row.
        height : float
            Total height of the colorbar strip (figure fraction).
        hpad : float
            Horizontal spacing between colorbars (figure fraction).
        vpad : float
            Vertical spacing between colorbars (figure fraction).
        gap : float
            Vertical gap between the matrix and the colorbar strip (figure fraction).
        border_color : str or None
            Color of the colorbar border. If None, uses the global text color.
        border_width : float
            Line width of the colorbar border.
        border_alpha : float
            Alpha value applied to the colorbar border.
        fontsize : float or None
            Font size for colorbar tick labels and label/title. If None, uses style label_fontsize.
        font : str or None
            Font family name for ticks and label/title. If None, uses Matplotlib default.
        color : str or None
            Text color for ticks and label/title. If None, uses style text_color.
        label_position : {"below", "above"}
            Where to place colorbar labels.
        """
        if label_position not in {"below", "above"}:
            raise ValueError("label_position must be 'below' or 'above'")
        self._colorbar_layout = {
            "nrows": nrows,
            "ncols": ncols,
            "height": height,
            "hpad": hpad,
            "vpad": vpad,
            "gap": gap,
            "border_color": border_color,
            "border_width": border_width,
            "border_alpha": border_alpha,
            "fontsize": fontsize,
            "font": font,
            "color": color,
            "label_position": label_position,
        }
        return self

    def set_background(self, color: str) -> Plotter:
        """
        Set figure background color (used for display and save).
        """
        self._background = color
        return self

    def set_label_track_order(self, order: Optional[Sequence[str]]) -> Plotter:
        """
        Set the order of label-panel tracks in the label panel.
        Pass None to use default order. Otherwise, must be a list/tuple of unique strings.
        """
        self._track_layout.set_order(order)
        return self

    def plot_cluster_bar(self, name: str, values: object, **kwargs) -> Plotter:
        """
        Declare a cluster-level label-panel bar (e.g. for p-values or other cluster metrics).
        Supported values: dict, pandas Series, DataFrame with columns (cluster, pval).
        Only kind='pvalue' is implemented.
        """
        # Normalize values into a dict: cluster_id -> value (float or None)
        value_map = None
        if isinstance(values, dict):
            value_map = dict(values)
        elif isinstance(values, pd.Series):
            value_map = {int(k): float(v) if pd.notna(v) else None for k, v in values.items()}
        elif isinstance(values, pd.DataFrame):
            # Accept columns: cluster, pval (by default)
            col_cluster = kwargs.get("cluster_col", "cluster")
            col_val = kwargs.get("pval_col", "pval")
            if col_cluster not in values.columns or col_val not in values.columns:
                raise ValueError(f"DataFrame must contain columns '{col_cluster}' and '{col_val}'")
            value_map = {
                int(row[col_cluster]): float(row[col_val]) if pd.notna(row[col_val]) else None
                for _, row in values.iterrows()
            }
        else:
            raise TypeError(
                "values must be dict, pandas Series, or DataFrame with 'cluster' and 'pval' columns"
            )
        kind = kwargs.get("kind", "pvalue")
        if kind != "pvalue":
            raise NotImplementedError("Only kind='pvalue' is implemented for cluster bar")
        width = kwargs.get("width", self._style.get("sigbar_width", 0.015))
        left_pad = kwargs.get("left_pad", 0.0)
        right_pad = kwargs.get("right_pad", 0.0)
        cmap = kwargs.get("cmap", self._style.get("sigbar_cmap", "YlOrBr"))
        norm = kwargs.get("norm", None)
        alpha = kwargs.get("alpha", self._style.get("sigbar_alpha", 0.9))
        enabled = kwargs.get("enabled", True)
        title = kwargs.get("title", None)

        if enabled:
            self._track_layout.register_track(
                name=name,
                kind="cluster",
                renderer=render_cluster_bar_track,
                left_pad=left_pad,
                width=width,
                right_pad=right_pad,
                enabled=enabled,
                payload={
                    "value_map": value_map,
                    "cmap": cmap,
                    "norm": norm,
                    "alpha": alpha,
                    "title": title,
                },
            )
        return self

    def plot_sigbar_legend(self, **kwargs) -> Plotter:
        """
        Declare a significance bar legend.

        Explains the color mapping of the cluster-level significance bar
        (based on -log10(p)). Off by default.
        """
        self._layers.append(("sigbar_legend", kwargs))
        return self

    # --------------------------------------------------------
    # Core plotting layers (declarative only)
    # --------------------------------------------------------
    def plot_matrix(self, **kwargs) -> Plotter:
        """
        Declare the main matrix heatmap layer.

        Parameters are stored verbatim and interpreted at render time.
        Use `gutter_color` to set the matrix panel background (helps mask edge artifacts).
        """
        self._layers.append(("matrix", kwargs))
        return self

    def plot_matrix_axis_labels(self, **kwargs) -> Plotter:
        """Declare axis labels for the matrix.

        Keyword arguments
        -----------------
        xlabel, ylabel : str
            Axis label text.
        fontsize : float
            Font size for both axis labels.
        font : str or None
            Font family to apply to both axis labels (e.g., "Helvetica").
        color : str
            Text color for both axis labels.
        """
        self._layers.append(("matrix_axis_labels", kwargs))
        return self

    def plot_row_ticks(self, labels: Optional[Sequence[str]] = None, **kwargs) -> Plotter:
        """
        Declare row tick labels for the matrix.

        Keyword arguments:
          - position (str): "left" or "right" (default: "right").
        """
        self._layers.append(("row_ticks", {"labels": labels, **kwargs}))
        return self

    def plot_col_ticks(self, labels: Optional[Sequence[str]] = None, **kwargs) -> Plotter:
        """
        Declare column tick labels for the matrix.

        Keyword arguments:
          - position (str): "top" or "bottom" (default: "top").
        """
        self._layers.append(("col_ticks", {"labels": labels, **kwargs}))
        return self

    def plot_bar_labels(self, **kwargs) -> Plotter:
        """
        Declare titles for bar tracks (shown below bars in label panel).
        """
        self._layers.append(("bar_labels", kwargs))
        return self

    def plot_dendrogram(self, **kwargs) -> Plotter:
        """
        Declare a dendrogram layer aligned to the matrix.

        Keyword arguments:
          - data_pad (float): Data-space padding to prevent edge clipping in the dendrogram panel (default: 0.25).
        """
        self._layers.append(("dendrogram", kwargs))
        return self

    def plot_cluster_bars(self, **kwargs) -> Plotter:
        """
        Declare a cluster membership bar layer.
        """
        self._layers.append(("cluster_bars", kwargs))
        return self

    def plot_gene_bar(self, values: Mapping[Any, Any], **kwargs) -> Plotter:
        """Declare a single row-level gene annotation bar.

        This layer is purely visual: it never affects clustering, ordering, or statistics.

        Parameters
        ----------
        values : dict
            Mapping from gene identifier (row index label) to either:
              - categorical value (mode="categorical")
              - numeric value (mode="continuous")

        Keyword options
        ---------------
        mode : {"categorical", "continuous"}, default "categorical"
            Categorical: each row gets a color via `colors[value]`.
            Continuous: each row gets a color from `cmap(norm(value))`.
        colors : dict, optional
            Category -> color mapping (categorical mode).
        cmap : str, optional
            Matplotlib colormap name (continuous mode). Default: "viridis".
        vmin, vmax : float, optional
            Color scale limits (continuous mode). If omitted, inferred from finite values.
        missing_color : str, optional
            Color for genes not present in `values` (or NaN in continuous mode).
        axes : list[float], optional
            Override axes box [x0, y0, w, h]. If omitted, auto-placed.
        placement : {"between_dendro_matrix", "label_panel"}, default "between_dendro_matrix"
            Where to draw the bar. "label_panel" places the bar in the right-side label panel,
            adjacent to the significance bar, and displaces label text accordingly.
        gene_bar_left_pad, gene_bar_width, gene_bar_right_pad : float, optional
            Padding/width for label_panel placement.
        """
        placement = kwargs.get("placement", "between_dendro_matrix")
        if placement == "label_panel":
            # Register as an explicit track in the label panel
            # Get explicit paddings
            left_pad = kwargs.get("gene_bar_left_pad", 0.0)
            width = kwargs.get("gene_bar_width", self._style.get("gene_bar_width", 0.015))
            right_pad = kwargs.get("gene_bar_right_pad", 0.0)

            enabled = kwargs.get("enabled", True)
            if enabled:
                self._track_layout.register_track(
                    name=kwargs.get("name", "gene_bar"),
                    kind="row",
                    renderer=render_gene_bar_track,
                    left_pad=left_pad,
                    width=width,
                    right_pad=right_pad if right_pad is not None else 0.0,
                    enabled=enabled,
                    payload={**kwargs, "values": values, "title": kwargs.get("title", None)},
                )
            return self

        # Default: render as its own bar axis between dendrogram and matrix
        self._layers.append(("gene_bar", {"values": values, **kwargs}))
        return self

    def plot_labels(self, **kwargs) -> Plotter:
        """
        Declare annotation label overlays (e.g. GO terms).
        """
        self._layers.append(("labels", kwargs))
        return self

    def plot_cluster_labels(self, cluster_labels: pd.DataFrame, **kwargs) -> Plotter:
        """
        Declare cluster-level textual labels.

        `cluster_labels` must be a DataFrame containing:
          - 'cluster' (int)
          - 'label'   (str)
          - 'pval'    (float, optional) — best p-value for the cluster

        Labels are rendered on the right margin, centered on each cluster span
        in dendrogram leaf order.

        Verbosity control
        -----------------
        Label content is fully controlled at the plotting level via `label_fields`.
        This allows users to simplify or enrich labels *without recomputing analysis*.

        Examples:
          - Ultra-clean labels (name only):
                label_fields=("label",)

          - Name + cluster size (no statistics):
                label_fields=("label", "n")

          - Name + p-value only:
                label_fields=("label", "p")

          - Diagnostics-only (no descriptive text):
                label_fields=("p",)

        Parameters
        ----------
        label_fields : tuple[str], optional
            Ordered fields to include in rendered labels.
            Allowed values:
              - "label" : cluster name
              - "n"     : cluster size
              - "p"     : cluster p-value (formatted)
            Default: ("label", "n", "p").
        overrides : dict, optional
            Optional per-cluster label overrides. Values may be:
              - str: override label only (stats suppressed by default)
              - dict with keys {"label", "pval", "hide_stats"}.
            If a p-value is provided in the override dict, it is shown for that cluster.
            If no p-value is provided, p-values are suppressed for that cluster.
        Other keyword arguments control typography, spacing, and panel geometry.
        """
        # Reject deprecated sigbar/track-order kwargs
        _deprecated = [
            "show_sigbar",
            "sigbar_width",
            "sigbar_left_pad",
            "sigbar_right_pad",
            "sigbar_cmap",
            "sigbar_min_logp",
            "sigbar_max_logp",
            "sigbar_alpha",
            "label_track_order",
        ]
        present = [k for k in _deprecated if k in kwargs]
        if present:
            raise TypeError(
                f"plot_cluster_labels no longer accepts {present}. "
                "Use plot_cluster_bar(...) and set_label_track_order(...) instead."
            )
        self._layers.append(("cluster_labels", {"df": cluster_labels, **kwargs}))
        return self

    def plot_title(self, title: str, **kwargs) -> Plotter:
        """
        Declare a plot title.
        """
        self._layers.append(("title", {"title": title, **kwargs}))
        return self

    # --------------------------------------------------------
    # Rendering
    # --------------------------------------------------------

    # _ordered_cluster_spans removed: Plotter now fully consumes layout from Results

    def _render(self) -> None:
        """
        Render the accumulated plot layers in declaration order.
        """
        if not self._layers:
            raise RuntimeError("No plot layers declared.")

        # Consume authoritative geometry from Results
        layout = self.results.cluster_layout()
        if len(layout.leaf_order) != self.matrix.df.shape[0]:
            raise ValueError("Dendrogram leaf order does not match matrix dimensions.")

        # NOTE:
        #   - layout.leaf_order controls row ordering (statistically meaningful)
        #   - layout.col_order controls column ordering (visual only)
        row_order = layout.leaf_order
        col_order = layout.col_order

        if row_order is None:
            raise ValueError("Row order is required for plotting.")

        if len(row_order) != self.matrix.df.shape[0]:
            raise ValueError("Dendrogram leaf order does not match matrix dimensions.")

        if col_order is not None:
            if len(col_order) != self.matrix.df.shape[1]:
                raise ValueError("Column order does not match matrix dimensions.")

        n_rows = self.matrix.df.shape[0]

        # Create figure and main axis
        fig, ax = plt.subplots(figsize=self._style["figsize"])
        fig.subplots_adjust(**self._style["subplots_adjust"])
        if self._background is not None:
            fig.patch.set_facecolor(self._background)

        # Single-ownership boundary registry (matrix-aligned horizontals)
        matrix_kwargs = None
        cluster_boundary_kwargs = None
        for _layer, _kwargs in self._layers:
            if _layer == "matrix":
                matrix_kwargs = _kwargs
            elif _layer == "cluster_labels":
                cluster_boundary_kwargs = _kwargs

        boundary_registry = BoundaryRegistry()

        # Minor row gridlines (internal only; suppress axis extremes)
        if matrix_kwargs is not None and matrix_kwargs.get("show_minor_rows", True):
            step = matrix_kwargs.get("minor_row_step", 1)
            lw = matrix_kwargs.get("minor_row_lw", 0.15)
            alpha = matrix_kwargs.get("minor_row_alpha", 0.15)
            for y in range(0, n_rows, step):
                b = y - 0.5
                if b <= -0.5 or b >= n_rows - 0.5:
                    continue
                boundary_registry.register(b, lw=lw, color="black", alpha=alpha)

        # Cluster boundaries (suppress axis extremes; cluster lines override minor lines)
        if cluster_boundary_kwargs is not None:
            lw = cluster_boundary_kwargs.get("boundary_lw", self._style["boundary_lw"])
            alpha = cluster_boundary_kwargs.get("boundary_alpha", self._style["boundary_alpha"])
            color = cluster_boundary_kwargs.get("boundary_color", self._style["boundary_color"])
            for _cid, s, _e in layout.cluster_spans:
                b = s - 0.5
                if b <= -0.5 or b >= n_rows - 0.5:
                    continue
                boundary_registry.register(b, lw=lw, color=color, alpha=alpha)

        dendro_boundary_style = None
        if cluster_boundary_kwargs is not None:
            dendro_boundary_style = {
                "color": cluster_boundary_kwargs.get(
                    "dendro_boundary_color",
                    self._style["dendro_boundary_color"],
                ),
                "lw": cluster_boundary_kwargs.get(
                    "dendro_boundary_lw",
                    self._style["dendro_boundary_lw"],
                ),
                "alpha": cluster_boundary_kwargs.get(
                    "dendro_boundary_alpha",
                    self._style["dendro_boundary_alpha"],
                ),
            }

        for layer, kwargs in self._layers:
            if layer == "gene_bar":
                if kwargs.get("placement", "between_dendro_matrix") == "label_panel":
                    # Registered in plot_gene_bar; nothing to render here.
                    continue
                renderer = GeneBarRenderer(**kwargs)
                renderer.render(fig, ax, self.matrix, layout, self._style)
                continue
            if layer == "matrix":
                figsize = kwargs.get("figsize", None)
                if figsize is not None:
                    fig.set_size_inches(figsize[0], figsize[1], forward=True)

                subplots_adjust = kwargs.get("subplots_adjust", None)
                if subplots_adjust is not None:
                    fig.subplots_adjust(**subplots_adjust)

                renderer = MatrixRenderer(**kwargs)
                renderer.render(
                    fig,
                    ax,
                    self.matrix,
                    layout,
                    self._style,
                    boundary_registry=boundary_registry,
                )

            # Note: orientation="left" already handles axis direction.
            # Do NOT manually reverse x-limits or the dendrogram will be mirrored.
            elif layer == "matrix_axis_labels":
                renderer = AxesRenderer("matrix_axis_labels", **kwargs)
                renderer.render(fig, ax, self.matrix, layout, self._style)

            elif layer == "row_ticks":
                renderer = AxesRenderer("row_ticks", **kwargs)
                renderer.render(fig, ax, self.matrix, layout, self._style)

            elif layer == "col_ticks":
                renderer = AxesRenderer("col_ticks", **kwargs)
                renderer.render(fig, ax, self.matrix, layout, self._style)

            elif layer == "dendrogram":
                renderer = DendrogramRenderer(**kwargs)
                renderer.render(
                    fig,
                    ax,
                    self.matrix,
                    layout,
                    self._style,
                    results=self.results,
                    boundary_style=dendro_boundary_style,
                )

            elif layer == "title":
                renderer = AxesRenderer("title", **kwargs)
                renderer.render(fig, ax, self.matrix, layout, self._style)

            elif layer == "cluster_labels":
                bar_kwargs = None
                for l, kw in self._layers:
                    if l == "bar_labels":
                        bar_kwargs = kw  # last one wins

                renderer_kwargs = dict(kwargs)
                df = renderer_kwargs.pop("df")
                renderer = ClusterLabelsRenderer(df, **renderer_kwargs)
                renderer.render(
                    fig,
                    ax,
                    self.matrix,
                    layout,
                    self._style,
                    self._track_layout,
                    bar_labels_kwargs=bar_kwargs,
                )

            elif layer == "sigbar_legend":
                renderer = SigbarLegendRenderer(**kwargs)
                renderer.render(fig, self._style)

            elif layer == "bar_labels":
                # Consumed inside the cluster label panel; no direct rendering.
                continue

            elif layer == "cluster_bars":
                raise NotImplementedError("plot_cluster_bars is not implemented yet")

            elif layer == "labels":
                raise NotImplementedError("plot_labels is not implemented yet")

            else:
                raise NotImplementedError(f"Unknown plot layer: {layer}")

        # ------------------------------------------------------------
        # Render bottom colorbar strip (global legends)
        # ------------------------------------------------------------
        colorbar_layout = None
        if self._colorbars:
            renderer = ColorbarRenderer(self._colorbars, self._colorbar_layout)
            colorbar_layout = renderer.render(fig, ax, self._style)
        # Matrix axis never owns ticks; always keep clean
        ax.set_xticks([])
        ax.set_yticks([])
        # Attach layout metadata for advanced users
        self.layout_ = layout
        self.colorbar_layout_ = colorbar_layout
        self._fig = fig

    def _figure_is_open(self) -> bool:
        if self._fig is None:
            return False
        try:
            return self._fig.number in plt.get_fignums()
        except Exception:
            return False

    def save(self, path: str, **kwargs) -> None:
        """
        Save the last rendered figure with correct background handling.
        """
        if self._fig is None or not self._figure_is_open():
            self._render()

        self._fig.savefig(
            path,
            facecolor=self._fig.get_facecolor(),
            **kwargs,
        )

    def show(self) -> None:
        """
        Show the last rendered figure with correct background handling.
        """
        if self._fig is None or not self._figure_is_open():
            self._render()

        plt.show()
