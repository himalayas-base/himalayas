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
    DendrogramRenderer,
    GeneBarRenderer,
    MatrixRenderer,
)
from .renderers.gene_bar import render_gene_bar_track
from .style import StyleConfig
from .track_layout import TrackLayoutManager


class Plotter:
    """
    Layered, matrix-first plotter for HiMaLAYAS. Rendering happens only when `show()` is called.
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

        def _renderer_cluster_bar(
            ax, x0, width, payload, cluster_spans, label_map, style, row_order
        ) -> None:
            # value_map: cluster_id -> float or None
            value_map = payload.get("value_map", {})

            cmap_in = payload.get("cmap", style["sigbar_cmap"])
            cmap = plt.get_cmap(cmap_in) if isinstance(cmap_in, str) else cmap_in

            alpha = payload.get("alpha", style["sigbar_alpha"])
            norm = payload.get("norm", None)

            source = payload.get("source", None)

            # Collect p-values for clusters in span order
            pvals = np.full(len(cluster_spans), np.nan, dtype=float)
            if source == "cluster_labels_pval":
                for i, (cid, _s, _e) in enumerate(cluster_spans):
                    _label, pval = label_map.get(cid, (None, np.nan))
                    if pval is None or pd.isna(pval):
                        pvals[i] = np.nan
                    else:
                        pvals[i] = float(pval)
            else:
                for i, (cid, _s, _e) in enumerate(cluster_spans):
                    v = value_map.get(cid, np.nan)
                    if v is None or pd.isna(v):
                        pvals[i] = np.nan
                    else:
                        pvals[i] = float(v)

            # Convert to -log10(p) (visual space)
            with np.errstate(divide="ignore", invalid="ignore"):
                logp = -np.log10(pvals)

            valid = np.isfinite(logp) & (logp >= 0)
            if not np.any(valid):
                return

            # Implicit normalization: if no norm provided, scale to [0, max(-log10(p))]
            if norm is None:
                vmax = float(np.nanmax(logp[valid]))
                if not np.isfinite(vmax) or vmax <= 0:
                    return
                norm = plt.Normalize(vmin=0.0, vmax=vmax)

            # Map through norm -> cmap
            scaled = np.full_like(logp, np.nan, dtype=float)
            try:
                scaled[valid] = np.asarray(norm(logp[valid]), dtype=float)
            except TypeError:
                # If norm doesn't support vector input, fall back to scalar mapping
                scaled[valid] = np.array([float(norm(v)) for v in logp[valid]], dtype=float)

            # Draw patches (loop only for patch placement; colors are precomputed)
            for (cid, s, e), sv in zip(cluster_spans, scaled):
                if not np.isfinite(sv):
                    continue
                bar_color = cmap(sv)
                ax.add_patch(
                    plt.Rectangle(
                        (x0, s - 0.5),
                        width,
                        e - s + 1,
                        facecolor=bar_color,
                        edgecolor="none",
                        alpha=alpha,
                        zorder=1,
                    )
                )

        if enabled:
            self._track_layout.register_track(
                name=name,
                kind="cluster",
                renderer=_renderer_cluster_bar,
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
    # Rendering (intentionally not implemented yet)
    # --------------------------------------------------------

    # _ordered_cluster_spans removed: Plotter now fully consumes layout from Results

    def render(self) -> None:
        """
        Render the accumulated plot layers.

        Rendering is executed in the order layers were declared.
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
                df = kwargs["df"]
                if not isinstance(df, pd.DataFrame):
                    raise TypeError("cluster_labels must be a pandas DataFrame.")
                if "cluster" not in df.columns or "label" not in df.columns:
                    raise ValueError(
                        "cluster_labels DataFrame must contain columns: 'cluster', 'label'."
                    )

                # Build mapping cluster_id -> (label, pval)
                overrides = kwargs.get("overrides", None)
                if overrides is not None and not isinstance(overrides, dict):
                    raise TypeError("overrides must be a dict mapping cluster_id -> label string")

                label_map = {}
                for _, row in df.iterrows():
                    cid = int(row["cluster"])
                    base_label = str(row["label"])
                    label = overrides.get(cid, base_label) if overrides else base_label
                    pval = row.get("pval", None)
                    label_map[cid] = (label, pval)

                # Use precomputed cluster spans from layout
                spans = layout.cluster_spans
                cluster_sizes = layout.cluster_sizes

                label_axes = kwargs.get("axes", self._style["label_axes"])
                ax_lab = fig.add_axes(label_axes, frameon=False)
                ax_lab.set_xlim(0, 1)
                ax_lab.set_ylim(-0.5, n_rows - 0.5)
                ax_lab.invert_yaxis()  # Match heatmap orientation
                ax_lab.set_xticks([])
                ax_lab.set_yticks([])
                # Draw a clean gutter separating matrix from label panel
                gutter_w = kwargs.get("label_gutter_width", self._style["label_gutter_width"])
                gutter_color = kwargs.get("label_gutter_color", self._style["label_gutter_color"])
                ax_lab.add_patch(
                    plt.Rectangle(
                        (0.0, -0.5),
                        gutter_w,
                        n_rows,
                        facecolor=gutter_color,
                        edgecolor="none",
                        zorder=0,
                    )
                )

                # ------------------------------------------------------------
                # Explicit label-panel track-rail model
                # ------------------------------------------------------------
                # Track default spacings
                LABEL_TEXT_PAD = kwargs.get(
                    "label_text_pad", self._style.get("label_bar_pad", 0.01)
                )

                # (2) Compute track geometry (rail layout)
                base_x = kwargs.get("label_x", self._style["label_x"])
                self._track_layout.compute_layout(base_x, gutter_w)
                tracks = self._track_layout.get_tracks()
                end_x = self._track_layout.get_end_x()
                if end_x is None:
                    end_x = base_x + gutter_w
                label_text_x = end_x + LABEL_TEXT_PAD

                # Resolve label separator span once (explicit geometry; no None sentinels)
                sep_xmin = kwargs.get(
                    "label_sep_xmin",
                    self._style.get("label_sep_xmin"),
                )
                sep_xmax = kwargs.get(
                    "label_sep_xmax",
                    self._style.get("label_sep_xmax"),
                )

                if sep_xmin is None:
                    sep_xmin = label_text_x
                if sep_xmax is None:
                    sep_xmax = 1.0

                sep_xmin = float(np.clip(sep_xmin, 0.0, 1.0))
                sep_xmax = float(np.clip(sep_xmax, 0.0, 1.0))
                if sep_xmin > sep_xmax:
                    sep_xmin, sep_xmax = sep_xmax, sep_xmin

                font = kwargs.get("font", "Helvetica")
                fontsize = kwargs.get("fontsize", self._style.get("label_fontsize", 9))
                max_words = kwargs.get("max_words", None)
                skip_unlabeled = kwargs.get("skip_unlabeled", False)
                label_fields = kwargs.get("label_fields", self._style["label_fields"])
                if not isinstance(label_fields, (list, tuple)):
                    raise TypeError("label_fields must be a list or tuple of strings")
                allowed_fields = {"label", "n", "p"}
                if any(f not in allowed_fields for f in label_fields):
                    raise ValueError(f"label_fields may only contain {allowed_fields}")

                # (3) Render all tracks (row and cluster level)
                # Row-level tracks: render once per track
                for track in tracks:
                    if track["kind"] == "row":
                        track["renderer"](
                            ax_lab,
                            track["x0"],
                            track["width"],
                            track["payload"],
                            self.matrix,
                            row_order,
                            self._style,
                        )
                # Cluster-level tracks: render once per cluster
                for track in tracks:
                    if track["kind"] == "cluster":
                        track["renderer"](
                            ax_lab,
                            track["x0"],
                            track["width"],
                            track["payload"],
                            spans,
                            label_map,
                            self._style,
                            row_order,
                        )

                # Render bar labels (titles below bars) if requested
                # IMPORTANT: bar labels must be anchored in axes coordinates (0..1)
                # and padded in physical units (points) so they stay flush across figsize changes.
                bar_kwargs = None
                for l, kw in self._layers:
                    if l == "bar_labels":
                        bar_kwargs = kw  # last one wins

                if bar_kwargs is not None:
                    bar_pad_pts = bar_kwargs.get("pad", 2)  # interpreted as POINTS
                    bar_rotation = bar_kwargs.get("rotation", 0)

                    for track in tracks:
                        title = track.get("payload", {}).get("title", None)
                        if not title:
                            continue
                        x_center = (track.get("x0", 0.0) + track.get("x1", 0.0)) / 2.0
                        text_kwargs = {
                            "font": bar_kwargs.get("font", "Helvetica"),
                            "fontsize": bar_kwargs.get("fontsize", 10),
                            "color": bar_kwargs.get(
                                "color", self._style.get("text_color", "black")
                            ),
                            "alpha": bar_kwargs.get("alpha", 1.0),
                        }
                        ax_lab.annotate(
                            title,
                            xy=(x_center, 0.0),
                            xycoords=ax_lab.transAxes,
                            xytext=(0, -bar_pad_pts),
                            textcoords="offset points",
                            ha="center",
                            va="top",
                            **text_kwargs,
                            rotation=bar_rotation,
                            clip_on=False,
                        )

                # (4) Render label text and separators
                for cid, s, e in spans:
                    y_center = (s + e) / 2.0
                    if cid not in label_map:
                        if skip_unlabeled:
                            continue
                        text = kwargs.get("placeholder_text", self._style["placeholder_text"])
                        text_kwargs = {
                            "font": font,
                            "fontsize": fontsize,
                            "color": kwargs.get(
                                "color",
                                kwargs.get("placeholder_color", self._style["placeholder_color"]),
                            ),
                            "alpha": kwargs.get("alpha", self._style["placeholder_alpha"]),
                        }
                    else:
                        label, pval = label_map[cid]
                        # Omit specified words from label (case-insensitive)
                        omit_words = kwargs.get("omit_words", self._style["label_omit_words"])
                        if omit_words:
                            omit = {w.lower() for w in omit_words}
                            words = [w for w in label.split() if w.lower() not in omit]
                            label = " ".join(words) if words else label
                        n_members = cluster_sizes.get(cid, None)
                        parts = []
                        for field in label_fields:
                            if field == "label":
                                parts.append(label)
                            elif field == "n" and n_members is not None:
                                parts.append(f"n={n_members}")
                            elif field == "p" and pval is not None and not pd.isna(pval):
                                parts.append(rf"$p$={pval:.2e}")
                        if not parts:
                            text = label
                        else:
                            if parts[0] == label:
                                head = label
                                tail = parts[1:]
                                text = f"{head} ({', '.join(tail)})" if tail else head
                            else:
                                text = ", ".join(parts)
                        # --------------------------------------------------
                        # Label text policy (single source of truth)
                        # --------------------------------------------------
                        wrap_text = kwargs.get("wrap_text", True)
                        wrap_width = kwargs.get(
                            "wrap_width",
                            self._style.get("label_wrap_width", None),
                        )
                        overflow = kwargs.get("overflow", "wrap")  # "wrap" | "ellipsis"

                        # Word-level truncation policy
                        words = text.split()
                        if max_words is not None and len(words) > max_words:
                            if overflow == "ellipsis":
                                text = " ".join(words[:max_words]) + "…"
                            else:  # overflow == "wrap"
                                text = " ".join(words[:max_words])

                        # Layout-level wrapping (purely visual)
                        if wrap_text and wrap_width is not None:
                            import textwrap

                            text = "\n".join(textwrap.wrap(text, width=wrap_width))
                        text_kwargs = {
                            "font": font,
                            "fontsize": fontsize,
                            "color": kwargs.get("color", self._style.get("text_color", "black")),
                            "alpha": kwargs.get("alpha", 0.9),
                        }
                    ax_lab.text(
                        label_text_x,
                        y_center,
                        text,
                        va="center",
                        ha="left",
                        **text_kwargs,
                        fontweight="normal",
                        clip_on=False,
                    )
                    # Draw label separator if not topmost cluster
                    if s > 0:
                        sep_color = kwargs.get("label_sep_color", self._style["label_sep_color"])
                        sep_lw = kwargs.get("label_sep_lw", self._style["label_sep_lw"])
                        sep_alpha = kwargs.get("label_sep_alpha", self._style["label_sep_alpha"])
                        ax_lab.axhline(
                            s - 0.5,
                            xmin=sep_xmin,
                            xmax=sep_xmax,
                            color=sep_color,
                            linewidth=sep_lw,
                            alpha=sep_alpha,
                            zorder=0,
                        )

            elif layer == "sigbar_legend":
                cmap = plt.get_cmap(
                    kwargs.get("sigbar_cmap", self._style.get("sigbar_cmap", "YlOrBr"))
                )
                norm = kwargs.get("norm", None)
                if norm is not None and hasattr(norm, "vmin") and hasattr(norm, "vmax"):
                    lo = float(norm.vmin)
                    hi = float(norm.vmax)
                else:
                    lo = kwargs.get("sigbar_min_logp", self._style.get("sigbar_min_logp", 2.0))
                    hi = kwargs.get("sigbar_max_logp", self._style.get("sigbar_max_logp", 10.0))
                legend_axes = kwargs.get("axes", [0.92, 0.20, 0.015, 0.25])
                ax_leg = fig.add_axes(legend_axes, frameon=False)
                grad = np.linspace(0, 1, 256).reshape(-1, 1)
                ax_leg.imshow(grad, aspect="auto", cmap=cmap, origin="lower")
                ax_leg.set_yticks([0, 255])
                ax_leg.set_yticklabels([f"{lo:g}", f"{hi:g}"])
                ax_leg.set_ylabel("-log10(p)", fontsize=8)
                ax_leg.set_xticks([])
                ax_leg.tick_params(axis="y", labelsize=7)
                for spine in ax_leg.spines.values():
                    spine.set_visible(False)

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
            colorbar_layout = self._colorbar_layout or {}
            nrows = colorbar_layout.get("nrows")
            ncols = colorbar_layout.get("ncols")
            height = colorbar_layout.get("height", 0.05)
            hpad = colorbar_layout.get("hpad", 0.01)
            vpad = colorbar_layout.get("vpad", 0.01)
            gap = colorbar_layout.get("gap", 0.02)
            label_position = colorbar_layout.get("label_position", "below")

            border_color = colorbar_layout.get("border_color")
            if border_color is None:
                border_color = self._style.get("text_color", "black")
            border_width = colorbar_layout.get("border_width", 0.8)
            border_alpha = colorbar_layout.get("border_alpha", 1.0)

            # Typography (explicit, colorbar-scoped)
            fontsize = colorbar_layout.get("fontsize")
            if fontsize is None:
                fontsize = self._style.get("label_fontsize", 9)

            text_color = colorbar_layout.get("color")
            if text_color is None:
                text_color = self._style.get("text_color", "black")

            font = colorbar_layout.get("font", None)

            N = len(self._colorbars)
            if nrows is None and ncols is None:
                nrows, ncols = 1, N
            elif nrows is None:
                nrows = int(np.ceil(N / ncols))
            elif ncols is None:
                ncols = int(np.ceil(N / nrows))

            # Matrix axis bbox defines horizontal alignment
            bbox = ax.get_position()
            strip_y0 = bbox.y0 - height - gap
            strip_x0 = bbox.x0
            strip_w = bbox.width
            strip_h = height

            cell_w = (strip_w - hpad * (ncols - 1)) / ncols
            cell_h = (strip_h - vpad * (nrows - 1)) / nrows

            from matplotlib.colorbar import ColorbarBase

            for i, cb in enumerate(self._colorbars):
                r = i // ncols
                c = i % ncols

                x0 = strip_x0 + c * (cell_w + hpad)
                y0 = strip_y0 + (nrows - 1 - r) * (cell_h + vpad)

                ax_cb = fig.add_axes([x0, y0, cell_w, cell_h], frameon=True)
                bar_text_color = cb.get("color")
                if bar_text_color is None:
                    bar_text_color = text_color

                    cbar = ColorbarBase(
                        ax_cb,
                        cmap=cb["cmap"],
                        norm=cb["norm"],
                        orientation="horizontal",
                        ticks=cb.get("ticks", None),
                    )
                    outline = getattr(cbar, "outline", None)
                    if outline is not None:
                        outline.set_edgecolor(border_color)
                        outline.set_linewidth(border_width)
                        outline.set_alpha(border_alpha)
                    else:
                        for spine in ax_cb.spines.values():
                            spine.set_visible(True)
                            spine.set_linewidth(border_width)
                            spine.set_edgecolor(border_color)
                            spine.set_alpha(border_alpha)

                ax_cb.tick_params(
                    axis="x",
                    labelsize=fontsize,
                    colors=bar_text_color,
                )
                ax_cb.set_yticks([])

                if font is not None:
                    for t in ax_cb.get_xticklabels():
                        t.set_fontname(font)

                label = cb.get("label")
                if label:
                    if label_position == "below":
                        ax_cb.set_xlabel(
                            label,
                            fontsize=fontsize,
                            color=bar_text_color,
                            labelpad=2,
                            fontname=font if font is not None else None,
                        )
                    else:
                        ax_cb.set_title(
                            label,
                            fontsize=fontsize,
                            color=bar_text_color,
                            pad=2,
                            fontname=font if font is not None else None,
                        )
        # Matrix axis never owns ticks; always keep clean
        ax.set_xticks([])
        ax.set_yticks([])
        # Attach layout metadata for advanced users
        self.layout_ = layout
        self.colorbar_layout_ = colorbar_layout
        self._fig = fig

    def save(self, path: str, **kwargs) -> None:
        """
        Save the last rendered figure with correct background handling.
        """
        if self._fig is None:
            raise RuntimeError("Nothing to save — call render() first.")

        self._fig.savefig(
            path,
            facecolor=self._fig.get_facecolor(),
            **kwargs,
        )

    def show(self) -> None:
        """
        Show the last rendered figure with correct background handling.
        """
        if self._fig is None:
            raise RuntimeError("Nothing to show — call render() first.")

        plt.show()
