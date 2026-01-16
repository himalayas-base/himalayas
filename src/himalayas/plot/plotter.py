# ============================================================
# Foundational Plotter spine for HiMaLAYAS (prototype)
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from typing import Any, Mapping, Optional, Sequence
from matplotlib.colors import TwoSlopeNorm


class Plotter:
    """
    Layered plot builder for HiMaLAYAS.

    The Plotter is a pure consumer of analysis results:
      - It never recomputes clustering or statistics
      - It never mutates Matrix, Clusters, or Results
      - It accumulates declarative plotting layers
      - Rendering happens only when `show()` is called

    Designed as a deterministic, matrix-first plotting spine:
    dendrogram order is authoritative and never mutated at render time.
    """

    def __init__(self, results: Any) -> None:
        # Store authoritative inputs (support both Results and filtered Results)
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

        # Ordered list of declared label panel tracks (track-rail model)
        # Each entry: {
        #   "name": str,
        #   "kind": "row" or "cluster",
        #   "renderer": callable,
        #   "left_pad": float,
        #   "width": float,
        #   "right_pad": float,
        #   "enabled": bool,
        #   "payload": dict,
        # }
        self._declared_label_tracks = []
        self._active_label_tracks = []

        # Layout state: user-defined label track order
        self._label_track_order = None

        # --------------------------------------------------------
        # Default styling/layout (user-overridable via layer kwargs)
        # --------------------------------------------------------
        self._style = {
            # Figure layout
            "figsize": (9, 7),
            "subplots_adjust": {"left": 0.15, "right": 0.70, "bottom": 0.05, "top": 0.95},

            # Dendrogram axis box [x0, y0, w, h]
            "dendro_axes": [0.06, 0.05, 0.09, 0.90],
            "dendro_color": "#888888",
            "dendro_lw": 1.0,

            # Label panel axis box [x0, y0, w, h]
            "label_axes": [0.70, 0.05, 0.29, 0.90],
            "label_x": 0.02,

            # Gutter between matrix and label panel
            "label_gutter_width": 0.01,
            "label_gutter_color": "white",
            # Padding between matrix and ylabel axis (fraction of figure width)
            "ylabel_pad": 0.015,

            # Gene-level annotation bar (row-level, purely visual)
            # Placed between dendrogram and matrix by default.
            "gene_bar_width": 0.012,
            "gene_bar_gap": 0.006,
            "gene_bar_missing_color": "#eeeeee",
            # Bars rendered inside the label panel (to the left of text)
            # (label_bar_default_width, label_bar_default_gap removed)

            # Default settings for cluster p-value bars (e.g., sigbar)
            # NOTE: scaling is controlled by an explicit `norm` passed to plot_cluster_bar.
            # `sigbar_min_logp` / `sigbar_max_logp` are legacy defaults used only for the legend.
            "sigbar_width": 0.015,
            "sigbar_cmap": "YlOrBr",
            "sigbar_min_logp": 2.0,
            "sigbar_max_logp": 10.0,
            "sigbar_alpha": 0.9,
            "show_sigbar": True,
            # (sigbar_gap removed)

            # Label panel bar/text spacing
            "label_bar_pad": 0.01,

            # Cluster boundary lines
            "boundary_color": "black",
            "boundary_lw": 0.5,
            "boundary_alpha": 0.6,
            "dendro_boundary_color": "white",
            "dendro_boundary_lw": 0.5,
            "dendro_boundary_alpha": 0.3,

            # Placeholder for unlabeled clusters
            "placeholder_text": "—",
            "placeholder_color": "#b22222",
            "placeholder_alpha": 0.6,

            # Default text color (used unless overridden via kwargs)
            "text_color": "black",
            "title_fontsize": 14,
            "title_pad": 15,
            "label_fontsize": 9,

            # Separator lines in label panel
            "label_sep_color": "gray",
            "label_sep_lw": 0.5,
            "label_sep_alpha": 0.3,
            # Optional override for label separator segment span (axes coords 0..1)
            # If None, separators start after gutter+sigbar+pad and extend to 1.0
            "label_sep_xmin": None,
            "label_sep_xmax": None,
            # Words to omit from displayed cluster labels
            "label_omit_words": None,
            # Which fields to show in cluster labels, in order
            # Allowed values: "label", "n", "p"
            "label_fields": ("label", "n", "p"),
            # Optional label wrapping (characters per line); None = disabled
            "label_wrap_width": None,
        }

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
        enabled: bool = True,
    ) -> "Plotter":
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
                "enabled": enabled,
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
    ) -> "Plotter":
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

    def set_background(self, color: str) -> "Plotter":
        """
        Set figure background color (used for display and save).
        """
        self._background = color
        return self

    def set_label_track_order(self, order: Optional[Sequence[str]]) -> "Plotter":
        """
        Set the order of label-panel tracks in the label panel.
        Pass None to use default order. Otherwise, must be a list/tuple of unique strings.
        """
        if order is None:
            self._label_track_order = None
            return self
        if not isinstance(order, (list, tuple)):
            raise TypeError("label_track_order must be None or a list/tuple of unique strings")
        names = list(order)
        if any(not isinstance(n, str) for n in names):
            raise TypeError("label_track_order must be a list/tuple of unique strings")
        if len(set(names)) != len(names):
            raise ValueError("label_track_order contains duplicate track names")
        self._label_track_order = tuple(names)
        return self

    def plot_cluster_bar(self, name: str, values: object, **kwargs) -> "Plotter":
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
            value_map = {int(row[col_cluster]): float(row[col_val]) if pd.notna(row[col_val]) else None
                         for _, row in values.iterrows()}
        else:
            raise TypeError("values must be dict, pandas Series, or DataFrame with 'cluster' and 'pval' columns")
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

        def _renderer_cluster_bar(ax, x0, width, payload, cluster_spans, label_map, style, row_order) -> None:
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
                    try:
                        p = float(pval)
                    except Exception:
                        p = np.nan
                    pvals[i] = p
            else:
                for i, (cid, _s, _e) in enumerate(cluster_spans):
                    v = value_map.get(cid, np.nan)
                    try:
                        p = float(v)
                    except Exception:
                        p = np.nan
                    pvals[i] = p

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
            except Exception:
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
        track_spec = {
            "name": name,
            "kind": "cluster",
            "renderer": _renderer_cluster_bar,
            "left_pad": left_pad,
            "width": width,
            "right_pad": right_pad,
            "enabled": enabled,
            "payload": {
                "value_map": value_map,
                "cmap": cmap,
                "norm": norm,
                "alpha": alpha,
                "title": title,
            },
        }
        if enabled:
            self._declared_label_tracks.append(track_spec)
        return self

    def plot_sigbar_legend(self, **kwargs) -> "Plotter":
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
    def plot_matrix(self, **kwargs) -> "Plotter":
        """
        Declare the main matrix heatmap layer.

        Parameters are stored verbatim and interpreted at render time.
        """
        self._layers.append(("matrix", kwargs))
        return self

    def plot_matrix_axis_labels(self, **kwargs) -> "Plotter":
        """
        Declare axis labels for the matrix.
        """
        self._layers.append(("matrix_axis_labels", kwargs))
        return self

    def plot_row_ticks(self, labels: Optional[Sequence[str]] = None, **kwargs) -> "Plotter":
        """
        Declare row tick labels for the matrix.
        """
        self._layers.append(("row_ticks", {"labels": labels, **kwargs}))
        return self

    def plot_col_ticks(self, labels: Optional[Sequence[str]] = None, **kwargs) -> "Plotter":
        """
        Declare column tick labels for the matrix.
        """
        self._layers.append(("col_ticks", {"labels": labels, **kwargs}))
        return self

    def plot_bar_labels(self, **kwargs) -> "Plotter":
        """
        Declare titles for bar tracks (shown below bars in label panel).
        """
        self._layers.append(("bar_labels", kwargs))
        return self

    def plot_dendrogram(self, **kwargs) -> "Plotter":
        """
        Declare a dendrogram layer aligned to the matrix.

        Keyword arguments:
          - data_pad (float): Data-space padding to prevent edge clipping in the dendrogram panel (default: 0.25).
        """
        self._layers.append(("dendrogram", kwargs))
        return self

    def plot_cluster_bars(self, **kwargs) -> "Plotter":
        """
        Declare a cluster membership bar layer.
        """
        self._layers.append(("cluster_bars", kwargs))
        return self

    def plot_gene_bar(self, values: Mapping[Any, Any], **kwargs) -> "Plotter":
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
            # Compose renderer
            def _renderer_row_bar(ax, x0, width, payload, matrix, row_order, style) -> None:
                values = payload["values"]
                mode = payload.get("mode", "categorical")
                missing_color = payload.get("missing_color", style["gene_bar_missing_color"])
                row_ids = matrix.df.index.to_numpy()[row_order]
                if mode == "categorical":
                    colors = payload.get("colors", None)
                    if colors is None or not isinstance(colors, dict):
                        raise TypeError("categorical label_panel gene bar requires `colors` as a dict")
                    for i, gid in enumerate(row_ids):
                        cat = values.get(gid, None)
                        c = colors.get(cat, missing_color)
                        ax.add_patch(
                            plt.Rectangle((x0, i - 0.5), width, 1.0, facecolor=c, edgecolor="none", zorder=2)
                        )
                elif mode == "continuous":
                    cmap = plt.get_cmap(payload.get("cmap", "viridis"))
                    vals = np.array([values.get(gid, np.nan) for gid in row_ids], dtype=float)
                    finite = np.isfinite(vals)
                    if not np.any(finite):
                        for i in range(len(row_ids)):
                            ax.add_patch(
                                plt.Rectangle((x0, i - 0.5), width, 1.0, facecolor=missing_color, edgecolor="none", zorder=2)
                            )
                    else:
                        vmin = payload.get("vmin", float(np.nanmin(vals[finite])))
                        vmax = payload.get("vmax", float(np.nanmax(vals[finite])))
                        if vmin == vmax:
                            vmax = vmin + 1e-12
                        norm = plt.Normalize(vmin=vmin, vmax=vmax)
                        for i, v in enumerate(vals):
                            c = missing_color if not np.isfinite(v) else cmap(norm(v))
                            ax.add_patch(
                                plt.Rectangle((x0, i - 0.5), width, 1.0, facecolor=c, edgecolor="none", zorder=2)
                            )
                else:
                    raise ValueError("label_panel gene bar mode must be 'categorical' or 'continuous'")
            # Compose track spec
            track_spec = {
                "name": kwargs.get("name", "gene_bar"),
                "kind": "row",
                "renderer": _renderer_row_bar,
                "left_pad": left_pad,
                "width": width,
                "right_pad": right_pad if right_pad is not None else 0.0,
                "enabled": kwargs.get("enabled", True),
                "payload": {**kwargs, "values": values, "title": kwargs.get("title", None)},
            }
            if track_spec["enabled"]:
                self._declared_label_tracks.append(track_spec)
            return self

        # Default: render as its own bar axis between dendrogram and matrix
        self._layers.append(("gene_bar", {"values": values, **kwargs}))
        return self

    def plot_labels(self, **kwargs) -> "Plotter":
        """
        Declare annotation label overlays (e.g. GO terms).
        """
        self._layers.append(("labels", kwargs))
        return self

    def plot_cluster_labels(self, cluster_labels: pd.DataFrame, **kwargs) -> "Plotter":
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
            "show_sigbar", "sigbar_width", "sigbar_left_pad", "sigbar_right_pad",
            "sigbar_cmap", "sigbar_min_logp", "sigbar_max_logp", "sigbar_alpha",
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

    def plot_title(self, title: str, **kwargs) -> "Plotter":
        """
        Declare a plot title.
        """
        self._layers.append(("title", {"title": title, **kwargs}))
        return self

    # --------------------------------------------------------
    # Rendering (intentionally not implemented yet)
    # --------------------------------------------------------

    # _ordered_cluster_spans removed: Plotter now fully consumes layout from Results

    def show(self) -> None:
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

        # Start from row-reordered data
        data = self.matrix.df.iloc[row_order, :]

        # Apply column order if provided (layout-only, visual)
        if col_order is not None:
            if len(col_order) != self.matrix.df.shape[1]:
                raise ValueError("Column order does not match matrix dimensions.")
            data = data.iloc[:, col_order]

        data = data.values

        fig, ax = plt.subplots(figsize=self._style["figsize"])
        fig.subplots_adjust(**self._style["subplots_adjust"])
        if self._background is not None:
            fig.patch.set_facecolor(self._background)

        ax_dend = None  # created if dendrogram layer is rendered

        # Precompute dendrogram geometry for rendering only (for dendrogram panel)
        dendro = None

        # (removed: Render-safety padding block)

        # (removed obsolete render-flag cleanup)


        n_rows, n_cols = data.shape

        for layer, kwargs in self._layers:
            if layer == "gene_bar":
                if kwargs.get("placement", "between_dendro_matrix") == "label_panel":
                    # Registered in plot_gene_bar; nothing to render here.
                    continue
                values = kwargs.get("values")
                if not isinstance(values, dict):
                    raise TypeError("plot_gene_bar expects `values` as a dict mapping row IDs to values")

                mode = kwargs.get("mode", "categorical")
                missing_color = kwargs.get("missing_color", self._style["gene_bar_missing_color"])

                # Auto-place between dendrogram and matrix
                dendro_axes = self._style["dendro_axes"]
                gap = kwargs.get("gene_bar_gap", self._style["gene_bar_gap"])
                bar_w = kwargs.get("gene_bar_width", self._style["gene_bar_width"])

                # Default vertical span matches dendrogram span (which matches matrix span)
                x0 = dendro_axes[0] + dendro_axes[2] + gap
                bar_axes = kwargs.get("axes", [x0, dendro_axes[1], bar_w, dendro_axes[3]])

                ax_bar = fig.add_axes(bar_axes, frameon=False)
                ax_bar.set_xlim(0, 1)
                ax_bar.set_ylim(-0.5, n_rows - 0.5)
                ax_bar.invert_yaxis()
                ax_bar.set_xticks([])
                ax_bar.set_yticks([])

                # Authoritative row IDs in plotted order
                row_ids = self.matrix.df.index.to_numpy()[row_order]

                if mode == "categorical":
                    colors = kwargs.get("colors", None)
                    if colors is None or not isinstance(colors, dict):
                        raise TypeError("categorical gene_bar requires `colors` as a dict mapping category -> color")

                    for i, gid in enumerate(row_ids):
                        cat = values.get(gid, None)
                        c = colors.get(cat, missing_color)
                        ax_bar.add_patch(
                            plt.Rectangle((0.0, i - 0.5), 1.0, 1.0, facecolor=c, edgecolor="none")
                        )

                elif mode == "continuous":
                    cmap = plt.get_cmap(kwargs.get("cmap", "viridis"))
                    vals = np.array([values.get(gid, np.nan) for gid in row_ids], dtype=float)

                    finite = np.isfinite(vals)
                    if not np.any(finite):
                        # All missing: draw a uniform bar
                        for i in range(len(row_ids)):
                            ax_bar.add_patch(
                                plt.Rectangle((0.0, i - 0.5), 1.0, 1.0, facecolor=missing_color, edgecolor="none")
                            )
                    else:
                        vmin = kwargs.get("vmin", float(np.nanmin(vals[finite])))
                        vmax = kwargs.get("vmax", float(np.nanmax(vals[finite])))
                        if vmin == vmax:
                            vmax = vmin + 1e-12
                        norm = plt.Normalize(vmin=vmin, vmax=vmax)

                        for i, v in enumerate(vals):
                            c = missing_color if not np.isfinite(v) else cmap(norm(v))
                            ax_bar.add_patch(
                                plt.Rectangle((0.0, i - 0.5), 1.0, 1.0, facecolor=c, edgecolor="none")
                            )

                else:
                    raise ValueError("gene_bar mode must be 'categorical' or 'continuous'")

                continue
            if layer == "matrix":
                figsize = kwargs.get("figsize", None)
                if figsize is not None:
                    fig.set_size_inches(figsize[0], figsize[1], forward=True)

                # Allow users to override subplot margins cleanly
                subplots_adjust = kwargs.get("subplots_adjust", None)
                if subplots_adjust is not None:
                    fig.subplots_adjust(**subplots_adjust)

                cmap = kwargs.get("cmap", "viridis")
                extent = (-0.5, n_cols - 0.5, n_rows - 0.5, -0.5)
                center = kwargs.get("center", None)
                vmin_kw = kwargs.get("vmin", None)
                vmax_kw = kwargs.get("vmax", None)

                if center is not None:
                    vmin = np.nanmin(data) if vmin_kw is None else vmin_kw
                    vmax = np.nanmax(data) if vmax_kw is None else vmax_kw
                    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
                    imshow_vmin = None
                    imshow_vmax = None
                else:
                    norm = None
                    imshow_vmin = vmin_kw
                    imshow_vmax = vmax_kw

                ax.imshow(
                    data,
                    cmap=cmap,
                    norm=norm,
                    vmin=imshow_vmin,
                    vmax=imshow_vmax,
                    aspect="auto",
                    interpolation="nearest",
                    origin="upper",
                    extent=extent,
                )
                ax.set_xlim(-0.5, n_cols - 0.5)
                ax.set_ylim(n_rows - 0.5, -0.5)

                # ------------------------------------------------------------
                # Outer matrix frame (visual scaffold only)
                # ------------------------------------------------------------
                outer_lw = kwargs.get("outer_lw", 1.2)
                outer_color = kwargs.get("outer_color", "black")
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(outer_lw)
                    spine.set_color(outer_color)

                # ------------------------------------------------------------
                # Subtle minor row gridlines (render-only, non-semantic)
                # ------------------------------------------------------------
                show_minor_rows = kwargs.get("show_minor_rows", True)
                if show_minor_rows:
                    minor_row_step = kwargs.get("minor_row_step", 1)
                    minor_row_lw = kwargs.get("minor_row_lw", 0.15)
                    minor_row_alpha = kwargs.get("minor_row_alpha", 0.15)

                    for y in range(0, n_rows, minor_row_step):
                        ax.axhline(
                            y - 0.5,
                            color="black",
                            lw=minor_row_lw,
                            alpha=minor_row_alpha,
                            zorder=2,
                        )

            # Note: orientation="left" already handles axis direction.
            # Do NOT manually reverse x-limits or the dendrogram will be mirrored.
            elif layer == "matrix_axis_labels":
                xlabel = kwargs.get("xlabel", "")
                ylabel = kwargs.get("ylabel", "")
                fontsize = kwargs.get("fontsize", 12)
                fontweight = kwargs.get("fontweight", "normal")
                xlabel_pad = kwargs.get("xlabel_pad", 8)
                # Set x-label on the matrix axis as usual
                ax.set_xlabel(
                    xlabel,
                    fontsize=fontsize,
                    fontweight=fontweight,
                    labelpad=xlabel_pad,
                    color=kwargs.get("color", self._style.get("text_color", "black")),
                )
                # Do NOT set y-label on the matrix axis
                # Instead, if ylabel is non-empty, create a new axis to the right of the matrix for the y-label
                if isinstance(ylabel, str) and ylabel.strip():
                    # Get matrix axis position in figure coordinates
                    bbox = ax.get_position()
                    pad_frac = kwargs.get("ylabel_pad", self._style.get("ylabel_pad", 0.015))
                    # Compute new axes: just to the right of the matrix
                    x = bbox.x1 + pad_frac
                    y = bbox.y0
                    width = 0.015
                    height = bbox.height
                    ax_ylabel = ax.figure.add_axes([x, y, width, height], frameon=False)
                    ax_ylabel.set_xticks([])
                    ax_ylabel.set_yticks([])
                    for spine in ax_ylabel.spines.values():
                        spine.set_visible(False)
                    text_kwargs = {
                        "font": kwargs.get("font", "Helvetica"),
                        "fontsize": fontsize,
                        "color": kwargs.get("color", self._style.get("text_color", "black")),
                        "alpha": kwargs.get("alpha", 1.0),
                    }
                    ax_ylabel.text(
                        0.5,
                        0.5,
                        ylabel,
                        transform=ax_ylabel.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        **text_kwargs,
                    )

            elif layer == "row_ticks":
                # Render row tick labels
                labels = kwargs.get("labels", None)
                font_size = kwargs.get("fontsize", 9)
                max_labels = kwargs.get("max_labels", None)
                # Default: use matrix row index
                base_labels = labels if labels is not None else list(self.matrix.df.index)
                # Reorder according to row_order
                ordered_labels = np.array(base_labels)[row_order]
                n = len(ordered_labels)
                visible = np.ones(n, dtype=bool)
                if max_labels is not None and max_labels < n:
                    # Evenly spaced visible mask
                    idxs = np.linspace(0, n - 1, num=max_labels, dtype=int)
                    visible[:] = False
                    visible[idxs] = True
                ax.set_yticks(np.arange(n))
                ax.set_yticklabels(ordered_labels, fontsize=font_size)
                # Hide non-visible labels
                for tick, vis in zip(ax.get_yticklabels(), visible):
                    tick.set_visible(vis)

            elif layer == "col_ticks":
                labels = kwargs.get("labels", None)
                font_size = kwargs.get("fontsize", 9)
                rotation = kwargs.get("rotation", 90)
                max_labels = kwargs.get("max_labels", None)
                # Default: use matrix col index
                base_labels = labels if labels is not None else list(self.matrix.df.columns)
                if col_order is not None:
                    ordered_labels = np.array(base_labels)[col_order]
                else:
                    ordered_labels = np.array(base_labels)
                n = len(ordered_labels)
                visible = np.ones(n, dtype=bool)
                if max_labels is not None and max_labels < n:
                    idxs = np.linspace(0, n - 1, num=max_labels, dtype=int)
                    visible[:] = False
                    visible[idxs] = True
                ax.set_xticks(np.arange(n))
                ax.set_xticklabels(ordered_labels, fontsize=font_size, rotation=rotation)
                for tick, vis in zip(ax.get_xticklabels(), visible):
                    tick.set_visible(vis)

            elif layer == "dendrogram":
                # User-overridable dendrogram panel geometry/styling
                dendro_axes = kwargs.get("axes", self._style["dendro_axes"])
                dendro_color = kwargs.get("color", self._style["dendro_color"])
                dendro_lw = kwargs.get("linewidth", self._style["dendro_lw"])
                # Data-space padding to prevent edge clipping (render-only)
                DATA_PAD = kwargs.get("data_pad", 0.25)

                ax_dend = fig.add_axes(dendro_axes, frameon=False)

                # Only call dendrogram for geometry (do not use for order)
                if dendro is None:
                    dendro = dendrogram(
                        self.results.clusters.linkage_matrix,
                        orientation="left",
                        no_labels=True,
                        color_threshold=-1,
                        above_threshold_color="#888888",
                        distance_sort=False,
                        no_plot=True,
                    )

                # SciPy dendrogram uses y positions spaced by 10 units:
                # leaves are centered at 5, 15, 25, ...
                dendro_y_min = min(min(y) for y in dendro["icoord"])
                dendro_y_max = max(max(y) for y in dendro["icoord"])

                # Target matrix row coordinate space: [-0.5, n-0.5]
                target_y_min = -0.5
                target_y_max = n_rows - 0.5

                # Linear mapping from dendrogram y-space -> matrix row space
                scale = (target_y_max - target_y_min) / (dendro_y_max - dendro_y_min)
                offset = target_y_min - scale * dendro_y_min

                # Draw the PRECOMPUTED dendrogram geometry with mapped coordinates
                for icoord, dcoord in zip(dendro["icoord"], dendro["dcoord"]):
                    icoord_mapped = [scale * y + offset for y in icoord]
                    ax_dend.plot(dcoord, icoord_mapped, color=dendro_color, linewidth=dendro_lw)

                ax_dend.set_ylim(target_y_min - DATA_PAD, target_y_max + DATA_PAD)
                ax_dend.invert_yaxis()   # match imshow row orientation
                ax_dend.invert_xaxis()   # left-oriented dendrogram

                ax_dend.set_xticks([])
                ax_dend.set_yticks([])
                ax_dend.spines['top'].set_visible(False)
                ax_dend.spines['right'].set_visible(False)
                ax_dend.spines['bottom'].set_visible(False)
                ax_dend.spines['left'].set_visible(False)

            elif layer == "title":
                ax.set_title(
                    kwargs["title"],
                    fontsize=kwargs.get("fontsize", self._style.get("title_fontsize", 14)),
                    pad=kwargs.get("pad", self._style.get("title_pad", 15)),
                    color=kwargs.get("color", self._style.get("text_color", "black")),
                )

            elif layer == "cluster_labels":
                df = kwargs["df"]
                if not isinstance(df, pd.DataFrame):
                    raise TypeError("cluster_labels must be a pandas DataFrame.")
                if "cluster" not in df.columns or "label" not in df.columns:
                    raise ValueError("cluster_labels DataFrame must contain columns: 'cluster', 'label'.")

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

                boundary_color = kwargs.get("boundary_color", self._style["boundary_color"])
                boundary_lw = kwargs.get("boundary_lw", self._style["boundary_lw"])
                boundary_alpha = kwargs.get("boundary_alpha", self._style["boundary_alpha"])

                dendro_boundary_color = kwargs.get("dendro_boundary_color", self._style["dendro_boundary_color"])
                dendro_boundary_lw = kwargs.get("dendro_boundary_lw", self._style["dendro_boundary_lw"])
                dendro_boundary_alpha = kwargs.get("dendro_boundary_alpha", self._style["dendro_boundary_alpha"])

                # Draw cluster boundary lines (including top and bottom for polish)
                for cid, s, e in spans:
                    boundary = s - 0.5

                    ax.axhline(boundary, color=boundary_color, linewidth=boundary_lw, alpha=boundary_alpha)
                    if ax_dend is not None:
                        ax_dend.axhline(boundary, color=dendro_boundary_color, linewidth=dendro_boundary_lw, alpha=dendro_boundary_alpha)

                # Bottom border after last cluster
                last_e = spans[-1][2]
                boundary = last_e + 0.5
                ax.axhline(boundary, color=boundary_color, linewidth=boundary_lw, alpha=boundary_alpha)
                if ax_dend is not None:
                    ax_dend.axhline(boundary, color=dendro_boundary_color, linewidth=dendro_boundary_lw, alpha=dendro_boundary_alpha)

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
                LABEL_TEXT_PAD = kwargs.get("label_text_pad", self._style.get("label_bar_pad", 0.01))

                # --- Track ordering API (label_track_order) ---
                tracks = []
                for t in self._declared_label_tracks:
                    if t.get("enabled", True):
                        tracks.append(dict(t))
                        tracks[-1]["payload"] = dict(tracks[-1].get("payload", {}))

                label_track_order = self._label_track_order
                if label_track_order is not None:
                    # Validate type: must be tuple or list of str
                    if not isinstance(label_track_order, (list, tuple)):
                        raise TypeError("label_track_order must be a list or tuple of strings")
                    names = list(label_track_order)
                    if any(not isinstance(n, str) for n in names):
                        raise TypeError("label_track_order must be a list or tuple of strings")

                    # Ensure active track names are unique (ordering by name must be unambiguous)
                    active_names = [t.get("name") for t in tracks]
                    if any(n is None for n in active_names):
                        raise ValueError("All label-panel tracks must have a non-empty 'name'")
                    if len(set(active_names)) != len(active_names):
                        raise ValueError(f"Active label-panel track names must be unique. Got: {active_names}")

                    # Check for duplicates in requested ordering
                    if len(set(names)) != len(names):
                        raise ValueError("label_track_order contains duplicate track names")

                    # Unknown names are an error (explicit is better than silent)
                    available = set(active_names)
                    unknown = [n for n in names if n not in available]
                    if unknown:
                        raise ValueError(f"Unknown track(s) in label_track_order: {unknown}. Available tracks: {active_names}")

                    # Reorder: tracks in user order, then any omitted in original order
                    name_to_track = {t["name"]: t for t in tracks}
                    ordered = [name_to_track[n] for n in names]
                    omitted = [t for t in tracks if t["name"] not in names]
                    tracks = ordered + omitted
                # --- End track ordering API ---
                self._active_label_tracks = tracks

                # (2) Compute track geometry (rail layout)
                base_x = kwargs.get("label_x", self._style["label_x"])
                x_cursor = base_x + gutter_w
                for track in self._active_label_tracks:
                    x_cursor += track["left_pad"]
                    track["x0"] = x_cursor
                    track["x1"] = x_cursor + track["width"]
                    x_cursor = track["x1"] + track["right_pad"]
                label_text_x = x_cursor + LABEL_TEXT_PAD

                font = kwargs.get("font", 'Helvetica')
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
                for track in self._active_label_tracks:
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
                for track in self._active_label_tracks:
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

                    for track in self._active_label_tracks:
                        title = track.get("payload", {}).get("title", None)
                        if not title:
                            continue
                        x_center = (track.get("x0", 0.0) + track.get("x1", 0.0)) / 2.0
                        text_kwargs = {
                            "font": bar_kwargs.get("font", "Helvetica"),
                            "fontsize": bar_kwargs.get("fontsize", 10),
                            "color": bar_kwargs.get("color", self._style.get("text_color", "black")),
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
                                parts.append(f"p={pval:.2e}")
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
                        # Separator start: label_sep_xmin
                        xmin = kwargs.get("label_sep_xmin", self._style.get("label_sep_xmin", None))
                        xmax = kwargs.get("label_sep_xmax", self._style.get("label_sep_xmax", None))
                        if xmin is None:
                            xmin = label_text_x
                        if xmax is None:
                            xmax = 1.0
                        xmin = float(np.clip(xmin, 0.0, 1.0))
                        xmax = float(np.clip(xmax, 0.0, 1.0))
                        if xmin > xmax:
                            xmin, xmax = xmax, xmin
                        ax_lab.axhline(
                            s - 0.5,
                            xmin=xmin,
                            xmax=xmax,
                            color=sep_color,
                            linewidth=sep_lw,
                            alpha=sep_alpha,
                            zorder=0,
                        )

            elif layer == "sigbar_legend":
                cmap = plt.get_cmap(kwargs.get("sigbar_cmap", self._style.get("sigbar_cmap", "YlOrBr")))
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
            enabled_bars = [cb for cb in self._colorbars if cb.get("enabled", True)]
            if enabled_bars:
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

                N = len(enabled_bars)
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

                for i, cb in enumerate(enabled_bars):
                    r = i // ncols
                    c = i % ncols

                    x0 = strip_x0 + c * (cell_w + hpad)
                    y0 = strip_y0 + (nrows - 1 - r) * (cell_h + vpad)

                    ax_cb = fig.add_axes([x0, y0, cell_w, cell_h], frameon=True)

                    # Set border styling for colorbar axes
                    for spine in ax_cb.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(border_width)
                        spine.set_edgecolor(border_color)
                        spine.set_alpha(border_alpha)

                    ColorbarBase(
                        ax_cb,
                        cmap=cb["cmap"],
                        norm=cb["norm"],
                        orientation="horizontal",
                        ticks=cb.get("ticks", None),
                    )

                    ax_cb.tick_params(
                        axis="x",
                        labelsize=fontsize,
                        colors=text_color,
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
                                color=text_color,
                                labelpad=2,
                                fontname=font if font is not None else None,
                            )
                        else:
                            ax_cb.set_title(
                                label,
                                fontsize=fontsize,
                                color=text_color,
                                pad=2,
                                fontname=font if font is not None else None,
                            )
        ax.set_xticks([])
        ax.set_yticks([])
        # Attach layout metadata for advanced users
        self.layout_ = layout
        self.colorbar_layout_ = colorbar_layout
        self._fig = fig
        plt.show()

    def save(self, path: str, **kwargs) -> None:
        """
        Save the last rendered figure with correct background handling.
        """
        if self._fig is None:
            raise RuntimeError("Nothing to save — call show() first.")

        self._fig.savefig(
            path,
            facecolor=self._fig.get_facecolor(),
            **kwargs,
        )
