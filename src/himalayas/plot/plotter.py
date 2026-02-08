"""
himalayas/plot/plotter
~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, Hashable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt

from .renderers import (
    AxesRenderer,
    BoundaryRegistry,
    ClusterLabelsRenderer,
    ColorbarRenderer,
    DendrogramRenderer,
    MatrixRenderer,
    render_cluster_bar_track,
    SigbarLegendRenderer,
)
from .renderers.label_bar import render_label_bar_track
from .style import StyleConfig
from .track_layout import TrackLayoutManager
from ..core.results import Results


class Plotter:
    """
    Class for building layered, matrix-first plots from analysis results.
    """

    def __init__(self, results: Results) -> None:
        """
        Initializes the Plotter instance.

        Args:
            results (Results): Results object with matrix and cluster layout.

        Raises:
            AttributeError: If required results attributes are missing.
            ValueError: If results.matrix is None.
        """
        self.results = results
        # Validation
        if not hasattr(results, "matrix"):
            raise AttributeError("Plotter expects Results with a `.matrix` attribute")
        if not hasattr(results, "cluster_layout"):
            raise AttributeError("Plotter expects Results exposing `cluster_layout()`")
        if results.matrix is None:
            raise ValueError("Plotter expects Results with a non-null matrix")

        self.matrix = results.matrix
        # Declarative plot plan (ordered)
        self._layers = []
        # Declarative colorbar specs (global, figure-aligned)
        self._colorbars = []
        self._colorbar_layout = None
        # Label panel track layout manager
        self._track_layout = TrackLayoutManager()
        # Default styling config
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
        Declares a global colorbar explaining a visual encoding. Colorbars are figure-aligned
        and rendered in a strip below the matrix. They are not data-aligned and do not participate
        in row or cluster layout.

        Kwargs:
            name (str): Colorbar name.
            cmap: Colormap name or instance.
            norm: Matplotlib normalization instance.
            label (Optional[str]): Colorbar label text. Defaults to None.
            ticks (Optional[Sequence[float]]): Tick locations. Defaults to None.
            color (Optional[str]): Tick/label color. Defaults to None.

        Returns:
            Plotter: Self for chaining.

        Raises:
            ValueError: If name is missing or empty.
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
        tick_decimals: Optional[int] = None,
    ) -> Plotter:
        """
        Declares layout for the bottom colorbar strip.

        Kwargs:
            nrows (Optional[int]): Grid rows. If None, infer from ncols. Defaults to None.
            ncols (Optional[int]): Grid columns. If None, infer from nrows. Defaults to None.
            height (float): Total strip height (figure fraction). Defaults to 0.05.
            hpad (float): Horizontal spacing (figure fraction). Defaults to 0.01.
            vpad (float): Vertical spacing (figure fraction). Defaults to 0.01.
            gap (float): Gap from matrix to strip (figure fraction). Defaults to 0.02.
            border_color (Optional[str]): Border color. Defaults to None.
            border_width (float): Border line width. Defaults to 0.8.
            border_alpha (float): Border alpha. Defaults to 1.0.
            fontsize (Optional[float]): Tick/label font size. Defaults to None.
            font (Optional[str]): Tick/label font family. Defaults to None.
            color (Optional[str]): Tick/label color. Defaults to None.
            label_position (str): Label placement ("below" or "above"). Defaults to "below".
            tick_decimals (Optional[int]): Maximum decimals shown on colorbar ticks.
                Trailing zeros are trimmed. Defaults to None.

        Returns:
            Plotter: Self for chaining.

        Raises:
            TypeError: If tick_decimals is not an integer when provided.
            ValueError: If label_position is not "below" or "above", or tick_decimals is negative.
        """
        # Validation
        if label_position not in {"below", "above"}:
            raise ValueError("label_position must be 'below' or 'above'")
        if tick_decimals is not None:
            if isinstance(tick_decimals, bool) or not isinstance(tick_decimals, int):
                raise TypeError("tick_decimals must be an int >= 0 or None")
            if tick_decimals < 0:
                raise ValueError("tick_decimals must be >= 0")
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
            "tick_decimals": tick_decimals,
        }
        return self

    def set_background(self, color: str) -> Plotter:
        """
        Sets figure background color (used for display and save).

        Args:
            color (str): Background color.

        Returns:
            Plotter: Self for chaining.
        """
        self._background = color
        return self

    def set_label_track_order(self, order: Optional[Sequence[str]] = None) -> Plotter:
        """
        Sets the order of label-panel tracks in the label panel.
        Pass None to use default order. Otherwise, must be a list/tuple of unique strings.
        Defaults to None.

        Args:
            order (Optional[Sequence[str]]): Track names in order. Defaults to None.

        Returns:
            Plotter: Self for chaining.
        """
        self._track_layout.set_order(order)
        return self

    def plot_cluster_bar(self, name: str, **kwargs) -> Plotter:
        """
        Declares a cluster-level label-panel bar from internal cluster labels.
        Bars are derived from per-cluster p-values generated by Results.cluster_labels().
        Only kind='pvalue' is implemented.

        Args:
            name (str): Track name.

        Kwargs:
            kind (str): Value kind, currently only "pvalue" is supported. Defaults to "pvalue".
            width (float): Track width. Defaults to style sigbar_width.
            left_pad (float): Left padding. Defaults to 0.0.
            right_pad (float): Right padding. Defaults to 0.0.
            cmap (str): Colormap name. Defaults to style sigbar_cmap.
            norm: Matplotlib normalization instance. Defaults to None.
            alpha (float): Bar alpha. Defaults to style sigbar_alpha.
            enabled (bool): Whether to register the track. Defaults to True.
            title (str): Optional bar title. Defaults to None.

        Returns:
            Plotter: Self for chaining.

        Raises:
            NotImplementedError: If kind is not "pvalue".
        """
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
        # Enable cluster bar track registration
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
                    "cmap": cmap,
                    "norm": norm,
                    "alpha": alpha,
                    "title": title,
                },
            )
        return self

    def plot_sigbar_legend(self, **kwargs) -> Plotter:
        """
        Declares a significance bar legend. Explains the color mapping of the cluster-level significance bar
        (based on -log10(p)). Off by default.

        Kwargs:
            **kwargs: Renderer keyword arguments. Defaults to {}.

        Returns:
            Plotter: Self for chaining.
        """
        self._layers.append(("sigbar_legend", kwargs))
        return self

    def plot_matrix(self, **kwargs) -> Plotter:
        """
        Declares the main matrix heatmap layer.
        Stores parameters verbatim and interprets them at render time.

        Kwargs:
            **kwargs: Renderer keyword arguments. Defaults to {}.

        Returns:
            Plotter: Self for chaining.
        """
        self._layers.append(("matrix", kwargs))
        return self

    def plot_matrix_axis_labels(self, **kwargs) -> Plotter:
        """
        Declares axis labels for the matrix.

        Kwargs:
            xlabel (str): X-axis label text. Defaults to "".
            ylabel (str): Y-axis label text. Defaults to "".
            fontsize (float): Font size for both axis labels. Defaults to 12.
            font (str): Font family to apply to both axis labels. Defaults to None.
            color (str): Text color for both axis labels. Defaults to style text_color.
            **kwargs: Renderer keyword arguments. Defaults to {}.

        Returns:
            Plotter: Self for chaining.
        """
        self._layers.append(("matrix_axis_labels", kwargs))
        return self

    def plot_row_ticks(self, labels: Optional[Sequence[str]] = None, **kwargs) -> Plotter:
        """
        Declares row tick labels for the matrix.

        Args:
            labels (Optional[Sequence[str]]): Optional row labels. Defaults to None.

        Kwargs:
            position (str): "left" or "right". Defaults to "right".

        Returns:
            Plotter: Self for chaining.
        """
        self._layers.append(("row_ticks", {"labels": labels, **kwargs}))
        return self

    def plot_col_ticks(self, labels: Optional[Sequence[str]] = None, **kwargs) -> Plotter:
        """
        Declares column tick labels for the matrix.

        Args:
            labels (Optional[Sequence[str]]): Optional column labels. Defaults to None.

        Kwargs:
            position (str): "top" or "bottom". Defaults to "top".

        Returns:
            Plotter: Self for chaining.
        """
        self._layers.append(("col_ticks", {"labels": labels, **kwargs}))
        return self

    def plot_bar_labels(self, **kwargs) -> Plotter:
        """
        Declares titles for bar tracks (shown below bars in label panel).

        Kwargs:
            **kwargs: Renderer keyword arguments. Defaults to {}.

        Returns:
            Plotter: Self for chaining.
        """
        self._layers.append(("bar_labels", kwargs))
        return self

    def plot_dendrogram(self, **kwargs) -> Plotter:
        """
        Declares a dendrogram layer aligned to the matrix.

        Kwargs:
            data_pad (float): Data-space padding to prevent edge clipping in the dendrogram panel.
                Defaults to 0.25.

        Returns:
            Plotter: Self for chaining.
        """
        self._layers.append(("dendrogram", kwargs))
        return self

    def plot_label_bar(self, values: Mapping[Hashable, Any], **kwargs) -> Plotter:
        """
        Declares a single row-level annotation bar in the label panel. Preserves clustering and
        ordering because it is purely visual.

        Args:
            values (Mapping): Mapping from row identifier (matrix index label) to either:
                - categorical value (mode="categorical")
                - numeric value (mode="continuous")

        Kwargs:
            mode ({"categorical", "continuous"}): Rendering mode; drives color mapping. Defaults to "categorical".
            colors (dict): Category -> color mapping (categorical mode, required). Defaults to None.
            cmap (str): Matplotlib colormap name (continuous mode). Defaults to "viridis".
            vmin (float): Color scale minimum (continuous mode). Defaults to None.
            vmax (float): Color scale maximum (continuous mode). Defaults to None.
            missing_color (str): Color for missing values. Defaults to None.
            left_pad (float): Left padding for the label bar track. Defaults to 0.0.
            width (float): Track width. Defaults to style label_bar_width.
            right_pad (float): Right padding for the label bar track. Defaults to 0.0.

        Returns:
            Plotter: Self for chaining.

        Raises:
            TypeError: If values cannot be converted to a dict.
        """
        # Validation
        if not isinstance(values, dict):
            try:
                values = dict(values)
            except Exception as exc:
                raise TypeError("plot_label_bar expects values convertible to dict") from exc

        # Always register as an explicit track in the label panel
        kwargs.pop("placement", None)
        width = kwargs.pop("width", self._style.get("label_bar_width", 0.015))
        left_pad = kwargs.pop("left_pad", 0.0)
        right_pad = kwargs.pop("right_pad", 0.0)
        enabled = kwargs.get("enabled", True)
        if enabled:
            self._track_layout.register_track(
                name=kwargs.get("name", "label_bar"),
                kind="row",
                renderer=render_label_bar_track,
                left_pad=left_pad,
                width=width,
                right_pad=right_pad if right_pad is not None else 0.0,
                enabled=enabled,
                payload={**kwargs, "values": values, "title": kwargs.get("title", None)},
            )
        return self

    def plot_cluster_labels(
        self,
        *,
        overrides: Optional[dict[int, str]] = None,
        **kwargs,
    ) -> Plotter:
        """
        Declares cluster-level labels shown in the right panel.
        Labels are generated from attached Results by default.

        Kwargs:
            overrides (dict[int, str]): Per-cluster label overrides keyed by cluster id.
                Override labels do not change bar values.
            term_col (str): Term id column for label generation. Defaults to "term".
            cluster_col (str): Cluster id column for label generation. Defaults to "cluster".
            weight_col (str): Weight/p-value column for label generation. Defaults to "pval".
            label_mode (str): Label mode, one of {"top_term", "compressed"}.
                Defaults to "top_term".
            label_col (Optional[str]): Optional display-name column. Defaults to "term_name".
            summary_max_words (int): Maximum words for compressed label generation.
                Defaults to 6.
            max_words (Optional[int]): Maximum words in rendered display labels.
                Defaults to None.
            label_fields (tuple[str]): Fields to include: "label", "n", "p".
                Defaults to ("label", "n", "p").
            **kwargs: Renderer keyword arguments. Defaults to {}.

        Returns:
            Plotter: Self for chaining.
        """
        label_option_keys = {
            "term_col",
            "cluster_col",
            "weight_col",
            "label_mode",
            "label_col",
            "summary_max_words",
        }
        label_options = {}
        for key in tuple(kwargs):
            if key in label_option_keys:
                label_options[key] = kwargs.pop(key)
        if "summary_max_words" in label_options:
            label_options["max_words"] = label_options.pop("summary_max_words")

        self._layers.append(
            (
                "cluster_labels",
                {
                    "_label_options": label_options,
                    "overrides": overrides,
                    **kwargs,
                },
            )
        )
        return self

    def plot_title(self, title: str, **kwargs) -> Plotter:
        """
        Declares a plot title.

        Args:
            title (str): Title text.

        Kwargs:
            **kwargs: Renderer keyword arguments. Defaults to {}.

        Returns:
            Plotter: Self for chaining.
        """
        self._layers.append(("title", {"title": title, **kwargs}))
        return self

    def _render(self) -> None:
        """
        Renders the accumulated plot layers in declaration order.

        Raises:
            RuntimeError: If no plot layers are declared.
            ValueError: If layout orders do not match matrix dimensions.
            ValueError: If cluster-level tracks are declared without plot_cluster_labels().
            NotImplementedError: If a declared layer type is not supported.
        """
        # Validation
        if not self._layers:
            raise RuntimeError("No plot layers declared.")
        has_cluster_label_layer = any(layer == "cluster_labels" for layer, _ in self._layers)
        has_cluster_track = any(
            track.get("enabled", True) and track.get("kind") == "cluster"
            for track in self._track_layout.tracks
        )
        if has_cluster_track and not has_cluster_label_layer:
            raise ValueError(
                "plot_cluster_bar() requires plot_cluster_labels() in the same plotting chain."
            )
        # Consume authoritative geometry from Results
        layout = self.results.cluster_layout()
        # NOTE:
        #   - Layout.leaf_order controls row ordering (statistically meaningful)
        #   - Layout.col_order controls column ordering (visual only)
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

        # Derive dendrogram boundary styling from label panel settings
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

        # Render declared layers in order
        for layer, kwargs in self._layers:
            if layer == "matrix":
                figsize = kwargs.get("figsize", None)
                if figsize is not None:
                    fig.set_size_inches(figsize[0], figsize[1], forward=True)
                subplots_adjust = kwargs.get("subplots_adjust", None)
                if subplots_adjust is not None:
                    fig.subplots_adjust(**subplots_adjust)
                renderer = MatrixRenderer(**kwargs)
                renderer.render(
                    ax,
                    self.matrix,
                    layout,
                    self._style,
                    boundary_registry=boundary_registry,
                )

            # Note: orientation="left" already handles axis direction
            # Do NOT manually reverse x-limits or the dendrogram will be mirrored
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
                # Prepare cluster label renderer inputs and render label panel
                renderer_kwargs = dict(kwargs)
                label_options = renderer_kwargs.pop("_label_options", {})
                df = self.results.cluster_labels(**label_options)
                renderer = ClusterLabelsRenderer(df, **renderer_kwargs)
                renderer.render(
                    fig,
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
                # Consumed inside the cluster label panel; no direct rendering
                continue
            else:
                raise NotImplementedError(f"Unknown plot layer: {layer}")

        # Render bottom colorbar strip (global legends)
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
        """
        Checks whether the current figure handle is still open.

        Returns:
            bool: True if the figure exists and is open, False otherwise.
        """
        if self._fig is None:
            return False
        try:
            return self._fig.number in plt.get_fignums()
        except (AttributeError, RuntimeError, ValueError):
            return False

    def save(self, path: str, **kwargs) -> None:
        """
        Saves the last rendered figure with correct background handling.

        Args:
            path (str): Output path for the figure.

        Kwargs:
            **kwargs: Additional matplotlib savefig options. Defaults to {}.
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
        Shows the last rendered figure with correct background handling.
        """
        if self._fig is None or not self._figure_is_open():
            self._render()
        plt.show()
