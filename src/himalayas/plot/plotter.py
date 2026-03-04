"""
himalayas/plot/plotter
~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from os import PathLike
from typing import Any, Optional, Sequence, Union

import matplotlib.pyplot as plt
import pandas as pd

from .renderers import (
    AxesRenderer,
    BoundaryRegistry,
    ClusterLabelsRenderer,
    ColorbarRenderer,
    LabelLegendRenderer,
    DendrogramRenderer,
    MatrixRenderer,
    render_cluster_bar_track,
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
        # Declarative categorical label-legend specs (global, figure-aligned)
        self._label_legends = []
        self._label_legend_layout = None
        # Label panel track layout manager
        self._track_layout = TrackLayoutManager()
        # Default styling config
        self.style = StyleConfig()
        self._style = self.style
        # Figure-level configuration (explicit, opt-in)
        self._background = None
        self._fig = None
        # Render metadata (attached after first render)
        self.layout_ = None
        self.colorbar_layout_ = None
        self.label_legend_layout_ = None

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
        label_pad: float = 2.0,
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
            label_pad (float): Padding between colorbar and its label text (points).
                Defaults to 2.0.
            tick_decimals (Optional[int]): Maximum decimals shown on colorbar ticks.
                Trailing zeros are trimmed. Defaults to None.

        Returns:
            Plotter: Self for chaining.

        Raises:
            TypeError: If tick_decimals is not an integer when provided, or label_pad
                is not numeric.
            ValueError: If label_position is not "below" or "above", tick_decimals
                is negative, or label_pad is negative.
        """
        # Validation
        if label_position not in {"below", "above"}:
            raise ValueError("label_position must be 'below' or 'above'")
        if isinstance(label_pad, bool) or not isinstance(label_pad, (int, float)):
            raise TypeError("label_pad must be a float >= 0")
        label_pad = float(label_pad)
        if label_pad < 0:
            raise ValueError("label_pad must be >= 0")
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
            "label_pad": label_pad,
            "tick_decimals": tick_decimals,
        }
        return self

    def add_label_legend(
        self,
        *,
        name: str,
        title: Optional[str] = None,
        colors: Optional[Mapping[Any, Any]] = None,
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        row_pad: Optional[float] = None,
        col_pad: Optional[float] = None,
        show_only_present: bool = True,
    ) -> Plotter:
        """
        Declares a categorical legend block for a row-level label bar.
        The referenced bar must be declared with plot_label_bar(name=..., mode="categorical", ...).

        Kwargs:
            name (str): Label-bar name to explain.
            title (Optional[str]): Legend title override. Defaults to None.
            colors (Optional[Mapping[Any, Any]]): Optional category-to-color override.
                If None, colors are inferred from the label bar. Defaults to None.
            nrows (Optional[int]): Grid rows for legend items. If None, infer from ncols.
                Defaults to None.
            ncols (Optional[int]): Grid columns for legend items. If None, infer from nrows.
                Defaults to None.
            row_pad (Optional[float]): Vertical spacing between rows inside this legend
                block (block-relative fraction). If None, uses renderer default.
                Defaults to None.
            col_pad (Optional[float]): Minimum horizontal spacing between items inside
                each legend row (block-relative fraction). Items are left-packed and this
                value is used as fixed inter-item spacing. If None, uses renderer default.
                Defaults to None.
            show_only_present (bool): Whether to include only categories present in the
                plotted matrix rows. Defaults to True.

        Returns:
            Plotter: Self for chaining.

        Raises:
            ValueError: If options are invalid.
        """
        # Validation
        if not isinstance(name, str) or not name:
            raise ValueError("label legend `name` must be a non-empty string")
        if any(spec.get("name") == name for spec in self._label_legends):
            raise ValueError(f"label legend already declared for name {name!r}")
        if nrows is not None:
            nrows = int(nrows)
            if nrows <= 0:
                raise ValueError("nrows must be > 0")
        if ncols is not None:
            ncols = int(ncols)
            if ncols <= 0:
                raise ValueError("ncols must be > 0")
        if row_pad is not None:
            row_pad = float(row_pad)
            if row_pad < 0:
                raise ValueError("row_pad must be >= 0")
        if col_pad is not None:
            col_pad = float(col_pad)
            if col_pad < 0:
                raise ValueError("col_pad must be >= 0")

        self._label_legends.append(
            {
                "name": name,
                "title": title,
                "colors": colors,
                "nrows": nrows,
                "ncols": ncols,
                "row_pad": row_pad,
                "col_pad": col_pad,
                "show_only_present": bool(show_only_present),
            }
        )
        return self

    def plot_label_legends(
        self,
        *,
        height: float = 0.08,
        gap: float = 0.01,
        vpad: float = 0.01,
        title_pad: float = 2.0,
        swatch_scale: float = 0.75,
        fontsize: Optional[float] = None,
        font: Optional[str] = None,
        color: Optional[str] = None,
    ) -> Plotter:
        """
        Declares layout for categorical label-legend blocks beneath the colorbar strip.

        Kwargs:
            height (float): Total strip height (figure fraction). Defaults to 0.08.
            gap (float): Gap from the anchor strip above (figure fraction). Defaults to 0.01.
            vpad (float): Vertical spacing between legend blocks (figure fraction).
                Defaults to 0.01.
            title_pad (float): Padding between legend title and items (points).
                Defaults to 2.0.
            swatch_scale (float): Relative swatch size inside each legend cell.
                Defaults to 0.75.
            fontsize (Optional[float]): Legend font size. Defaults to None.
            font (Optional[str]): Legend font family. Defaults to None.
            color (Optional[str]): Legend text color. Defaults to None.

        Returns:
            Plotter: Self for chaining.

        Raises:
            ValueError: If geometric arguments are invalid.
        """
        height = float(height)
        gap = float(gap)
        vpad = float(vpad)
        title_pad = float(title_pad)
        swatch_scale = float(swatch_scale)
        # Validation
        if height <= 0:
            raise ValueError("height must be > 0")
        if gap < 0:
            raise ValueError("gap must be >= 0")
        if vpad < 0:
            raise ValueError("vpad must be >= 0")
        if title_pad < 0:
            raise ValueError("title_pad must be >= 0")
        if swatch_scale <= 0:
            raise ValueError("swatch_scale must be > 0")

        self._label_legend_layout = {
            "height": height,
            "gap": gap,
            "vpad": vpad,
            "title_pad": title_pad,
            "swatch_scale": swatch_scale,
            "fontsize": fontsize,
            "font": font,
            "color": color,
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

    def set_label_panel(
        self,
        *,
        axes: Optional[Sequence[float]] = None,
        track_x: Optional[float] = None,
        gutter_width: Optional[float] = None,
        gutter_color: Optional[str] = None,
        text_pad: Optional[float] = None,
    ) -> Plotter:
        """
        Configures label-panel geometry shared by cluster labels and bar tracks.

        Kwargs:
            axes (Optional[Sequence[float]]): Label panel axes [x, y, w, h]. Defaults to None.
            track_x (Optional[float]): X start (axes coordinates) for tracks. Defaults to None.
            gutter_width (Optional[float]): Width of the matrix-adjacent gutter. Defaults to None.
            gutter_color (Optional[str]): Color of the label-panel gutter. Defaults to None.
            text_pad (Optional[float]): Padding between tracks and label text. Defaults to None.

        Returns:
            Plotter: Self for chaining.

        Raises:
            TypeError: If axes is not a numeric sequence of length 4.
            ValueError: If gutter_width/text_pad are negative.
        """
        if axes is not None:
            if not isinstance(axes, (list, tuple)) or len(axes) != 4:
                raise TypeError("set_label_panel(axes=...) expects a sequence of length 4")
            self._style.set("label_axes", [float(v) for v in axes])
        if track_x is not None:
            self._style.set("label_x", float(track_x))
        if gutter_width is not None:
            gw = float(gutter_width)
            if gw < 0:
                raise ValueError("set_label_panel(gutter_width=...) must be >= 0")
            self._style.set("label_gutter_width", gw)
        if gutter_color is not None:
            self._style.set("label_gutter_color", str(gutter_color))
        if text_pad is not None:
            tp = float(text_pad)
            if tp < 0:
                raise ValueError("set_label_panel(text_pad=...) must be >= 0")
            self._style.set("label_bar_pad", tp)
        return self

    def _register_label_track(
        self,
        *,
        name: str,
        kind: str,
        renderer,
        left_pad: float,
        width: float,
        right_pad: float,
        enabled: bool,
        payload: Dict[str, Any],
    ) -> None:
        """
        Registers a label-panel track if enabled.

        Args:
            name (str): Track name.
            kind (str): Track kind ("cluster" or "row").
            renderer: Track renderer callable.
            left_pad (float): Left padding.
            width (float): Track width.
            right_pad (float): Right padding.
            enabled (bool): Whether to register.
            payload (Dict[str, Any]): Renderer payload.
        """
        if not enabled:
            return
        self._track_layout.register_track(
            name=name,
            kind=kind,
            renderer=renderer,
            left_pad=left_pad,
            width=width,
            right_pad=right_pad,
            enabled=True,
            payload=payload,
        )

    def _has_track_kind(self, kind: str) -> bool:
        """
        Checks whether any enabled track of a given kind is registered.

        Args:
            kind (str): Track kind to check.

        Returns:
            bool: True if at least one enabled track matches kind.
        """
        return any(
            track.get("enabled", True) and track.get("kind") == kind
            for track in self._track_layout.tracks
        )

    def _collect_layer_kwargs(
        self,
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Collects frequently used layer kwargs, using last-one-wins semantics.

        Returns:
            tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
                Matrix kwargs, cluster-label kwargs, and bar-label kwargs.
        """
        matrix_kwargs = None
        cluster_label_kwargs = None
        bar_label_kwargs = None
        for layer_name, kwargs in self._layers:
            if layer_name == "matrix":
                matrix_kwargs = kwargs
            elif layer_name == "cluster_labels":
                cluster_label_kwargs = kwargs
            elif layer_name == "bar_labels":
                bar_label_kwargs = kwargs
        return matrix_kwargs, cluster_label_kwargs, bar_label_kwargs

    def _resolve_label_legend_specs(self) -> list[Dict[str, Any]]:
        """
        Resolves categorical label-legend specs against declared row-level label tracks.

        Returns:
            list[Dict[str, Any]]: Normalized label-legend specs.

        Raises:
            ValueError: If referenced tracks are invalid for categorical legends.
        """
        if not self._label_legends:
            return []

        # Index enabled tracks by name and collect row-track names for legend validation.
        enabled_tracks = [t for t in self._track_layout.tracks if t.get("enabled", True)]
        track_by_name = {track["name"]: track for track in enabled_tracks}
        row_track_names = [t["name"] for t in enabled_tracks if t.get("kind") == "row"]
        available_row_tracks = sorted(row_track_names)
        resolved = []
        for spec in self._label_legends:
            name = spec["name"]
            track = track_by_name.get(name, None)
            # Validation
            if track is None:
                raise ValueError(
                    "Unknown label-legend name "
                    f"{name!r}. Available row label bars: {available_row_tracks}"
                )
            if track.get("kind") != "row":
                raise ValueError(f"label legend {name!r} must reference a row-level label bar")

            payload = track.get("payload", {})
            mode = payload.get("mode", "categorical")
            # Validation
            if mode != "categorical":
                raise ValueError(
                    f"label legend {name!r} must use plot_label_bar(..., mode='categorical')"
                )

            colors = spec.get("colors", None)
            if colors is None:
                colors = payload.get("colors", None)
            # Validation
            if not isinstance(colors, Mapping) or not colors:
                raise ValueError(f"label legend {name!r} requires a non-empty `colors` mapping")

            # Resolve visible categories and legend title from bar payload plus legend overrides.
            values = payload.get("values", {})
            row_ids = list(self.matrix.df.index)
            present_categories = {values.get(row_id, None) for row_id in row_ids}
            show_only_present = bool(spec.get("show_only_present", True))
            title = spec.get("title", None)
            if title is None or str(title) == "":
                title = payload.get("title", None)
            if title is None or str(title) == "":
                title = name

            items = [
                {
                    "value": category,
                    "label": str(category),
                    "color": color,
                }
                for category, color in colors.items()
                if (category in present_categories) or (not show_only_present)
            ]
            resolved.append(
                {
                    "name": name,
                    "title": str(title),
                    "items": items,
                    "nrows": spec.get("nrows"),
                    "ncols": spec.get("ncols"),
                    "row_pad": spec.get("row_pad"),
                    "col_pad": spec.get("col_pad"),
                }
            )

        return resolved

    def _render_label_panel(
        self,
        fig,
        layout,
        *,
        bar_kwargs: Optional[Dict[str, Any]],
        cluster_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Renders the shared label panel with optional cluster labels.

        Args:
            fig: Matplotlib figure.
            layout: Cluster layout object.
            bar_kwargs (Optional[Dict[str, Any]]): Bar-label renderer kwargs.
            cluster_kwargs (Optional[Dict[str, Any]]): Cluster-label layer kwargs.
                If None, renders tracks only without cluster text.
        """
        if cluster_kwargs is None:
            # Keep renderer input schema aligned with Results.cluster_labels() output.
            df = pd.DataFrame(
                columns=["cluster", "label", "pval", "qval", "score", "n", "term", "fe"]
            )
            renderer = ClusterLabelsRenderer(df, skip_unlabeled=True)
        else:
            renderer_kwargs = dict(cluster_kwargs)
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

    def plot_cluster_bar(self, name: str, **kwargs) -> Plotter:
        """
        Declares a cluster-level label-panel bar from internal cluster labels.
        Bars are derived from per-cluster ranking scores generated by Results.cluster_labels().

        Args:
            name (str): Track name.

        Kwargs:
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
        """
        if "kind" in kwargs:
            raise TypeError("plot_cluster_bar() got an unexpected keyword argument 'kind'")
        width = kwargs.get("width", self._style.get("sigbar_width", 0.015))
        left_pad = kwargs.get("left_pad", 0.0)
        right_pad = kwargs.get("right_pad", 0.0)
        cmap = kwargs.get("cmap", self._style.get("sigbar_cmap", "YlOrBr"))
        norm = kwargs.get("norm", None)
        alpha = kwargs.get("alpha", self._style.get("sigbar_alpha", 0.9))
        enabled = kwargs.get("enabled", True)
        title = kwargs.get("title", None)
        self._register_label_track(
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
            xlabel_pad (float): Padding between x-axis label and axis (points).
                Defaults to renderer default.
            ylabel_pad (float): Padding between matrix and external y-label axis
                (figure fraction). Defaults to style ylabel_pad.
            **kwargs: Renderer keyword arguments. Defaults to {}.

        Returns:
            Plotter: Self for chaining.
        """
        if "xlabel_pad" in kwargs and kwargs["xlabel_pad"] is not None:
            kwargs["xlabel_pad"] = float(kwargs["xlabel_pad"])
        if "ylabel_pad" in kwargs and kwargs["ylabel_pad"] is not None:
            kwargs["ylabel_pad"] = float(kwargs["ylabel_pad"])
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
            pad (float): Vertical offset from track to title text (points).
                Defaults to renderer default.
            **kwargs: Renderer keyword arguments. Defaults to {}.

        Returns:
            Plotter: Self for chaining.
        """
        if "pad" in kwargs and kwargs["pad"] is not None:
            kwargs["pad"] = float(kwargs["pad"])
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
        if "placement" in kwargs:
            raise TypeError("plot_label_bar() got an unexpected keyword argument 'placement'")
        width = kwargs.pop("width", self._style.get("label_bar_width", 0.015))
        left_pad = kwargs.pop("left_pad", 0.0)
        right_pad = kwargs.pop("right_pad", 0.0)
        enabled = kwargs.get("enabled", True)
        self._register_label_track(
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
        overrides: Optional[Dict[int, str]] = None,
        **kwargs,
    ) -> Plotter:
        """
        Declares cluster-level labels shown in the right panel.
        Labels are generated from attached Results by default.

        Kwargs:
            overrides (Dict[int, str]): Per-cluster label overrides keyed by cluster id.
                Override labels do not change bar values.
            rank_by (str): Ranking statistic for representative terms, one of {"p", "q"}.
                Defaults to "p".
            label_mode (str): Label mode, one of {"top_term", "compressed"}.
                Defaults to "top_term".
            max_words (Optional[int]): Maximum words in rendered display labels.
                Defaults to None.
            label_fields (Optional[tuple[str]]): Fields to include: "label", "n", "p", "q", "fe".
                If None, suppresses base label/stat text.
                Defaults to ("label", "n", "p").
            label_prefix (Optional[str]): Optional prefix mode for display labels.
                Supported values are None and "cid". Defaults to None.
            **kwargs: Renderer keyword arguments. Defaults to {}.

        Returns:
            Plotter: Self for chaining.

        Raises:
            TypeError: If unknown keyword arguments are provided.
        """
        label_option_keys = {
            "rank_by",
            "label_mode",
            "max_words",
        }
        renderer_option_keys = {
            "font",
            "fontsize",
            "color",
            "alpha",
            "skip_unlabeled",
            "label_fields",
            "label_prefix",
            "placeholder_text",
            "placeholder_color",
            "placeholder_alpha",
            "label_sep_xmin",
            "label_sep_xmax",
            "label_sep_color",
            "label_sep_lw",
            "label_sep_alpha",
            "boundary_color",
            "boundary_lw",
            "boundary_alpha",
            "dendro_boundary_color",
            "dendro_boundary_lw",
            "dendro_boundary_alpha",
            "omit_words",
            "wrap_text",
            "wrap_width",
            "overflow",
        }
        allowed_keys = label_option_keys | renderer_option_keys
        unknown_keys = sorted(set(kwargs) - allowed_keys)
        if unknown_keys:
            unknown_str = ", ".join(repr(k) for k in unknown_keys)
            raise TypeError(
                f"plot_cluster_labels() got unexpected keyword argument(s): {unknown_str}"
            )

        label_options = {}
        for key in tuple(kwargs):
            if key in label_option_keys:
                label_options[key] = kwargs[key]
                if key != "max_words":
                    kwargs.pop(key)

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
            pad (float): Padding between title text and matrix axis (points).
                Defaults to style title_pad.
            **kwargs: Renderer keyword arguments. Defaults to {}.

        Returns:
            Plotter: Self for chaining.
        """
        if "pad" in kwargs and kwargs["pad"] is not None:
            kwargs["pad"] = float(kwargs["pad"])
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
        has_row_track = self._has_track_kind("row")
        has_cluster_track = self._has_track_kind("cluster")
        if has_cluster_track and not has_cluster_label_layer:
            raise ValueError(
                "plot_cluster_bar() requires plot_cluster_labels() in the same plotting chain."
            )
        matrix_kwargs, cluster_boundary_kwargs, bar_kwargs = self._collect_layer_kwargs()
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
                self._render_label_panel(fig, layout, bar_kwargs=bar_kwargs, cluster_kwargs=kwargs)
            elif layer == "bar_labels":
                # Consumed inside the cluster label panel; no direct rendering
                continue
            else:
                raise NotImplementedError(f"Unknown plot layer: {layer}")
        # Row-level label tracks can render without cluster label text.
        if has_row_track and not has_cluster_label_layer:
            self._render_label_panel(fig, layout, bar_kwargs=bar_kwargs)

        # Render bottom colorbar strip (global legends)
        colorbar_layout = None
        if self._colorbars:
            renderer = ColorbarRenderer(self._colorbars, self._colorbar_layout)
            colorbar_layout = renderer.render(fig, ax, self._style)
        # Render categorical label-legend blocks beneath colorbars/matrix.
        label_legend_layout = None
        if self._label_legends:
            specs = self._resolve_label_legend_specs()
            renderer = LabelLegendRenderer(specs, self._label_legend_layout)
            label_legend_layout = renderer.render(
                fig,
                ax,
                self._style,
                colorbar_layout=colorbar_layout,
            )
        # Matrix axis never owns ticks; always keep clean
        ax.set_xticks([])
        ax.set_yticks([])
        # Attach layout metadata for advanced users
        self.layout_ = layout
        self.colorbar_layout_ = colorbar_layout
        self.label_legend_layout_ = label_legend_layout
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

    def save(self, path: Union[str, PathLike[str]], **kwargs: Any) -> None:
        """
        Saves the last rendered figure with correct background handling.

        Args:
            path (Union[str, PathLike[str]]): Output path for the figure.

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
