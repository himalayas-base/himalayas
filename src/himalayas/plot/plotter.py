"""
himalayas/plot/plotter
~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from os import PathLike
from typing import Any, Optional, Sequence, Union, cast

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
        # Declarative plot plan (ordered).
        self._layers = []
        # Declarative colorbar specs (global, figure-aligned).
        self._colorbars = []
        self._colorbar_layout = None
        # Declarative categorical label-legend specs (global, figure-aligned).
        self._label_legends = []
        self._label_legend_layout = None
        # Label panel track layout manager.
        self._track_layout = TrackLayoutManager()
        # Default styling config.
        self.style = StyleConfig()
        self._style = self.style
        # Figure-level configuration (explicit, opt-in).
        self._background = None
        self._fig = None
        # Render metadata (attached after first render).
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
        renderer: Callable[..., None],
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

    def plot_cluster_bar(
        self,
        name: str,
        *,
        width: Optional[float] = None,
        left_pad: float = 0.0,
        right_pad: float = 0.0,
        cmap: Optional[str] = None,
        norm: Any = None,
        alpha: Optional[float] = None,
        enabled: bool = True,
        title: Optional[str] = None,
    ) -> Plotter:
        """
        Declares a cluster-level label-panel bar from internal cluster labels.
        Bars are derived from per-cluster ranking scores generated by Results.cluster_labels().

        Args:
            name (str): Track name.

        Kwargs:
            width (Optional[float]): Track width (figure fraction). Defaults to style sigbar_width.
            left_pad (float): Left padding (figure fraction). Defaults to 0.0.
            right_pad (float): Right padding (figure fraction). Defaults to 0.0.
            cmap (Optional[str]): Colormap name. Defaults to style sigbar_cmap.
            norm: Matplotlib normalization instance. Defaults to None.
            alpha (Optional[float]): Bar opacity. Defaults to style sigbar_alpha.
            enabled (bool): Whether to register the track. Defaults to True.
            title (Optional[str]): Optional bar title shown below the track. Defaults to None.

        Returns:
            Plotter: Self for chaining.
        """
        self._register_label_track(
            name=name,
            kind="cluster",
            renderer=render_cluster_bar_track,
            left_pad=left_pad,
            width=(
                width if width is not None else cast(float, self._style.get("sigbar_width", 0.015))
            ),
            right_pad=right_pad,
            enabled=enabled,
            payload={
                "cmap": cmap if cmap is not None else self._style.get("sigbar_cmap", "YlOrBr"),
                "norm": norm,
                "alpha": alpha if alpha is not None else self._style.get("sigbar_alpha", 0.9),
                "title": title,
            },
        )
        return self

    def plot_matrix(
        self,
        *,
        cmap: str = "viridis",
        center: Optional[float] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show_minor_rows: bool = True,
        minor_row_step: int = 1,
        minor_row_lw: float = 0.15,
        minor_row_alpha: float = 0.15,
        outer_lw: float = 1.2,
        outer_color: str = "black",
        gutter_color: Optional[str] = None,
        figsize: Optional[tuple[float, float]] = None,
        subplots_adjust: Optional[dict[str, float]] = None,
    ) -> Plotter:
        """
        Declares the main matrix heatmap layer.

        Kwargs:
            cmap (str): Colormap name. Defaults to "viridis".
            center (Optional[float]): Center value for diverging normalization. Defaults to None.
            vmin (Optional[float]): Minimum color scale value. Defaults to None.
            vmax (Optional[float]): Maximum color scale value. Defaults to None.
            show_minor_rows (bool): Whether to draw minor row grid lines. Defaults to True.
            minor_row_step (int): Row step for minor grid lines. Defaults to 1.
            minor_row_lw (float): Minor grid line width. Defaults to 0.15.
            minor_row_alpha (float): Minor grid line opacity. Defaults to 0.15.
            outer_lw (float): Outer border line width. Defaults to 1.2.
            outer_color (str): Outer border color. Defaults to "black".
            gutter_color (Optional[str]): Background gutter color behind the matrix. Defaults to None.
            figsize (Optional[tuple[float, float]]): Figure size override in inches (width, height).
                Defaults to None.
            subplots_adjust (Optional[dict[str, float]]): Override for figure subplot spacing.
                Defaults to None.

        Returns:
            Plotter: Self for chaining.
        """
        layer_kwargs: dict[str, Any] = {
            "cmap": cmap,
            "show_minor_rows": show_minor_rows,
            "minor_row_step": minor_row_step,
            "minor_row_lw": minor_row_lw,
            "minor_row_alpha": minor_row_alpha,
            "outer_lw": outer_lw,
            "outer_color": outer_color,
        }
        if center is not None:
            layer_kwargs["center"] = center
        if vmin is not None:
            layer_kwargs["vmin"] = vmin
        if vmax is not None:
            layer_kwargs["vmax"] = vmax
        if gutter_color is not None:
            layer_kwargs["gutter_color"] = gutter_color
        if figsize is not None:
            layer_kwargs["figsize"] = figsize
        if subplots_adjust is not None:
            layer_kwargs["subplots_adjust"] = subplots_adjust
        self._layers.append(("matrix", layer_kwargs))
        return self

    def plot_matrix_axis_labels(
        self,
        *,
        xlabel: str = "",
        ylabel: str = "",
        fontsize: Optional[float] = None,
        fontweight: str = "normal",
        font: Optional[str] = None,
        color: Optional[str] = None,
        alpha: float = 1.0,
        xlabel_pad: Optional[float] = None,
        ylabel_pad: Optional[float] = None,
        **kwargs: Any,
    ) -> Plotter:
        """
        Declares axis labels for the matrix.

        Kwargs:
            xlabel (str): X-axis label text. Defaults to "".
            ylabel (str): Y-axis label text. Defaults to "".
            fontsize (Optional[float]): Font size for both axis labels (points). Defaults to 12.
            fontweight (str): Font weight for both axis labels. Defaults to "normal".
            font (Optional[str]): Font family to apply to both axis labels. Defaults to None.
            color (Optional[str]): Text color for both axis labels. Defaults to style text_color.
            alpha (float): Text opacity for both axis labels. Defaults to 1.0.
            xlabel_pad (Optional[float]): Padding between x-axis label and axis (points).
                Defaults to 8.
            ylabel_pad (Optional[float]): Padding between matrix and external y-label axis
                (figure fraction). Defaults to style ylabel_pad.
            **kwargs: Renderer keyword arguments. Defaults to {}.

        Returns:
            Plotter: Self for chaining.
        """
        if xlabel_pad is not None:
            xlabel_pad = float(xlabel_pad)
        if ylabel_pad is not None:
            ylabel_pad = float(ylabel_pad)
        layer_kwargs: dict[str, Any] = {
            "xlabel": xlabel,
            "ylabel": ylabel,
            "fontweight": fontweight,
            "alpha": alpha,
        }
        if fontsize is not None:
            layer_kwargs["fontsize"] = fontsize
        if font is not None:
            layer_kwargs["font"] = font
        if color is not None:
            layer_kwargs["color"] = color
        if xlabel_pad is not None:
            layer_kwargs["xlabel_pad"] = xlabel_pad
        if ylabel_pad is not None:
            layer_kwargs["ylabel_pad"] = ylabel_pad
        self._layers.append(("matrix_axis_labels", {**layer_kwargs, **kwargs}))
        return self

    def plot_row_ticks(
        self,
        labels: Optional[Sequence[str]] = None,
        *,
        fontsize: Optional[float] = None,
        max_labels: Optional[int] = None,
        position: str = "right",
        pad: Optional[float] = None,
        **kwargs: Any,
    ) -> Plotter:
        """
        Declares row tick labels for the matrix.

        Args:
            labels (Optional[Sequence[str]]): Optional row labels. Defaults to None.

        Kwargs:
            fontsize (Optional[float]): Tick-label font size (points). Defaults to 9.
            max_labels (Optional[int]): Maximum number of labels to display, evenly sampled.
                If None, all labels are shown. Defaults to None.
            position (str): Tick-label placement, one of {"left", "right"}. Defaults to "right".
            pad (Optional[float]): Tick-label padding in points. Defaults to Matplotlib default.
            **kwargs: Renderer keyword arguments. Defaults to {}.

        Returns:
            Plotter: Self for chaining.
        """
        if pad is not None:
            pad = float(pad)
        layer_kwargs: dict[str, Any] = {
            "labels": labels,
            "max_labels": max_labels,
            "position": position,
        }
        if fontsize is not None:
            layer_kwargs["fontsize"] = fontsize
        if pad is not None:
            layer_kwargs["pad"] = pad
        self._layers.append(("row_ticks", {**layer_kwargs, **kwargs}))
        return self

    def plot_col_ticks(
        self,
        labels: Optional[Sequence[str]] = None,
        *,
        fontsize: Optional[float] = None,
        max_labels: Optional[int] = None,
        position: str = "top",
        rotation: float = 90,
        pad: Optional[float] = None,
        **kwargs: Any,
    ) -> Plotter:
        """
        Declares column tick labels for the matrix.

        Args:
            labels (Optional[Sequence[str]]): Optional column labels. Defaults to None.

        Kwargs:
            fontsize (Optional[float]): Tick-label font size (points). Defaults to 9.
            max_labels (Optional[int]): Maximum number of labels to display, evenly sampled.
                If None, all labels are shown. Defaults to None.
            position (str): Tick-label placement, one of {"top", "bottom"}. Defaults to "top".
            rotation (float): Tick-label rotation in degrees. Defaults to 90.
            pad (Optional[float]): Tick-label padding in points. Defaults to Matplotlib default.
            **kwargs: Renderer keyword arguments. Defaults to {}.

        Returns:
            Plotter: Self for chaining.
        """
        if pad is not None:
            pad = float(pad)
        layer_kwargs: dict[str, Any] = {
            "labels": labels,
            "max_labels": max_labels,
            "position": position,
            "rotation": rotation,
        }
        if fontsize is not None:
            layer_kwargs["fontsize"] = fontsize
        if pad is not None:
            layer_kwargs["pad"] = pad
        self._layers.append(("col_ticks", {**layer_kwargs, **kwargs}))
        return self

    def plot_bar_labels(
        self,
        *,
        fontsize: Optional[float] = None,
        font: Optional[str] = None,
        color: Optional[str] = None,
        alpha: float = 1.0,
        pad: Optional[float] = None,
        rotation: float = 0,
        **kwargs: Any,
    ) -> Plotter:
        """
        Declares titles for bar tracks (shown below bars in label panel).

        Kwargs:
            fontsize (Optional[float]): Title font size (points). Defaults to 10.
            font (Optional[str]): Font family for title text. Defaults to "Helvetica".
            color (Optional[str]): Title text color. Defaults to style text_color.
            alpha (float): Title text opacity. Defaults to 1.0.
            pad (Optional[float]): Vertical offset from track to title text (points).
                Defaults to 2.
            rotation (float): Title text rotation in degrees. Defaults to 0.
            **kwargs: Renderer keyword arguments. Defaults to {}.

        Returns:
            Plotter: Self for chaining.
        """
        if pad is not None:
            pad = float(pad)
        layer_kwargs: dict[str, Any] = {"alpha": alpha, "rotation": rotation}
        if fontsize is not None:
            layer_kwargs["fontsize"] = fontsize
        if font is not None:
            layer_kwargs["font"] = font
        if color is not None:
            layer_kwargs["color"] = color
        if pad is not None:
            layer_kwargs["pad"] = pad
        self._layers.append(("bar_labels", {**layer_kwargs, **kwargs}))
        return self

    def plot_dendrogram(
        self,
        *,
        axes: Optional[Sequence[float]] = None,
        color: Optional[str] = None,
        linewidth: Optional[float] = None,
        data_pad: float = 0.25,
    ) -> Plotter:
        """
        Declares a dendrogram layer aligned to the matrix.

        Kwargs:
            axes (Optional[Sequence[float]]): Axes position [x0, y0, width, height] in figure
                fraction. Defaults to style dendro_axes.
            color (Optional[str]): Dendrogram line color. Defaults to style dendro_color.
            linewidth (Optional[float]): Dendrogram line width. Defaults to style dendro_lw.
            data_pad (float): Data-space padding to prevent edge clipping in the dendrogram panel.
                Defaults to 0.25.

        Returns:
            Plotter: Self for chaining.
        """
        layer_kwargs: dict[str, Any] = {"data_pad": data_pad}
        if axes is not None:
            layer_kwargs["axes"] = axes
        if color is not None:
            layer_kwargs["color"] = color
        if linewidth is not None:
            layer_kwargs["linewidth"] = linewidth
        self._layers.append(("dendrogram", layer_kwargs))
        return self

    def plot_label_bar(
        self,
        values: Mapping[Hashable, Any],
        *,
        name: str = "label_bar",
        mode: str = "categorical",
        colors: Optional[dict] = None,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        missing_color: Optional[str] = None,
        width: Optional[float] = None,
        left_pad: float = 0.0,
        right_pad: float = 0.0,
        enabled: bool = True,
        title: Optional[str] = None,
    ) -> Plotter:
        """
        Declares a single row-level annotation bar in the label panel. Preserves clustering and
        ordering because it is purely visual.

        Args:
            values (Mapping): Mapping from row identifier (matrix index label) to either:
                - categorical value (mode="categorical")
                - numeric value (mode="continuous")

        Kwargs:
            name (str): Track name. Defaults to "label_bar".
            mode ({"categorical", "continuous"}): Rendering mode; drives color mapping.
                Defaults to "categorical".
            colors (Optional[dict]): Category -> color mapping (categorical mode, required).
                Defaults to None.
            cmap (str): Matplotlib colormap name (continuous mode). Defaults to "viridis".
            vmin (Optional[float]): Color scale minimum (continuous mode). Defaults to None.
            vmax (Optional[float]): Color scale maximum (continuous mode). Defaults to None.
            missing_color (Optional[str]): Color for missing values. Defaults to None.
            width (Optional[float]): Track width (figure fraction). Defaults to style label_bar_width.
            left_pad (float): Left padding (figure fraction). Defaults to 0.0.
            right_pad (float): Right padding (figure fraction). Defaults to 0.0.
            enabled (bool): Whether to register the track. Defaults to True.
            title (Optional[str]): Optional bar title shown below the track. Defaults to None.

        Returns:
            Plotter: Self for chaining.

        Raises:
            TypeError: If values cannot be converted to a dict.
        """
        if not isinstance(values, dict):
            try:
                values = dict(values)
            except Exception as exc:
                raise TypeError("plot_label_bar expects values convertible to dict") from exc
        self._register_label_track(
            name=name,
            kind="row",
            renderer=render_label_bar_track,
            left_pad=left_pad,
            width=(
                width
                if width is not None
                else cast(float, self._style.get("label_bar_width", 0.015))
            ),
            right_pad=right_pad,
            enabled=enabled,
            payload={
                "values": values,
                "mode": mode,
                "colors": colors,
                "cmap": cmap,
                "vmin": vmin,
                "vmax": vmax,
                "missing_color": missing_color,
                "title": title,
            },
        )
        return self

    def plot_cluster_labels(
        self,
        *,
        overrides: Optional[dict[int, str]] = None,
        fontsize: Optional[float] = None,
        color: Optional[str] = None,
        font: Optional[str] = None,
        alpha: Optional[float] = None,
        # Label generation
        rank_by: str = "p",
        label_mode: str = "top_term",
        max_words: Optional[int] = None,
        # Label content
        label_fields: Optional[Sequence[str]] = None,
        label_prefix: Optional[str] = None,
        # Unlabeled clusters
        skip_unlabeled: bool = False,
        placeholder_text: Optional[str] = None,
        placeholder_color: Optional[str] = None,
        placeholder_alpha: Optional[float] = None,
        # Separator lines
        label_sep_xmin: Optional[float] = None,
        label_sep_xmax: Optional[float] = None,
        label_sep_color: Optional[str] = None,
        label_sep_lw: Optional[float] = None,
        label_sep_alpha: Optional[float] = None,
        # Cluster boundaries
        boundary_color: Optional[str] = None,
        boundary_lw: Optional[float] = None,
        boundary_alpha: Optional[float] = None,
        dendro_boundary_color: Optional[str] = None,
        dendro_boundary_lw: Optional[float] = None,
        dendro_boundary_alpha: Optional[float] = None,
        # Text formatting
        omit_words: Optional[Sequence[str]] = None,
        wrap_text: bool = True,
        wrap_width: Optional[int] = None,
        overflow: str = "wrap",
    ) -> Plotter:
        """
        Declares cluster-level labels shown in the right panel.
        Labels are generated from attached Results by default.

        Kwargs:
            overrides (Optional[dict[int, str]]): Per-cluster label overrides keyed by cluster id.
                Override labels do not change bar values. Defaults to None.
            fontsize (Optional[float]): Label font size (points). Defaults to None.
            color (Optional[str]): Label text color. Defaults to None.
            font (Optional[str]): Font family for label text. Defaults to None.
            alpha (Optional[float]): Label text opacity. Defaults to None.

            Label generation:
            rank_by (str): Ranking statistic for representative terms, one of {"p", "q"}.
                Defaults to "p".
            label_mode (str): Label composition mode, one of {"top_term", "compressed"}.
                Defaults to "top_term".
            max_words (Optional[int]): Maximum words in rendered display labels. Applies to
                both compressed label generation and final display truncation. Defaults to None.

            Label content:
            label_fields (Optional[Sequence[str]]): Fields to include in each label: one or
                more of "label", "n", "p", "q", "fe". If None, suppresses base label/stat
                text. Defaults to ("label", "n", "p").
            label_prefix (Optional[str]): Prefix prepended to each label, one of
                {None, "cid", "alpha"}. Defaults to None.

            Unlabeled clusters:
            skip_unlabeled (bool): If True, clusters without a label are omitted entirely.
                Defaults to False.
            placeholder_text (Optional[str]): Text shown for unlabeled clusters when
                skip_unlabeled is False. Defaults to style placeholder_text.
            placeholder_color (Optional[str]): Text color for placeholder labels.
                Defaults to style placeholder_color.
            placeholder_alpha (Optional[float]): Opacity for placeholder labels.
                Defaults to style placeholder_alpha.

            Separator lines:
            label_sep_xmin (Optional[float]): Left extent of inter-cluster separator lines
                (0–1, figure fraction). Defaults to the label text x-position.
            label_sep_xmax (Optional[float]): Right extent of inter-cluster separator lines
                (0–1, figure fraction). Defaults to 1.0.
            label_sep_color (Optional[str]): Separator line color.
                Defaults to style label_sep_color.
            label_sep_lw (Optional[float]): Separator line width (points).
                Defaults to style label_sep_lw.
            label_sep_alpha (Optional[float]): Separator line opacity.
                Defaults to style label_sep_alpha.

            Cluster boundaries:
            boundary_color (Optional[str]): Matrix cluster boundary line color.
                Defaults to style boundary_color.
            boundary_lw (Optional[float]): Matrix cluster boundary line width (points).
                Defaults to style boundary_lw.
            boundary_alpha (Optional[float]): Matrix cluster boundary line opacity.
                Defaults to style boundary_alpha.
            dendro_boundary_color (Optional[str]): Dendrogram boundary line color.
                Defaults to style dendro_boundary_color.
            dendro_boundary_lw (Optional[float]): Dendrogram boundary line width (points).
                Defaults to style dendro_boundary_lw.
            dendro_boundary_alpha (Optional[float]): Dendrogram boundary line opacity.
                Defaults to style dendro_boundary_alpha.

            Text formatting:
            omit_words (Optional[Sequence[str]]): Words to strip from labels
                (case-insensitive). Defaults to style label_omit_words.
            wrap_text (bool): Whether to wrap long label text onto multiple lines.
                Defaults to True.
            wrap_width (Optional[int]): Characters per wrapped line.
                Defaults to style label_wrap_width.
            overflow (str): Handling mode for text exceeding wrap_width, one of
                {"wrap", "ellipsis"}. Defaults to "wrap".

        Returns:
            Plotter: Self for chaining.
        """
        # Build label-generation options forwarded to Results.cluster_labels().
        label_options: dict[str, Any] = {"rank_by": rank_by, "label_mode": label_mode}
        if max_words is not None:
            label_options["max_words"] = max_words

        layer_kwargs: dict[str, Any] = {"_label_options": label_options, "overrides": overrides}
        # Text appearance
        if fontsize is not None:
            layer_kwargs["fontsize"] = fontsize
        if color is not None:
            layer_kwargs["color"] = color
        if font is not None:
            layer_kwargs["font"] = font
        if alpha is not None:
            layer_kwargs["alpha"] = alpha
        # Label content
        if label_fields is not None:
            layer_kwargs["label_fields"] = label_fields
        if label_prefix is not None:
            layer_kwargs["label_prefix"] = label_prefix
        # Unlabeled clusters
        layer_kwargs["skip_unlabeled"] = skip_unlabeled
        if placeholder_text is not None:
            layer_kwargs["placeholder_text"] = placeholder_text
        if placeholder_color is not None:
            layer_kwargs["placeholder_color"] = placeholder_color
        if placeholder_alpha is not None:
            layer_kwargs["placeholder_alpha"] = placeholder_alpha
        # Separator lines
        if label_sep_xmin is not None:
            layer_kwargs["label_sep_xmin"] = label_sep_xmin
        if label_sep_xmax is not None:
            layer_kwargs["label_sep_xmax"] = label_sep_xmax
        if label_sep_color is not None:
            layer_kwargs["label_sep_color"] = label_sep_color
        if label_sep_lw is not None:
            layer_kwargs["label_sep_lw"] = label_sep_lw
        if label_sep_alpha is not None:
            layer_kwargs["label_sep_alpha"] = label_sep_alpha
        # Cluster boundaries
        if boundary_color is not None:
            layer_kwargs["boundary_color"] = boundary_color
        if boundary_lw is not None:
            layer_kwargs["boundary_lw"] = boundary_lw
        if boundary_alpha is not None:
            layer_kwargs["boundary_alpha"] = boundary_alpha
        if dendro_boundary_color is not None:
            layer_kwargs["dendro_boundary_color"] = dendro_boundary_color
        if dendro_boundary_lw is not None:
            layer_kwargs["dendro_boundary_lw"] = dendro_boundary_lw
        if dendro_boundary_alpha is not None:
            layer_kwargs["dendro_boundary_alpha"] = dendro_boundary_alpha
        # Text formatting — max_words is also read by the renderer, not only label generation.
        if max_words is not None:
            layer_kwargs["max_words"] = max_words
        if omit_words is not None:
            layer_kwargs["omit_words"] = omit_words
        layer_kwargs["wrap_text"] = wrap_text
        if wrap_width is not None:
            layer_kwargs["wrap_width"] = wrap_width
        layer_kwargs["overflow"] = overflow
        self._layers.append(("cluster_labels", layer_kwargs))
        return self

    def plot_title(
        self,
        title: str,
        *,
        fontsize: Optional[float] = None,
        pad: Optional[float] = None,
        color: Optional[str] = None,
        font: Optional[str] = None,
        alpha: Optional[float] = None,
        **kwargs: Any,
    ) -> Plotter:
        """
        Declares a plot title.

        Args:
            title (str): Title text.

        Kwargs:
            fontsize (Optional[float]): Title font size (points). Defaults to style title_fontsize.
            pad (Optional[float]): Padding between title text and matrix axis (points).
                Defaults to style title_pad.
            color (Optional[str]): Title text color. Defaults to style text_color.
            font (Optional[str]): Font family for title text. Defaults to None.
            alpha (Optional[float]): Title text opacity. Defaults to None.
            **kwargs: Renderer keyword arguments. Defaults to {}.

        Returns:
            Plotter: Self for chaining.
        """
        if pad is not None:
            pad = float(pad)
        layer_kwargs: dict[str, Any] = {"title": title}
        if fontsize is not None:
            layer_kwargs["fontsize"] = fontsize
        if pad is not None:
            layer_kwargs["pad"] = pad
        if color is not None:
            layer_kwargs["color"] = color
        if font is not None:
            layer_kwargs["font"] = font
        if alpha is not None:
            layer_kwargs["alpha"] = alpha
        self._layers.append(("title", {**layer_kwargs, **kwargs}))
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
        # Consume authoritative geometry from Results.
        layout = self.results.cluster_layout()
        # NOTE:
        #   - Layout.leaf_order controls row ordering (statistically meaningful).
        #   - Layout.col_order controls column ordering (visual only).
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
        # Create figure and main axis.
        fig, ax = plt.subplots(figsize=self._style["figsize"])
        fig.subplots_adjust(**self._style["subplots_adjust"])
        if self._background is not None:
            fig.patch.set_facecolor(self._background)

        # Single-ownership boundary registry (matrix-aligned horizontals).
        boundary_registry = BoundaryRegistry()

        # Minor row gridlines (internal only; suppress axis extremes).
        if matrix_kwargs is not None and matrix_kwargs.get("show_minor_rows", True):
            step = matrix_kwargs.get("minor_row_step", 1)
            lw = matrix_kwargs.get("minor_row_lw", 0.15)
            alpha = matrix_kwargs.get("minor_row_alpha", 0.15)
            for y in range(0, n_rows, step):
                b = y - 0.5
                if b <= -0.5 or b >= n_rows - 0.5:
                    continue
                boundary_registry.register(b, lw=lw, color="black", alpha=alpha)

        # Cluster boundaries (suppress axis extremes; cluster lines override minor lines).
        if cluster_boundary_kwargs is not None:
            lw = cluster_boundary_kwargs.get("boundary_lw", self._style["boundary_lw"])
            alpha = cluster_boundary_kwargs.get("boundary_alpha", self._style["boundary_alpha"])
            color = cluster_boundary_kwargs.get("boundary_color", self._style["boundary_color"])
            for _cid, s, _e in layout.cluster_spans:
                b = s - 0.5
                if b <= -0.5 or b >= n_rows - 0.5:
                    continue
                boundary_registry.register(b, lw=lw, color=color, alpha=alpha)

        # Derive dendrogram boundary styling from label panel settings.
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

        # Render declared layers in order.
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
                # Consumed inside the cluster label panel; no direct rendering.
                continue
            else:
                raise NotImplementedError(f"Unknown plot layer: {layer}")
        # Row-level label tracks can render without cluster label text.
        if has_row_track and not has_cluster_label_layer:
            self._render_label_panel(fig, layout, bar_kwargs=bar_kwargs)

        # Render bottom colorbar strip (global legends).
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
        # Matrix axis never owns ticks; always keep clean.
        ax.set_xticks([])
        ax.set_yticks([])
        # Attach layout metadata for advanced users.
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

    def save(
        self,
        path: Union[str, PathLike[str]],
        *,
        dpi: Optional[float] = None,
        format: Optional[str] = None,
        bbox_inches: Optional[str] = None,
        pad_inches: Optional[float] = None,
        transparent: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """
        Saves the last rendered figure with correct background handling.

        Args:
            path (Union[str, PathLike[str]]): Output path for the figure.

        Kwargs:
            dpi (Optional[float]): Resolution in dots per inch. Defaults to the figure dpi.
            format (Optional[str]): Output format, e.g. "png", "pdf", "svg".
                Inferred from the path extension when None. Defaults to None.
            bbox_inches (Optional[str]): Bounding box adjustment. Pass "tight" to crop
                whitespace. Also accepts a matplotlib.transforms.Bbox instance.
                Defaults to None.
            pad_inches (Optional[float]): Padding around the figure when
                bbox_inches="tight" (inches). Defaults to 0.1.
            transparent (Optional[bool]): If True, axes and figure backgrounds are
                rendered transparent. Defaults to None.
            **kwargs: Additional matplotlib savefig options.
        """
        if self._fig is None or not self._figure_is_open():
            self._render()
        savefig_kwargs: dict[str, Any] = {}
        if dpi is not None:
            savefig_kwargs["dpi"] = dpi
        if format is not None:
            savefig_kwargs["format"] = format
        if bbox_inches is not None:
            savefig_kwargs["bbox_inches"] = bbox_inches
        if pad_inches is not None:
            savefig_kwargs["pad_inches"] = pad_inches
        if transparent is not None:
            savefig_kwargs["transparent"] = transparent
        self._fig.savefig(
            path,
            facecolor=self._fig.get_facecolor(),
            **savefig_kwargs,
            **kwargs,
        )

    def show(self) -> None:
        """
        Shows the last rendered figure with correct background handling.
        """
        if self._fig is None or not self._figure_is_open():
            self._render()
        plt.show()
