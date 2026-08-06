"""
himalayas/plot/style
~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple, TypedDict, Union

try:
    from typing import TypeAlias
except ImportError:  # Python <3.10
    from typing_extensions import TypeAlias

from matplotlib.colors import Colormap

# Type alias for style values.
StyleValue: TypeAlias = Union[
    str,
    float,
    int,
    bool,
    None,
    Sequence[float],
    Sequence[str],
    Mapping[str, float],
    Colormap,
]


class StyleDefaults(TypedDict):
    """
    Typed dictionary for plot style defaults.
    """

    figsize: Tuple[float, float]
    subplots_adjust: Dict[str, float]
    dendro_axes: Sequence[float]
    dendro_color: str
    dendro_lw: float
    label_axes: Sequence[float]
    label_x: float
    label_gutter_width: float
    label_gutter_color: str
    ylabel_pad: float
    matrix_gutter_color: Optional[str]
    label_bar_width: float
    label_bar_missing_color: str
    sigbar_width: float
    sigbar_cmap: Union[str, Colormap]
    sigbar_alpha: float
    label_bar_pad: float
    boundary_color: str
    boundary_lw: float
    boundary_alpha: float
    placeholder_text: str
    placeholder_color: str
    placeholder_alpha: float
    text_color: str
    title_fontsize: float
    title_pad: float
    label_fontsize: float
    label_sep_color: str
    label_sep_lw: float
    label_sep_alpha: float
    label_sep_xmin: Optional[float]
    label_sep_xmax: Optional[float]
    label_omit_words: Optional[Sequence[str]]
    label_fields: Tuple[str, ...]
    label_wrap_width: Optional[int]
    compact_axes: Optional[Sequence[float]]
    compact_marker_width: float
    compact_bridge_width: float
    compact_table_pad: float
    compact_marker_fontsize: float
    compact_line_color: str
    compact_line_lw: float
    compact_line_alpha: float
    compact_line_shape: str
    compact_line_style: str
    compact_line_start: str
    compact_line_end: str
    compact_cluster_span_cap_width: float
    cluster_span_color: Optional[str]
    cluster_span_lw: float
    cluster_span_alpha: float
    cluster_span_gap: float
    cluster_span_cap_width: float
    cluster_span_left_pad: float
    cluster_span_right_pad: float


DEFAULT_STYLE: StyleDefaults = {
    # Figure layout
    "figsize": (9, 7),
    "subplots_adjust": {
        "left": 0.15,
        "right": 0.70,
        "bottom": 0.05,
        "top": 0.95,
    },
    # Dendrogram axis box [x0, y0, w, h].
    "dendro_axes": [0.06, 0.05, 0.09, 0.90],
    "dendro_color": "#888888",
    "dendro_lw": 1.0,
    # Label panel axis box [x0, y0, w, h].
    "label_axes": [0.70, 0.05, 0.29, 0.90],
    "label_x": 0.02,
    # Gutter between matrix and label panel.
    "label_gutter_width": 0.01,
    "label_gutter_color": "white",
    # Padding between matrix and ylabel axis (fraction of figure width).
    "ylabel_pad": 0.015,
    # Matrix panel background (used to mask edge artifacts if desired).
    "matrix_gutter_color": None,
    # Row-level annotation bar (label-panel track).
    "label_bar_width": 0.012,
    "label_bar_missing_color": "#eeeeee",
    # Bars rendered inside the label panel (to the left of text).
    # (label_bar_default_width, label_bar_default_gap removed).
    # Default settings for cluster score bars (e.g., sigbar).
    # NOTE: scaling is controlled by an explicit `norm` passed to plot_cluster_bar.
    "sigbar_width": 0.015,
    "sigbar_cmap": "YlOrBr",
    "sigbar_alpha": 0.9,
    # (Sigbar_gap removed)
    # Label panel bar/text spacing.
    "label_bar_pad": 0.01,
    # Cluster boundary lines.
    "boundary_color": "black",
    "boundary_lw": 0.5,
    "boundary_alpha": 0.6,
    # Placeholder for unlabeled clusters.
    "placeholder_text": "\u2014",
    "placeholder_color": "#b22222",
    "placeholder_alpha": 0.6,
    # Default text color (used unless overridden via kwargs).
    "text_color": "black",
    "title_fontsize": 14,
    # Matrix title padding in points (Matplotlib text units).
    "title_pad": 15,
    "label_fontsize": 9,
    # Separator lines in label panel.
    "label_sep_color": "gray",
    "label_sep_lw": 0.5,
    "label_sep_alpha": 0.3,
    # Optional override for label separator segment span (label-axes fraction;
    # 0..1 spans the label panel, values outside extend beyond it).
    # If None, separators start after gutter+sigbar+pad and extend to 1.0.
    "label_sep_xmin": None,
    "label_sep_xmax": None,
    # Words to omit from displayed cluster labels.
    "label_omit_words": None,
    # Which fields to show in cluster labels, in order.
    # Allowed values: "label", "n", "p", "q", "fe".
    "label_fields": ("label", "n", "p"),
    # Optional label wrapping (characters per line); None = disabled.
    "label_wrap_width": None,
    # Compact radiating-label panel axis box [x0, y0, w, h].
    # None: defaults to label_axes (set via set_label_panel) unless explicitly overridden.
    "compact_axes": None,
    # Marker column width (fraction of compact_axes width).
    "compact_marker_width": 0.08,
    # Leader-line bridge width (fraction of compact_axes width).
    "compact_bridge_width": 0.45,
    # Padding between the bridge and the label table (fraction of compact_axes width).
    "compact_table_pad": 0.02,
    "compact_marker_fontsize": 8,
    "compact_line_color": "#c0562c",
    "compact_line_lw": 0.9,
    "compact_line_alpha": 0.65,
    # One of {"straight", "curved", "elbow"}.
    "compact_line_shape": "straight",
    # One of {"solid", "dashed", "dotted"}.
    "compact_line_style": "solid",
    # Connector-start point decoration, one of {"tick", "round", "none"}, used only
    # when cluster_span is None.
    "compact_line_start": "tick",
    # One of {"tick", "arrow", "round", "none"}.
    "compact_line_end": "tick",
    # Optional end-cap width (axes fraction) for a cluster_span="line" span, scaled
    # for the narrower bridge axis (compact_bridge_width) rather than the full label
    # panel. 0.0 draws a bare line; caps are opt-in.
    "compact_cluster_span_cap_width": 0.0,
    # Cluster-abreast span, opt-in via cluster_span=... on plot_cluster_labels()
    # or plot_cluster_labels_compact(). None color inherits label_sep_color.
    "cluster_span_color": None,
    "cluster_span_lw": 1.0,
    "cluster_span_alpha": 0.8,
    "cluster_span_gap": 0.15,
    # Optional end-cap width (axes fraction). 0.0 draws a bare line; caps are opt-in.
    "cluster_span_cap_width": 0.0,
    # Horizontal padding around the span centerline, independent of label_bar_pad.
    "cluster_span_left_pad": 0.0,
    "cluster_span_right_pad": 0.01,
}


class StyleConfig:
    """
    Class for storing plot style defaults and overrides.
    """

    def __init__(self, defaults: Optional[Mapping[str, StyleValue]] = None) -> None:
        """
        Initializes the StyleConfig instance.

        Args:
            defaults (Optional[Mapping[str, StyleValue]]): Base style defaults. Defaults to None.
        """
        if defaults is None:
            defaults = DEFAULT_STYLE
        self._defaults: Dict[str, StyleValue] = dict(defaults)
        self._overrides: Dict[str, StyleValue] = {}

    def get(self, key: str, default: Optional[StyleValue] = None) -> StyleValue:
        """
        Gets a style value with override priority.

        Args:
            key (str): Style key.
            default (Optional[StyleValue]): Default value if key not found. Defaults to None.

        Returns:
            StyleValue: Resolved style value.
        """
        if key in self._overrides:
            return self._overrides[key]
        return self._defaults.get(key, default)

    def set(self, key: str, value: StyleValue) -> None:
        """
        Overrides a style value.

        Args:
            key (str): Style key.
            value (StyleValue): Style value to set.
        """
        self._overrides[key] = value

    def update(self, overrides: Mapping[str, StyleValue]) -> None:
        """
        Applies multiple overrides at once.

        Args:
            overrides (Mapping[str, StyleValue]): Mapping of style keys to values.
        """
        for key, value in overrides.items():
            self._overrides[key] = value

    def as_dict(self) -> Dict[str, StyleValue]:
        """
        Returns a merged view of defaults and overrides.

        Returns:
            Dict[str, StyleValue]: Merged style dictionary.
        """
        merged = dict(self._defaults)
        merged.update(self._overrides)
        return merged

    def __getitem__(self, key: str) -> StyleValue:
        """
        Gets a style value with override priority.

        Args:
            key (str): Style key.

        Returns:
            StyleValue: Resolved style value.
        """
        return self.get(key)

    def __contains__(self, key: object) -> bool:
        """
        Checks if a style key exists in defaults or overrides.

        Args:
            key (object): Style key.

        Returns:
            bool: True if key exists, False otherwise.
        """
        return key in self._overrides or key in self._defaults
