"""Style configuration for HiMaLAYAS plots."""

from __future__ import annotations

from typing import Any, Mapping, Optional

DEFAULT_STYLE = {
    # Figure layout
    "figsize": (9, 7),
    "subplots_adjust": {
        "left": 0.15,
        "right": 0.70,
        "bottom": 0.05,
        "top": 0.95,
    },
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
    # Matrix panel background (used to mask edge artifacts if desired)
    "matrix_gutter_color": None,
    # Gene-level annotation bar (row-level, purely visual)
    # Placed between dendrogram and matrix by default
    "gene_bar_width": 0.012,
    "gene_bar_gap": 0.006,
    "gene_bar_missing_color": "#eeeeee",
    # Bars rendered inside the label panel (to the left of text)
    # (Label_bar_default_width, label_bar_default_gap removed)
    # Default settings for cluster p-value bars (e.g., sigbar)
    # NOTE: scaling is controlled by an explicit `norm` passed to plot_cluster_bar
    # `Sigbar_min_logp` / `sigbar_max_logp` are legacy defaults used only for the legend
    "sigbar_width": 0.015,
    "sigbar_cmap": "YlOrBr",
    "sigbar_min_logp": 2.0,
    "sigbar_max_logp": 10.0,
    "sigbar_alpha": 0.9,
    "show_sigbar": True,
    # (Sigbar_gap removed)
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
    "placeholder_text": "\u2014",
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


class StyleConfig:
    """
    Class for storing plot style defaults and overrides.
    """

    def __init__(self, defaults: Optional[Mapping[str, Any]] = None) -> None:
        """
        Initializes the StyleConfig instance.
        """
        if defaults is None:
            defaults = DEFAULT_STYLE
        self._defaults = dict(defaults)
        self._overrides: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Gets a style value with override priority."""
        if key in self._overrides:
            return self._overrides[key]
        return self._defaults.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Overrides a style value."""
        self._overrides[key] = value

    def update(self, overrides: Mapping[str, Any]) -> None:
        """Applies multiple overrides at once."""
        for key, value in overrides.items():
            self._overrides[key] = value

    def as_dict(self) -> dict[str, Any]:
        """Returns a merged view of defaults and overrides."""
        merged = dict(self._defaults)
        merged.update(self._overrides)
        return merged

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, key: object) -> bool:
        return key in self._overrides or key in self._defaults
