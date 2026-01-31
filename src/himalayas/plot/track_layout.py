"""
himalayas/plot/track_layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


class TrackLayoutManager:
    """
    Class for computing and storing label-panel track geometry.
    """

    def __init__(self) -> None:
        """
        Initializes the TrackLayoutManager instance.
        """
        self.tracks: List[Dict[str, Any]] = []
        self.order: Optional[Tuple[str, ...]] = None
        self._active_tracks: List[Dict[str, Any]] = []
        self._end_x: Optional[float] = None

    def register_track(
        self,
        name: str,
        renderer: Any,
        width: float,
        left_pad: float = 0.0,
        right_pad: float = 0.0,
        enabled: bool = True,
        kind: str = "row",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Registers a label-panel track.

        Args:
            name (str): Track name.
            renderer (Any): Track renderer instance.
            width (float): Track width in figure coordinates.
            left_pad (float): Left padding in figure coordinates.
            right_pad (float): Right padding in figure coordinates.
            enabled (bool): Whether the track is enabled.
            kind (str): Track kind, either 'row' or 'cluster'.
            payload (Optional[Dict[str, Any]]): Additional track-specific data.
        """
        # Validation
        if not isinstance(name, str) or not name:
            raise ValueError("track `name` must be a non-empty string")
        if kind not in {"row", "cluster"}:
            raise ValueError("track `kind` must be 'row' or 'cluster'")

        # Store track
        track = {
            "name": name,
            "kind": kind,
            "renderer": renderer,
            "left_pad": float(left_pad),
            "width": float(width),
            "right_pad": float(right_pad),
            "enabled": bool(enabled),
            "payload": dict(payload) if payload is not None else {},
        }
        self.tracks.append(track)

    def set_order(self, order: Optional[Sequence[str]]) -> None:
        """
        Sets the explicit track order.

        Args:
            order (Optional[Sequence[str]]): List/tuple of track names in desired order,
                or None to use registration order.

        Raises:
            TypeError: If order is not None or a list/tuple of strings.
            ValueError: If order contains duplicate or unknown track names.
        """
        # No reordering requested
        if order is None:
            self.order = None
            return

        # Validation
        if not isinstance(order, (list, tuple)):
            raise TypeError("label_track_order must be None or a list/tuple of unique strings")
        names = list(order)
        if any(not isinstance(n, str) for n in names):
            raise TypeError("label_track_order must be a list/tuple of unique strings")
        if len(set(names)) != len(names):
            raise ValueError("label_track_order contains duplicate track names")

        self.order = tuple(names)

    def _ordered_tracks(self) -> List[Dict[str, Any]]:
        """
        Returns the list of active tracks in the desired order.

        Returns:
            List[Dict[str, Any]]: Ordered list of active track dictionaries.

        Raises:
            ValueError: If there are unknown or duplicate track names in the order.
        """
        tracks = [dict(t) for t in self.tracks if t.get("enabled", True)]
        for track in tracks:
            track["payload"] = dict(track.get("payload", {}))
        active_names = [t.get("name") for t in tracks]
        # Validation
        if any(not n for n in active_names):
            raise ValueError("All label-panel tracks must have a non-empty 'name'")
        if len(set(active_names)) != len(active_names):
            raise ValueError(f"Active label-panel track names must be unique. Got: {active_names}")

        # No reordering requested
        if self.order is None:
            return tracks

        # Validate requested order
        names = list(self.order)
        available = set(active_names)
        unknown = [n for n in names if n not in available]
        if unknown:
            raise ValueError(
                "Unknown track(s) in label_track_order: "
                f"{unknown}. Available tracks: {active_names}"
            )
        # Reorder tracks
        name_to_track = {t["name"]: t for t in tracks}
        ordered = [name_to_track[n] for n in names]
        omitted = [t for t in tracks if t["name"] not in names]

        return ordered + omitted

    def compute_layout(
        self,
        base_x: float,
        gutter_width: float,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Computes the x0/x1 geometry for all active tracks.

        Args:
            base_x (float): Starting x position in figure coordinates.
            gutter_width (float): Gutter width before the first track.

        Returns:
            Dict[str, Tuple[float, float]]: Mapping track name → (x0, x1).
        """
        tracks = self._ordered_tracks()
        x_cursor = float(base_x) + float(gutter_width)
        for track in tracks:
            x_cursor += float(track["left_pad"])
            track["x0"] = x_cursor
            track["x1"] = x_cursor + float(track["width"])
            x_cursor = track["x1"] + float(track["right_pad"])

        # Store results
        self._active_tracks = tracks
        self._end_x = x_cursor
        return {t["name"]: (t["x0"], t["x1"]) for t in tracks}

    def get_tracks(self) -> List[Dict[str, Any]]:
        """
        Returns the list of active tracks after layout.

        Returns:
            List[Dict[str, Any]]: List of active track dictionaries.
        """
        return list(self._active_tracks)

    def get_end_x(self) -> Optional[float]:
        """
        Returns the ending x position after layout.

        Returns:
            Optional[float]: Ending x position in figure coordinates, or None if layout not computed.
        """
        return self._end_x
