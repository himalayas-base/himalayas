"""
himalayas/core/matrix
~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Matrix:
    """
    Class for holding a validated matrix with labeled rows for clustering and plotting.
    """

    def __init__(self, df: pd.DataFrame, *, axis: str = "rows") -> None:
        """
        Initializes the Matrix instance.

        Args:
            df (pd.DataFrame): Input data with labeled rows.

        Kwargs:
            axis (str): Row/column orientation of labels. Defaults to "rows".

        Raises:
            ValueError: If the matrix labels are not unique.
        """
        # Defensive copy: Matrix contents are treated as immutable downstream
        self.df = df.copy()
        self.values = self.df.values
        self.labels = self.df.index.to_numpy(dtype=object)
        self.axis = axis

        self._validate()

    def _validate(self) -> None:
        """
        Validates matrix contents and properties.

        Raises:
            ValueError: If matrix is invalid.
        """
        # Validation
        if self.df.index.has_duplicates:
            raise ValueError("Matrix labels must be unique")
