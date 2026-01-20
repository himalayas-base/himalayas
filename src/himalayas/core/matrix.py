"""
himalayas/core/matrix
~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from himalayas.util.warnings import warn


class Matrix:
    """
    Immutable container for a data matrix.

    Note: this object should be treated as immutable. Downstream clustering,
    layouts, and enrichment assume matrix contents are frozen.
    """

    def __init__(
        self, df: pd.DataFrame, *, matrix_semantics: str = "similarity", axis: str = "rows"
    ) -> None:
        """
        Initializes Matrix.

        Args:
            df (pd.DataFrame): DataFrame holding matrix contents with
                row labels as index.
            matrix_semantics (str, optional): Semantics of matrix values.
                One of {"similarity", "distance"}. Defaults to "similarity".
            axis (str, optional): Axis along which rows are organized.
                Currently only "rows" is supported. Defaults to "rows".
        """
        # Defensive copy: Matrix contents are treated as immutable downstream
        self.df = df.copy()
        self.values = self.df.values
        self.labels = self.df.index.to_numpy(dtype=object)
        self.matrix_semantics = matrix_semantics
        self.axis = axis

        self._validate()

    def _validate(self) -> None:
        """
        Validates matrix contents and properties.

        Raises:
            ValueError: If matrix is invalid.
        """
        if self.df.index.has_duplicates:
            raise ValueError("Matrix labels must be unique")

        if self.matrix_semantics == "similarity":
            diag = np.diag(self.values)
            if not np.allclose(diag, 1.0):
                warn("Similarity matrix diagonal is not all 1.0", RuntimeWarning)
