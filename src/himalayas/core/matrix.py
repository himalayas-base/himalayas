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
    Class for holding a validated matrix with labeled rows for clustering and plotting.
    """

    def __init__(
        self, df: pd.DataFrame, *, matrix_semantics: str = "similarity", axis: str = "rows"
    ) -> None:
        """
        Initializes the Matrix instance.
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
