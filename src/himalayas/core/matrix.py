"""Matrix container and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from himalayas.util.warnings import warn


class Matrix:
    """
    Minimal matrix container for HiMaLAYAS prototyping.

    Holds a validated matrix with labeled rows.
    Rows correspond to observations (e.g. genes, recipes, documents).
    Index values are referred to internally as `labels`.

    Notes
    -----
    This object should be treated as immutable. Downstream clustering,
    layouts, and enrichment assume matrix contents do not change.
    """

    def __init__(self, df: pd.DataFrame, *, matrix_semantics: str = "similarity", axis: str = "rows") -> None:
        # Defensive copy: Matrix contents are treated as immutable downstream
        self.df = df.copy()
        self.values = self.df.values
        self.labels = self.df.index.to_numpy(dtype=object)
        self.matrix_semantics = matrix_semantics
        self.axis = axis

        self._validate()

    def _validate(self) -> None:
        # if self.df.shape[0] != self.df.shape[1]:
        #     raise ValueError("Matrix must be square")

        # if not np.all(self.df.index == self.df.columns):
        #     raise ValueError("Row and column labels must match")

        if self.df.index.has_duplicates:
            raise ValueError("Matrix labels must be unique")

        if self.matrix_semantics == "similarity":
            diag = np.diag(self.values)
            if not np.allclose(diag, 1.0):
                warn("Similarity matrix diagonal is not all 1.0", RuntimeWarning)
