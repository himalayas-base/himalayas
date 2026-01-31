"""
himalayas/core/annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Dict, Iterable, Set, cast

from himalayas.util.warnings import warn

from .matrix import Matrix


class Annotations:
    """
    Class for storing term-to-label annotations aligned to a matrix and filtering to present labels.
    """

    def __init__(self, term_to_labels: Dict[str, Iterable[str]], matrix: Matrix) -> None:
        """
        Initializes the Annotations instance.
        """
        self.matrix_labels = set(matrix.labels)
        self.term_to_labels: Dict[str, Set[str]] = {}
        self._validate_and_filter(term_to_labels)

    def _validate_and_filter(self, term_to_labels: Dict[str, Iterable[str]]) -> None:
        """
        Validates and filters the input term-to-labels mapping to ensure
        that only labels present in the matrix are retained. Terms with no
        overlapping labels are dropped, and a warning is issued if any terms
        are dropped. Raises an error if no terms remain after filtering.

        Args:
            term_to_labels (Dict[str, Iterable[str]]): Mapping of terms to their
                associated labels.

        Raises:
            ValueError: If no annotation terms overlap with matrix labels
        """
        dropped_terms = []
        for term, labels in term_to_labels.items():
            # Filter labels to those present in the matrix
            labels = set(labels) & self.matrix_labels
            if len(labels) == 0:
                dropped_terms.append(term)
                continue
            self.term_to_labels[term] = labels

        # Check if any terms remain after filtering
        if len(self.term_to_labels) == 0:
            raise ValueError(
                "No annotation terms overlap matrix labels " "(all terms dropped after filtering)"
            )
        # Warn if any terms were dropped
        if dropped_terms:
            kept = len(self.term_to_labels)
            dropped = len(dropped_terms)
            total = kept + dropped
            warn(
                f"Dropped {dropped}/{total} annotations with no overlap to matrix labels",
                RuntimeWarning,
            )

    def terms(self) -> list:
        """
        Returns the list of annotation terms.

        Returns:
            list: List of annotation terms.
        """
        return list(self.term_to_labels.keys())

    def rebind(self, matrix: Matrix) -> Annotations:
        """
        Returns a new Annotations object filtered to labels present in `matrix`. This
        is intended for explicit zoom / subset workflows. The original Annotations object
        is not mutated.

        Args:
            matrix (Matrix): The Matrix object to align annotations to.

        Returns:
            Annotations: A new Annotations object aligned to `matrix`.
        """
        return Annotations(cast(Dict[str, Iterable[str]], self.term_to_labels), matrix)
