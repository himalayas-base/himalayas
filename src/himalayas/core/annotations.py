"""
himalayas/core/annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from collections.abc import Iterable as IterableABC
from typing import Dict, Iterable, Set

from .matrix import Matrix
from ..util.warnings import warn


class Annotations:
    """
    Class for storing term-to-labels annotations aligned to a matrix and filtering to labels present in the matrix.
    Original term labels are retained internally for explicit universe resolution in enrichment.
    """

    def __init__(self, term_to_labels: Dict[str, Iterable[str]], matrix: Matrix) -> None:
        """
        Initializes the Annotations instance.

        Args:
            term_to_labels (Dict[str, Iterable[str]]): Mapping of terms to label lists.
            matrix (Matrix): Matrix providing the label universe.
        """
        self.matrix_labels = set(matrix.labels)
        self.term_to_labels: Dict[str, Set[str]] = {}
        # Preserve original term labels so subset rebinds can still resolve
        # global-universe term sizes when enrichment uses a background matrix.
        self._source_term_to_labels: Dict[str, Set[str]] = {}
        self._validate_and_filter(term_to_labels)

    def _validate_and_filter(self, term_to_labels: Dict[str, Iterable[str]]) -> None:
        """
        Validates and filters the input term-to-labels mapping to ensure that only labels
        present in the matrix are retained. Terms with no overlapping labels are dropped,
        and a warning is issued if any terms are dropped. Raises an error if no terms remain
        after filtering.

        Args:
            term_to_labels (Dict[str, Iterable[str]]): Mapping of terms to their
                associated labels.

        Raises:
            ValueError: If no annotation terms overlap with matrix labels.
        """
        # Filter term labels to the matrix universe and track dropped terms.
        dropped_terms = []
        for term, labels in term_to_labels.items():
            if isinstance(labels, (str, bytes)):
                raise TypeError(
                    f"Labels for term {term!r} must be an iterable of labels, not a string"
                )
            if not isinstance(labels, IterableABC):
                raise TypeError(f"Labels for term {term!r} must be an iterable of labels")
            labels = set(labels)
            self._source_term_to_labels[term] = labels
            # Filter labels to those present in the matrix.
            labels = labels & self.matrix_labels
            if len(labels) == 0:
                dropped_terms.append(term)
                continue
            self.term_to_labels[term] = labels

        # Validation: check if any terms remain after filtering.
        if len(self.term_to_labels) == 0:
            raise ValueError(
                "No annotation terms overlap matrix labels " "(all terms dropped after filtering)"
            )
        # Warn if any terms were dropped.
        if dropped_terms:
            kept = len(self.term_to_labels)
            dropped = len(dropped_terms)
            total = kept + dropped
            warn(
                f"Dropped {dropped}/{total} annotations with no overlap to matrix labels",
                RuntimeWarning,
            )

    @property
    def terms(self) -> list[str]:
        """
        Returns annotation terms.

        Returns:
            list[str]: List of annotation terms.
        """
        return list(self.term_to_labels.keys())

    def rebind(self, matrix: Matrix) -> Annotations:
        """
        Returns a new Annotations object filtered to labels present in `matrix`. This
        is intended for explicit zoom/subset workflows. The original Annotations object
        is not mutated. Source term labels are preserved across rebinds so enrichment
        with a global `background` can still resolve global term sizes.

        Args:
            matrix (Matrix): The Matrix object to align annotations to.

        Returns:
            Annotations: A new Annotations object aligned to `matrix`.
        """
        # Rebind from currently visible terms to preserve explicit stepwise subsetting semantics.
        return Annotations(self.term_to_labels, matrix)

    def labels_for_universe(self, universe_labels: Iterable[str]) -> Dict[str, Set[str]]:
        """
        Resolves term labels intersected with an explicit enrichment universe.

        Args:
            universe_labels (Iterable[str]): Labels defining the target universe.

        Returns:
            Dict[str, Set[str]]: Per-term labels intersected to the requested universe.
        """
        universe = set(universe_labels)
        out: Dict[str, Set[str]] = {}
        for term, labels in self._source_term_to_labels.items():
            overlap = labels & universe
            if overlap:
                out[term] = overlap
        return out
