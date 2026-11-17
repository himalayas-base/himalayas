"""
himalayas/core/annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from collections.abc import Iterable as IterableABC
from os import PathLike
from typing import Dict, Iterable, List, Mapping, Optional, Set, Union

from .matrix import Matrix
from ..util.warnings import warn


def _read_gmt(path: Union[str, PathLike]) -> Dict[str, List[str]]:
    """
    Reads a GMT gene-set file: one term per line, tab-separated as name, description, then labels.
    The description is ignored and blank lines are skipped.

    Args:
        path (Union[str, PathLike]): Path to the GMT file.

    Returns:
        Dict[str, List[str]]: Mapping of term names to label lists, in file order.

    Raises:
        ValueError: If a line has no name, no description field, or no labels, if a term name
            repeats, or if the file holds no terms.
    """
    term_to_labels: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            # Validation
            fields = line.split("\t")
            term = fields[0].strip()
            labels = [label.strip() for label in fields[2:] if label.strip()]
            if not term or len(fields) < 3 or not labels:
                raise ValueError(
                    f"Line {line_number}: a GMT line needs a name, a description, and labels"
                )
            if term in term_to_labels:
                raise ValueError(f"Line {line_number}: duplicate term {term!r}")
            term_to_labels[term] = labels
    if not term_to_labels:
        raise ValueError("The GMT file contains no terms")
    return term_to_labels


class Annotations:
    """
    Class for storing term-to-labels annotations aligned to a matrix and filtering to labels present in the matrix.
    Original term labels are retained internally for explicit universe resolution in enrichment.
    """

    def __init__(
        self,
        term_to_labels: Mapping[str, Iterable[str]],
        matrix: Matrix,
        *,
        min_term_size: int = 2,
        max_term_size: Optional[int] = None,
    ) -> None:
        """
        Initializes the Annotations instance.

        Args:
            term_to_labels (Dict[str, Iterable[str]]): Mapping of terms to label lists.
            matrix (Matrix): Matrix providing the label universe.

        Kwargs:
            min_term_size (int): Minimum number of matrix-overlapping labels a term must have
                to be retained. Defaults to 2. (Separate from `Analysis.enrich`'s
                `min_overlap`.)
            max_term_size (Optional[int]): Maximum number of matrix-overlapping labels a term
                may have to be retained. None disables the upper bound. Defaults to None.
        """
        self.matrix_labels = set(matrix.labels)
        self.term_to_labels: Dict[str, Set[str]] = {}
        self._min_term_size = min_term_size
        self._max_term_size = max_term_size
        # Preserve original term labels so subset rebinds can still resolve
        # global-universe term sizes when enrichment uses a background matrix.
        self._source_term_to_labels: Dict[str, Set[str]] = {}
        self._validate_and_filter(term_to_labels)

    def _validate_and_filter(self, term_to_labels: Mapping[str, Iterable[str]]) -> None:
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
            k = len(labels)
            if k < self._min_term_size:
                dropped_terms.append(term)
                continue
            if self._max_term_size is not None and k > self._max_term_size:
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
                f"Dropped {dropped}/{total} annotations after matrix filtering (size or overlap constraints)",
                RuntimeWarning,
            )

    @classmethod
    def from_gmt(
        cls,
        path: Union[str, PathLike],
        matrix: Matrix,
        *,
        min_term_size: int = 2,
        max_term_size: Optional[int] = None,
    ) -> Annotations:
        """
        Builds an Annotations instance from a GMT gene-set file. Each line holds a term name,
        a description (ignored), and the term's labels, separated by tabs.

        Args:
            path (Union[str, PathLike]): Path to the GMT file.
            matrix (Matrix): Matrix providing the label universe.

        Kwargs:
            min_term_size (int): Minimum number of matrix-overlapping labels a term must have
                to be retained. Defaults to 2.
            max_term_size (Optional[int]): Maximum number of matrix-overlapping labels a term
                may have to be retained. None disables the upper bound. Defaults to None.

        Returns:
            Annotations: Annotations aligned to `matrix`.

        Raises:
            ValueError: If the file is malformed, repeats a term name, or holds no terms, or if
                no term overlaps the matrix labels.
        """
        return cls(
            _read_gmt(path),
            matrix,
            min_term_size=min_term_size,
            max_term_size=max_term_size,
        )

    def to_gmt(self, path: Union[str, PathLike]) -> None:
        """
        Writes the retained terms to a GMT file, with "-" as each description. Only labels
        present in the matrix are written, in sorted order.

        Args:
            path (Union[str, PathLike]): Path of the GMT file to write.
        """
        # One tab-separated line per term: name, description, labels.
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for term, labels in self.term_to_labels.items():
                fh.write("\t".join([term, "-", *sorted(labels)]) + "\n")

    @property
    def terms(self) -> List[str]:
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
        return Annotations(
            self.term_to_labels,
            matrix,
            min_term_size=self._min_term_size,
            max_term_size=self._max_term_size,
        )

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
