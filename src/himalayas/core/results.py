"""
himalayas/core/results
~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .clustering import Clusters
from .layout import ClusterLayout
from .matrix import Matrix

_TERM_FIELD = "term"
_CLUSTER_FIELD = "cluster"
_TERM_NAME_FIELD = "term_name"
_CLUSTER_LABEL_OUTPUT_FIELDS = ["cluster", "label", "pval", "qval", "score", "n", "term"]


def _summarize_terms(
    words: Iterable[str],
    weights: Optional[Iterable[float]] = None,
    *,
    max_words: int = 6,
) -> str:
    """
    Compresses term phrases into a short representative label.
    Uses NLTK tokenization/stopwords/lemmatization when available and falls back to regex
    tokenization otherwise.

    Args:
        words (Iterable[str]): Term phrases.
        weights (Optional[Iterable[float]]): Optional term weights. Defaults to None.

    Kwargs:
        max_words (int): Maximum number of tokens in the output. Defaults to 6.

    Returns:
        str: Space-separated representative label.

    Raises:
        ValueError: If weights length does not match words length.
    """
    # Align optional weights to terms and validate equal lengths.
    counts = Counter()
    words_iter = list(words)
    if weights is None:
        weights = [1.0] * len(words_iter)
    else:
        weights = list(weights)
        if len(weights) != len(words_iter):
            raise ValueError("weights length must match words length")

    # Prefer richer NLTK normalization; fallback to regex-only tokenization.
    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize

        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        def tokenize(text: str) -> List[str]:
            return word_tokenize(text)

        def normalize(tok: str) -> Optional[str]:
            tok = re.sub(r"[^\w\-]", "", tok.lower())
            if not tok or tok in stop_words:
                return None
            return lemmatizer.lemmatize(tok)

    # Keep labeling usable even when NLTK or corpora are unavailable.
    except (ImportError, LookupError):

        def tokenize(text: str) -> List[str]:
            return re.split(r"\s+", text)

        def normalize(tok: str) -> Optional[str]:
            tok = re.sub(r"[^\w\-]", "", tok.lower())
            return tok or None

    # Aggregate weighted token frequencies across all phrases.
    for phrase, weight in zip(words_iter, weights):
        for token in tokenize(str(phrase)):
            token = normalize(token)
            if token is None:
                continue
            counts[token] += weight

    # Return a stable placeholder when no usable tokens remain.
    if not counts:
        return "N/A"

    top_tokens = [tok for tok, _ in counts.most_common(max_words)]
    return " ".join(top_tokens)


def _cluster_size_from_rows(sub: pd.DataFrame) -> Optional[int]:
    """
    Resolves cluster size from enrichment rows when available.

    Args:
        sub (pd.DataFrame): Per-cluster subset.

    Returns:
        Optional[int]: Cluster size, if available.
    """
    if "n" not in sub.columns:
        return None
    n_vals = sub["n"].dropna()
    if n_vals.empty:
        return None
    try:
        return int(n_vals.iloc[0])
    except (TypeError, ValueError):
        return None


def _as_optional_float(value: Any) -> Optional[float]:
    """
    Converts a scalar value to float when present and numeric.

    Args:
        value (Any): Input scalar.

    Returns:
        Optional[float]: Parsed float or None when missing/non-numeric.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    except ValueError:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_rank_spec(
    *,
    rank_by: str = "p",
) -> str:
    """
    Resolves the score column used for ranking cluster terms.

    Kwargs:
        rank_by (str): Semantic ranking target, one of {"p", "q"}.
            Defaults to "p".

    Returns:
        str: Score column name ("pval" or "qval") for ranking.

    Raises:
        ValueError: If rank_by is unsupported.
    """
    if rank_by not in {"p", "q"}:
        raise ValueError("rank_by must be one of {'p', 'q'}")
    return "pval" if rank_by == "p" else "qval"


class Results:
    """
    Class for holding analysis results and attached context for plotting or subsetting.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        method: str,
        *,
        matrix: Optional[Matrix] = None,
        clusters: Optional[Clusters] = None,
        layout: Optional[ClusterLayout] = None,
        parent: Optional[Results] = None,
    ) -> None:
        """
        Initializes the Results instance.

        Args:
            df (pd.DataFrame): Results table.
            method (str): Analysis method identifier.

        Kwargs:
            matrix (Optional[Matrix]): Matrix associated with these results. Defaults to None.
            clusters (Optional[Clusters]): Clusters associated with these results. Defaults to None.
            layout (Optional[ClusterLayout]): Cached layout for plotting. Defaults to None.
            parent (Optional[Results]): Parent results for provenance. Defaults to None.
        """
        self.df = df
        self.method = method
        self.matrix = matrix
        self.clusters = clusters
        self.parent = parent
        self._layout = layout
        self.params: Dict[str, Any] = {}
        if clusters is not None:
            self.params["linkage_threshold"] = clusters.threshold

    def filter(self, expr, **kwargs) -> Results:
        """
        Returns a Results object filtered by a DataFrame query expression.

        Args:
            expr (str): DataFrame query expression.

        Kwargs:
            **kwargs: Additional keyword arguments for pd.DataFrame.query. Defaults to {}.

        Returns:
            Results: New Results object with filtered DataFrame.
        """
        filtered_df = self.df.query(expr, **kwargs)
        return Results(
            filtered_df,
            method=self.method,
            matrix=self.matrix,
            clusters=self.clusters,
            layout=self._layout,
            parent=self,
        )

    def subset(
        self,
        cluster: int,
    ) -> Results:
        """
        Returns a Results object restricted to a single cluster.

        Args:
            cluster (int): Cluster id to subset.

        Returns:
            Results: New Results object restricted to the cluster.

        Raises:
            ValueError: If matrix or clusters are not attached.
            KeyError: If the cluster id is not found.
        """
        # Validation
        if self.matrix is None or self.clusters is None:
            raise ValueError("Results must have matrix and clusters to subset")
        cid = int(cluster)
        if cid not in self.clusters.cluster_to_labels:
            raise KeyError(f"Cluster {cid} not found")

        # Subset matrix (rows only)
        labels = sorted(self.clusters.cluster_to_labels[cid])
        df_sub = self.matrix.df.loc[labels]
        sub_matrix = Matrix(
            df_sub,
            axis=self.matrix.axis,
        )
        # Return new Results view
        return Results(
            df=pd.DataFrame(),
            method="subset",
            matrix=sub_matrix,
            clusters=None,
            layout=None,
            parent=self,
        )

    @staticmethod
    def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
        """
        Performs Benjamini–Hochberg FDR correction.

        Args:
            pvals (np.ndarray): Array of p-values.

        Returns:
            np.ndarray: FDR-adjusted q-values aligned to input order.

        Raises:
            ValueError: If p-values are non-finite or outside [0, 1].
        """
        pvals = np.asarray(pvals, dtype=float)
        if pvals.size == 0:
            return pvals

        # Validation
        if np.any((pvals < 0) | (pvals > 1) | ~np.isfinite(pvals)):
            raise ValueError("p-values must be finite and in [0, 1]")

        # Compute BH q-values and restore original order
        m = pvals.size
        order = np.argsort(pvals)
        ranked = pvals[order]
        q = ranked * (m / (np.arange(1, m + 1)))
        q = np.minimum.accumulate(q[::-1])[::-1]
        q = np.clip(q, 0.0, 1.0)
        out = np.empty_like(q)
        out[order] = q

        return out

    def with_qvalues(self, pval_col: str = "pval", qval_col: str = "qval") -> Results:
        """
        Returns a new Results with BH-FDR q-values added as `qval_col`. Does not mutate
        the original Results.

        Args:
            pval_col (str): Column name for p-values. Defaults to "pval".
            qval_col (str): Column name for q-values. Defaults to "qval".

        Returns:
            Results: New Results object with q-values added.

        Raises:
            KeyError: If the p-value column is missing.
        """
        # Validation
        if pval_col not in self.df.columns:
            raise KeyError(f"Missing p-value column: {pval_col!r}")

        # Compute q-values while preserving row alignment
        df2 = self.df.copy()
        pvals = df2[pval_col].to_numpy(dtype=float)
        # Preserve row alignment: NaN p-values yield NaN q-values
        qvals = np.full_like(pvals, np.nan, dtype=float)
        mask = np.isfinite(pvals)
        qvals[mask] = self._bh_fdr(pvals[mask])
        df2[qval_col] = qvals

        return Results(
            df2,
            method=self.method,
            matrix=self.matrix,
            clusters=self.clusters,
            layout=self._layout,
            parent=self,
        )

    def cluster_layout(self) -> ClusterLayout:
        """
        Returns the authoritative clustering layout for downstream plotting.

        Returns:
            ClusterLayout: Attached cluster layout object.

        Raises:
            ValueError: If no ClusterLayout is attached.
        """
        if self._layout is None:
            raise ValueError("Results has no attached ClusterLayout")
        return self._layout

    def cluster_spans(self) -> List[Tuple[int, int, int]]:
        """
        Returns cluster spans in dendrogram order.

        Returns:
            List[Tuple[int, int, int]]: List of (cluster_id, start, end) spans.
        """
        return self.cluster_layout().cluster_spans

    def cluster_labels(
        self,
        *,
        rank_by: str = "p",
        label_mode: str = "top_term",
        max_words: int = 6,
    ) -> pd.DataFrame:
        """
        Builds per-cluster textual labels for plotting.

        Kwargs:
            rank_by (str): Ranking statistic for representative terms, one of {"p", "q"}.
                Defaults to "p".
            label_mode (str): One of {"top_term", "compressed"}. Defaults to "top_term".
            max_words (int): Maximum words for compressed labels. Defaults to 6.

        Returns:
            pd.DataFrame: Columns ["cluster", "label", "pval", "qval", "score", "n", "term"].

        Raises:
            KeyError: If required columns are missing.
            ValueError: If label_mode is unsupported.
        """
        # Validation
        if _TERM_FIELD not in self.df.columns:
            raise KeyError(f"Missing column: {_TERM_FIELD}")
        if _CLUSTER_FIELD not in self.df.columns:
            raise KeyError(f"Missing column: {_CLUSTER_FIELD}")
        if label_mode not in {"top_term", "compressed"}:
            raise ValueError("label_mode must be one of {'top_term', 'compressed'}")
        score_col = _resolve_rank_spec(rank_by=rank_by)
        score_present = score_col in self.df.columns
        if label_mode == "top_term" and not score_present:
            raise KeyError(f"Missing column required for label_mode='top_term': {score_col}")
        # Keep compressed-mode compatibility when ranking by p-values:
        # if p-values are absent, labels are still generated from uniform term weights.
        # q-based ranking remains strict because q-values are always optional downstream.
        if label_mode == "compressed" and rank_by == "q" and not score_present:
            raise KeyError(f"Missing column required for rank_by={rank_by!r}: {score_col}")

        # Resolve display label source with optional human-readable fallback.
        label_source = _TERM_NAME_FIELD if _TERM_NAME_FIELD in self.df.columns else _TERM_FIELD

        # Precompute compressed-mode weights; enforce p-values for top-term mode.
        df = self.df.copy()
        if label_mode == "compressed":
            if score_present:
                df["_weight"] = -np.log10(df[score_col].clip(lower=1e-300))
            else:
                # Enables compressed labels even when p-values are absent.
                df["_weight"] = 1.0

        # Build one canonical label row per cluster.
        rows = []
        for cid, sub in df.groupby(_CLUSTER_FIELD, sort=False):
            cid_int = int(cid)
            best_idx = None
            if label_mode == "top_term":
                best_idx = sub[score_col].astype(float).idxmin()
                best_row = df.loc[best_idx]
                label_val = best_row[label_source]
                if label_source != _TERM_FIELD and (label_val is None or pd.isna(label_val)):
                    label_val = best_row[_TERM_FIELD]
                label = str(label_val)
                best_term = str(best_row[_TERM_FIELD])
            else:
                if label_source != _TERM_FIELD:
                    labels = sub[label_source].where(pd.notna(sub[label_source]), sub[_TERM_FIELD])
                    labels = labels.tolist()
                else:
                    labels = sub[_TERM_FIELD].tolist()
                label = _summarize_terms(
                    labels,
                    sub["_weight"].tolist(),
                    max_words=max_words,
                )
                if score_present:
                    best_idx = sub[score_col].astype(float).idxmin()
                    best_term = str(sub.loc[best_idx, _TERM_FIELD])
                else:
                    best_term = None
            best_pval = None
            best_qval = None
            best_score = None
            if best_idx is not None:
                best_row = df.loc[best_idx]
                best_pval = _as_optional_float(best_row.get("pval", None))
                best_qval = _as_optional_float(best_row.get("qval", None))
                best_score = _as_optional_float(best_row.get(score_col, None))

            # Prefer explicit enrichment `n`; otherwise infer from attached clusters.
            n_members = _cluster_size_from_rows(sub)
            if n_members is None and self.clusters is not None:
                labels_for_cluster = self.clusters.cluster_to_labels.get(cid_int, None)
                if labels_for_cluster is not None:
                    n_members = int(len(labels_for_cluster))

            rows.append(
                {
                    "cluster": cid_int,
                    "label": label,
                    "pval": best_pval,
                    "qval": best_qval,
                    "score": best_score,
                    "n": n_members,
                    "term": best_term,
                }
            )

        if not rows:
            return pd.DataFrame(columns=_CLUSTER_LABEL_OUTPUT_FIELDS)

        return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
