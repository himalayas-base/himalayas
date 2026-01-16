"""Lightweight term summarization with optional NLP."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def summarize_terms(
    words: Iterable[str], weights: Optional[Iterable[float]] = None, max_words: int = 6
) -> str:
    """
    Compress a list of annotation terms into a short representative label.

    Uses NLTK (tokenization, stopwords, lemmatization) if available.
    Falls back to a simple regex-based tokenizer if NLTK is unavailable.

    Parameters
    ----------
    words : iterable of str
        Term names (already human-readable).
    weights : iterable of float, optional
        Optional weights (e.g. -log10 pval). Must align with words.
    max_words : int
        Maximum number of words to return.

    Returns
    -------
    str
        Space-separated representative label.
    """
    counts = Counter()

    words_iter = list(words)
    if weights is None:
        weights = [1.0] * len(words_iter)
    else:
        weights = list(weights)
        if len(weights) != len(words_iter):
            raise ValueError("weights length must match words length")

    # Attempt to use NLTK lazily
    try:
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        def tokenize(text: str) -> List[str]:
            return word_tokenize(text)

        def normalize(tok: str) -> Optional[str]:
            tok = re.sub(r"[^\w\-]", "", tok.lower())
            if not tok or tok in stop_words:
                return None
            return lemmatizer.lemmatize(tok)

    except (ImportError, LookupError):
        # Fallback: regex tokenization, lowercase, no lemmatization
        def tokenize(text: str) -> List[str]:
            return re.split(r"\s+", text)

        def normalize(tok: str) -> Optional[str]:
            tok = re.sub(r"[^\w\-]", "", tok.lower())
            return tok or None

    for phrase, w in zip(words_iter, weights):
        for t in tokenize(str(phrase)):
            t = normalize(t)
            if t is None:
                continue
            counts[t] += w

    if not counts:
        return "N/A"

    top_tokens = [tok for tok, _ in counts.most_common(max_words)]
    return " ".join(top_tokens)


def summarize_clusters(
    df: pd.DataFrame,
    term_col: str = "term_name",
    cluster_col: str = "cluster",
    weight_col: str = "pval",
    *,
    label_mode: str = "compressed",
) -> pd.DataFrame:
    """
    Return a DataFrame mapping cluster to short textual label, suitable for plotting.

    Parameters
    ----------
    df : pandas.DataFrame
        Enrichment results containing term and cluster columns.
    term_col : str
        Column containing human-readable term names.
    cluster_col : str
        Column containing cluster IDs.
    weight_col : str
        Column containing p-values (used for both best p-value and weighting).
    label_mode : {"compressed", "top_term"}
        - "compressed": keyword compression across all terms in a cluster,
          weighted by -log10(p).
        - "top_term": Use the single most enriched term (minimum p-value) as the label.

    Returns
    -------
    pandas.DataFrame
        Columns: ["cluster", "label", "pval"].
    """
    if term_col not in df.columns:
        raise KeyError(f"Missing column: {term_col}")
    if cluster_col not in df.columns:
        raise KeyError(f"Missing column: {cluster_col}")

    if label_mode not in {"compressed", "top_term"}:
        raise ValueError("label_mode must be one of {'compressed', 'top_term'}")

    df = df.copy()

    # Precompute weights only when needed (avoid unnecessary columns/work)
    if label_mode == "compressed":
        if weight_col in df.columns:
            df["_weight"] = -np.log10(df[weight_col].clip(lower=1e-300))
        else:
            df["_weight"] = 1.0
    else:
        # top_term requires a p-value column to define 'most enriched'
        if weight_col not in df.columns:
            raise KeyError(f"Missing column required for label_mode='top_term': {weight_col}")

    rows = []
    for cid, sub in df.groupby(cluster_col, sort=False):
        cid_int = int(cid)

        if label_mode == "top_term":
            # Deterministic: choose the single smallest p-value row.
            best_idx = sub[weight_col].astype(float).idxmin()
            best_row = df.loc[best_idx]
            label = str(best_row[term_col])
            best_pval = float(best_row[weight_col])
        else:
            label = summarize_terms(
                sub[term_col].tolist(),
                sub["_weight"].tolist(),
            )
            best_pval = sub[weight_col].min() if weight_col in sub.columns else None

        rows.append(
            {
                "cluster": cid_int,
                "label": label,
                "pval": best_pval,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["cluster", "label", "pval"])

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
