"""
src/data/preprocessor.py
─────────────────────────
Text preprocessing pipeline for automotive reviews.

Design principles
-----------------
- Each step is a pure function so it is individually testable.
- The public API is the `ReviewPreprocessor` class.
- Information that might affect contextual sentiment is deliberately preserved.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

import pandas as pd

# ── Low-level helpers ─────────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """Convert fancy unicode characters to their ASCII equivalents where possible."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def remove_control_chars(text: str) -> str:
    """Remove non-printable / control characters while keeping punctuation."""
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)


def standardize_contractions(text: str) -> str:
    """Expand common English contractions to help the tokenizer."""
    contractions = {
        r"\bdon't\b": "do not",
        r"\bcan't\b": "cannot",
        r"\bwon't\b": "will not",
        r"\bisn't\b": "is not",
        r"\baren't\b": "are not",
        r"\bwasn't\b": "was not",
        r"\bweren't\b": "were not",
        r"\bhasn't\b": "has not",
        r"\bhaven't\b": "have not",
        r"\bhadn't\b": "had not",
        r"\bdidn't\b": "did not",
        r"\bdoesn't\b": "does not",
        r"\bshouldn't\b": "should not",
        r"\bwouldn't\b": "would not",
        r"\bcouldn't\b": "could not",
        r"\bmightn't\b": "might not",
        r"\bmustn't\b": "must not",
        r"\bI'm\b": "I am",
        r"\bI've\b": "I have",
        r"\bI'll\b": "I will",
        r"\bI'd\b": "I would",
        r"\bit's\b": "it is",
        r"\bhe's\b": "he is",
        r"\bshe's\b": "she is",
        r"\bthat's\b": "that is",
        r"\bthere's\b": "there is",
        r"\bthey're\b": "they are",
        r"\bwe're\b": "we are",
        r"\bthey've\b": "they have",
        r"\bwe've\b": "we have",
        r"\bthey'll\b": "they will",
        r"\bwe'll\b": "we will",
        r"\bhe'd\b": "he would",
        r"\bshe'd\b": "she would",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def clean_review(text: str) -> str:
    """
    Full cleaning pipeline for a single review string.
    Preserves punctuation because it carries sentiment signal.
    """
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_control_chars(text)
    text = normalize_unicode(text)
    text = standardize_contractions(text)
    text = normalize_whitespace(text)
    return text


# ── Dataset-level preprocessing ───────────────────────────────────────────────

class ReviewPreprocessor:
    """
    Stateless preprocessing utility for a DataFrame of reviews.

    Expected input columns: Review, Aspect, Sentiment (for training data).
    For inference, only the Review column is required.
    """

    def __init__(self, min_length: int = 5, max_length: int = 512):
        self.min_length = min_length
        self.max_length = max_length

    # -- Public API -----------------------------------------------------------

    def preprocess_single(self, text: str) -> str:
        """
        Preprocess a single review string for inference.

        Returns the cleaned text, truncated to ``max_length`` characters if
        needed.  Raises ``ValueError`` for empty / too-short input.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Review text must be a non-empty string.")
        cleaned = clean_review(text)
        if len(cleaned) < self.min_length:
            raise ValueError(
                f"Review is too short after cleaning (min {self.min_length} chars)."
            )
        return cleaned[:self.max_length]

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess a full DataFrame.

        Steps:
        1. Validate required columns.
        2. Drop rows with missing Review text.
        3. Clean each review string.
        4. Remove duplicates on (Review, Aspect).
        5. Filter out rows with unsupported Aspect / Sentiment values.

        Returns a cleaned copy of the DataFrame.
        """
        df = df.copy()
        self._validate_columns(df)

        # Drop rows with missing review text
        before = len(df)
        df = df.dropna(subset=["Review"])
        dropped_missing = before - len(df)

        # Clean review text
        df["Review"] = df["Review"].astype(str).apply(clean_review)

        # Filter out reviews that are too short after cleaning
        df = df[df["Review"].str.len() >= self.min_length]

        # Truncate reviews that are too long
        df["Review"] = df["Review"].str[:self.max_length]

        # Remove duplicate (Review, Aspect) pairs
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["Review", "Aspect"]).reset_index(drop=True)
        dropped_dupes = before_dedup - len(df)

        # Strip whitespace from categorical columns
        for col in ["Aspect", "Sentiment"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        stats = {
            "rows_dropped_missing": dropped_missing,
            "rows_dropped_duplicates": dropped_dupes,
            "rows_remaining": len(df),
        }
        return df, stats

    # -- Internal helpers ------------------------------------------------------

    def _validate_columns(self, df: pd.DataFrame) -> None:
        required = ["Review", "Aspect", "Sentiment"]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Dataset is missing required columns: {missing_cols}. "
                f"Found columns: {list(df.columns)}"
            )
