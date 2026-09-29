"""
src/inference/aspect_extractor.py
───────────────────────────────────
Keyword-based automotive aspect extractor.

Strategy
--------
1. Normalise the review text (lower-case, strip punctuation variants).
2. For each configured aspect, check whether any of its keywords appear
   as whole words / phrases in the normalised text.
3. Return the ordered list of matched aspects.

The approach is:
- Deterministic and interpretable.
- GPU-free — runs before the transformer.
- Configurable: keywords are managed in config.yaml.

Implicit-aspect handling is explicitly not covered here (constraint §12).
"""

from __future__ import annotations

import re
from typing import List

from src.config_loader import get_aspects, get_aspect_keywords


class AspectExtractor:
    """
    Keyword-driven aspect extractor for automotive reviews.

    Usage
    -----
    >>> extractor = AspectExtractor()
    >>> extractor.extract("The mileage is great but service is poor.")
    ['Mileage', 'Service']
    """

    def __init__(self):
        self._aspects: List[str] = get_aspects()
        self._kw_map: dict[str, List[str]] = get_aspect_keywords()
        # Pre-compile one regex per aspect (word-boundary anchored)
        self._patterns: dict[str, re.Pattern] = {}
        for aspect in self._aspects:
            keywords = self._kw_map.get(aspect, [])
            if not keywords:
                continue
            # Sort longest keywords first to prefer specific matches
            sorted_kw = sorted(keywords, key=len, reverse=True)
            # Escape and join with word boundaries
            pattern_str = r"|".join(
                r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)"
                for kw in sorted_kw
            )
            self._patterns[aspect] = re.compile(pattern_str, re.IGNORECASE)

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, review: str) -> List[str]:
        """
        Return the list of aspects detected in *review*, in taxonomy order.

        Parameters
        ----------
        review : Cleaned review text (may be raw; normalisation is applied internally).

        Returns
        -------
        List of aspect names (subset of the configured taxonomy).
        An empty list means no known aspects were detected.
        """
        if not review or not review.strip():
            return []

        # Normalise for matching
        norm = self._normalise(review)

        found = []
        for aspect in self._aspects:
            pattern = self._patterns.get(aspect)
            if pattern and pattern.search(norm):
                found.append(aspect)

        return found

    def extract_with_keywords(self, review: str) -> dict[str, List[str]]:
        """
        Extended version that also returns which keywords triggered each match.
        Useful for debugging / explanation UI.
        """
        if not review or not review.strip():
            return {}
        norm = self._normalise(review)
        result: dict[str, List[str]] = {}
        for aspect in self._aspects:
            pattern = self._patterns.get(aspect)
            if not pattern:
                continue
            matches = pattern.findall(norm)
            if matches:
                result[aspect] = list(set(m.strip() for m in matches))
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        """Lower-case; replace certain punctuation with spaces to help matching."""
        text = text.lower()
        # Replace hyphens and underscores with spaces for multi-word keyword matching
        text = re.sub(r"[-_]", " ", text)
        return text
