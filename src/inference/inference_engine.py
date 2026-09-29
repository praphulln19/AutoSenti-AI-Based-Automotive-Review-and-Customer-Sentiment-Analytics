"""
src/inference/inference_engine.py
───────────────────────────────────
High-level inference engine that orchestrates the full ABSA pipeline:

    review text
        → aspect extraction
        → sentiment classification (per aspect)
        → structured AspectSentimentResult

Supports:
- Single review inference
- Batch review inference
- Model hot-swap (BERT ↔ XLM-R) with cached loading

The engine is the ONLY public interface that the presentation layer interacts
with — it hides all model / tokenizer / device details.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config_loader import get_model_config, get_project_root
from src.data.preprocessor import ReviewPreprocessor
from src.inference.aspect_extractor import AspectExtractor
from src.inference.sentiment_classifier import SentimentClassifier, SentimentResult
from src.models.model_factory import load_tokenizer_and_model

logger = logging.getLogger(__name__)


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class AspectSentimentResult:
    """One (aspect, sentiment) pair for a single review."""
    review: str
    aspect: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    triggered_keywords: List[str] = field(default_factory=list)


@dataclass
class ReviewAnalysisResult:
    """Full analysis result for one review."""
    review: str
    model_key: str
    aspects_found: List[str]
    predictions: List[AspectSentimentResult]
    no_aspects_detected: bool = False

    @property
    def as_dataframe(self) -> pd.DataFrame:
        if not self.predictions:
            return pd.DataFrame(columns=["Aspect", "Sentiment", "Confidence"])
        rows = [
            {
                "Aspect": p.aspect,
                "Sentiment": p.sentiment,
                "Confidence": f"{p.confidence:.1%}",
                "Positive %": f"{p.probabilities.get('Positive', 0):.1%}",
                "Negative %": f"{p.probabilities.get('Negative', 0):.1%}",
                "Neutral %": f"{p.probabilities.get('Neutral', 0):.1%}",
            }
            for p in self.predictions
        ]
        return pd.DataFrame(rows)


# ── Inference engine ──────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Caching inference engine.

    Models are loaded once and reused across calls.
    Switching model_key triggers a reload.
    """

    def __init__(self):
        self._preprocessor = ReviewPreprocessor()
        self._extractor = AspectExtractor()
        self._classifier: Optional[SentimentClassifier] = None
        self._current_model_key: Optional[str] = None
        self._current_checkpoint: Optional[str] = None

    # ── Checkpoint resolution ─────────────────────────────────────────────────

    @staticmethod
    def resolve_checkpoint(model_key: str) -> Optional[str]:
        """
        Return the path to the best available fine-tuned checkpoint for
        *model_key*, or None if no fine-tuned weights exist yet.

        Priority order:
          1. models/checkpoints/<key>/best_model/   ← preferred fine-tuned weights
          2. models/checkpoints/<key>/checkpoint-*/ ← latest epoch checkpoint
          3. None  → fall back to pretrained HuggingFace Hub weights
        """
        ckpt_root = get_project_root() / "models" / "checkpoints" / model_key

        # 1. Prefer the explicit best_model directory
        best = ckpt_root / "best_model"
        if best.is_dir() and (best / "config.json").exists():
            return str(best)

        # 2. Fall back to the latest epoch checkpoint
        if ckpt_root.is_dir():
            epoch_ckpts = sorted(
                [d for d in ckpt_root.iterdir()
                 if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda d: int(d.name.split("-")[-1]),
            )
            if epoch_ckpts:
                latest = epoch_ckpts[-1]
                if (latest / "config.json").exists():
                    return str(latest)

        return None  # no fine-tuned weights yet

    # ── Model management ──────────────────────────────────────────────────────

    def load_model(self, model_key: str, checkpoint_path: Optional[str] = None) -> None:
        """
        Load (or hot-swap) the transformer model.

        Parameters
        ----------
        model_key       : "bert" or "xlmr"
        checkpoint_path : Optional explicit path to a fine-tuned checkpoint
                          directory.  If None, auto-resolves the best available
                          fine-tuned checkpoint; falls back to HuggingFace Hub
                          pretrained weights only if no checkpoint exists.
        """
        # Auto-resolve fine-tuned checkpoint when none is specified
        if checkpoint_path is None:
            checkpoint_path = self.resolve_checkpoint(model_key)
            if checkpoint_path:
                logger.info(
                    "Auto-resolved fine-tuned checkpoint for '%s': %s",
                    model_key, checkpoint_path,
                )
            else:
                logger.warning(
                    "No fine-tuned checkpoint found for '%s' — "
                    "loading pretrained weights (predictions will be random until training completes).",
                    model_key,
                )

        cache_key = (model_key, checkpoint_path)
        if (self._current_model_key == model_key
                and self._current_checkpoint == checkpoint_path
                and checkpoint_path is not None):
            logger.debug("Model '%s' already loaded from same checkpoint — skipping.", model_key)
            return

        logger.info("Loading model: %s  checkpoint=%s", model_key, checkpoint_path)
        tokenizer, model, device = load_tokenizer_and_model(
            model_key, checkpoint_path=checkpoint_path
        )
        cfg = get_model_config(model_key)
        max_length = cfg.get("max_length", 256)

        self._classifier = SentimentClassifier(model, tokenizer, device, max_length)
        self._current_model_key = model_key
        self._current_checkpoint = checkpoint_path
        logger.info("Model '%s' ready on device=%s  source=%s",
                    model_key, device, checkpoint_path or "HuggingFace Hub (pretrained)")


    @property
    def is_model_loaded(self) -> bool:
        return self._classifier is not None

    @property
    def current_model_key(self) -> Optional[str]:
        return self._current_model_key

    # ── Single review ─────────────────────────────────────────────────────────

    def analyse_review(self, review_text: str) -> ReviewAnalysisResult:
        """
        Run ABSA on a single review.

        Raises
        ------
        RuntimeError  if no model has been loaded.
        ValueError    if review_text is empty after preprocessing.
        """
        if not self.is_model_loaded:
            raise RuntimeError(
                "No model loaded. Call load_model() before running inference."
            )

        # Preprocess
        cleaned = self._preprocessor.preprocess_single(review_text)

        # Aspect extraction
        aspect_keywords = self._extractor.extract_with_keywords(cleaned)
        aspects = list(aspect_keywords.keys())

        if not aspects:
            return ReviewAnalysisResult(
                review=review_text,
                model_key=self._current_model_key,
                aspects_found=[],
                predictions=[],
                no_aspects_detected=True,
            )

        # Sentiment classification
        pairs = [(cleaned, asp) for asp in aspects]
        sentiment_results: List[SentimentResult] = self._classifier.classify_batch(pairs)

        predictions = [
            AspectSentimentResult(
                review=review_text,
                aspect=sr.aspect,
                sentiment=sr.sentiment,
                confidence=sr.confidence,
                probabilities=sr.probabilities,
                triggered_keywords=aspect_keywords.get(sr.aspect, []),
            )
            for sr in sentiment_results
        ]

        return ReviewAnalysisResult(
            review=review_text,
            model_key=self._current_model_key,
            aspects_found=aspects,
            predictions=predictions,
        )

    # ── Batch reviews ─────────────────────────────────────────────────────────

    def analyse_batch(
        self,
        reviews: List[str],
        progress_callback=None,
    ) -> List[ReviewAnalysisResult]:
        """
        Run ABSA on a list of reviews.

        Parameters
        ----------
        reviews           : List of raw review strings.
        progress_callback : Optional callable(current, total) for progress updates.

        Returns
        -------
        List of ReviewAnalysisResult, one per input review.
        """
        if not self.is_model_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")

        results = []
        total = len(reviews)
        for i, review in enumerate(reviews, 1):
            try:
                result = self.analyse_review(review)
            except ValueError as exc:
                logger.warning("Skipping review %d due to validation error: %s", i, exc)
                result = ReviewAnalysisResult(
                    review=review,
                    model_key=self._current_model_key,
                    aspects_found=[],
                    predictions=[],
                    no_aspects_detected=True,
                )
            results.append(result)
            if progress_callback:
                progress_callback(i, total)

        return results

    def analyse_dataframe(
        self,
        df: pd.DataFrame,
        review_column: str = "Review",
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        Analyse all reviews in a DataFrame and return a flat results DataFrame.

        Input columns: at minimum the review_column.
        Output columns: Review, Aspect, Sentiment, Confidence, + probability columns.
        """
        reviews = df[review_column].astype(str).tolist()
        batch_results = self.analyse_batch(reviews, progress_callback=progress_callback)

        rows = []
        for res in batch_results:
            if res.no_aspects_detected or not res.predictions:
                rows.append({
                    "Review": res.review,
                    "Aspect": "—",
                    "Sentiment": "—",
                    "Confidence": "—",
                    "Positive %": "—",
                    "Negative %": "—",
                    "Neutral %": "—",
                })
            else:
                for p in res.predictions:
                    rows.append({
                        "Review": res.review,
                        "Aspect": p.aspect,
                        "Sentiment": p.sentiment,
                        "Confidence": p.confidence,
                        "Positive %": p.probabilities.get("Positive", 0),
                        "Negative %": p.probabilities.get("Negative", 0),
                        "Neutral %": p.probabilities.get("Neutral", 0),
                    })

        return pd.DataFrame(rows)
