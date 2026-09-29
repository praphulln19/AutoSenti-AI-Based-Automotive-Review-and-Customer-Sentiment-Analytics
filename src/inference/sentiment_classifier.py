"""
src/inference/sentiment_classifier.py
───────────────────────────────────────
Transformer-based sentiment classifier for individual (aspect, review) pairs.

Design
------
- Accepts a loaded model + tokenizer (dependency-injected for testability).
- Outputs structured SentimentResult objects with label + confidence.
- Handles both single pairs and batches efficiently.
- Device-agnostic (works on CPU, CUDA, MPS).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.models.absa_dataset import ABSADataset

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Structured output for a single aspect-sentiment prediction."""
    aspect: str
    sentiment: str        # "Positive" | "Negative" | "Neutral"
    confidence: float     # softmax probability of the predicted class [0, 1]
    probabilities: Dict[str, float]   # {label: probability} for all classes


class SentimentClassifier:
    """
    Classifies sentiment for (aspect, review) pairs using a fine-tuned
    transformer model.

    Parameters
    ----------
    model     : A loaded PreTrainedModel for sequence classification.
    tokenizer : Matching PreTrainedTokenizer.
    device    : torch.device to run inference on.
    max_length: Tokenisation max length (model-specific).
    """

    LABELS = ABSADataset.ID2LABEL   # {0: "Positive", 1: "Negative", 2: "Neutral"}

    def __init__(self, model, tokenizer, device: torch.device, max_length: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.model.eval()

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, review: str, aspect: str) -> SentimentResult:
        """
        Classify sentiment for a single (review, aspect) pair.
        """
        return self.classify_batch([(review, aspect)])[0]

    def classify_batch(
        self, pairs: List[Tuple[str, str]], batch_size: int = 32
    ) -> List[SentimentResult]:
        """
        Classify a list of (review, aspect) tuples efficiently.

        Parameters
        ----------
        pairs      : List of (review_text, aspect_text) tuples.
        batch_size : Number of pairs processed per forward pass.

        Returns
        -------
        List of SentimentResult, one per input pair.
        """
        results: List[SentimentResult] = []

        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start: start + batch_size]
            reviews = [p[0] for p in batch_pairs]
            aspects = [p[1] for p in batch_pairs]

            encoding = self.tokenizer(
                text=aspects,
                text_pair=reviews,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits          # (batch, num_labels)
                probs = F.softmax(logits, dim=-1)  # (batch, num_labels)

            for i, (review, aspect) in enumerate(batch_pairs):
                prob_vec = probs[i].cpu().tolist()
                pred_id = int(torch.argmax(probs[i]).item())
                label = self.LABELS[pred_id]
                confidence = prob_vec[pred_id]
                prob_dict = {self.LABELS[j]: round(prob_vec[j], 4) for j in range(len(prob_vec))}

                results.append(SentimentResult(
                    aspect=aspect,
                    sentiment=label,
                    confidence=round(confidence, 4),
                    probabilities=prob_dict,
                ))

        return results
