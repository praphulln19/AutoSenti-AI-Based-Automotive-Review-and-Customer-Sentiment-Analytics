"""
src/models/absa_dataset.py
───────────────────────────
PyTorch Dataset for Aspect-Based Sentiment Analysis.

Each sample is an (aspect, review) pair encoded as:

    "[CLS] <aspect> [SEP] <review> [SEP]"

which is the standard BERT sentence-pair format.  The model then learns
to predict the sentiment label for *that specific aspect* given the review.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class ABSADataset(Dataset):
    """
    Dataset for aspect-level sentiment classification.

    Parameters
    ----------
    reviews   : List of raw review strings.
    aspects   : List of aspect strings (parallel to reviews).
    labels    : Optional list of integer labels (0=Positive,1=Negative,2=Neutral).
                Omit for inference mode.
    tokenizer : A Hugging Face PreTrainedTokenizer instance.
    max_length: Maximum token length (model-specific).
    """

    LABEL2ID: Dict[str, int] = {"Positive": 0, "Negative": 1, "Neutral": 2}
    ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

    def __init__(
        self,
        reviews: List[str],
        aspects: List[str],
        tokenizer,
        max_length: int = 256,
        labels: Optional[List[int]] = None,
    ):
        if len(reviews) != len(aspects):
            raise ValueError("reviews and aspects must have the same length.")
        if labels is not None and len(labels) != len(reviews):
            raise ValueError("labels must have the same length as reviews.")

        self.reviews = reviews
        self.aspects = aspects
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            text=self.aspects[idx],       # Sentence A = aspect
            text_pair=self.reviews[idx],  # Sentence B = review
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item: Dict[str, torch.Tensor] = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        # token_type_ids is not produced by all tokenizer variants (e.g. RoBERTa)
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item

    # ── Class helpers ──────────────────────────────────────────────────────────

    @classmethod
    def labels_from_strings(cls, sentiment_strings: List[str]) -> List[int]:
        """Convert a list of sentiment strings to integer label IDs."""
        unknown = set(sentiment_strings) - set(cls.LABEL2ID.keys())
        if unknown:
            raise ValueError(f"Unknown sentiment labels encountered: {unknown}")
        return [cls.LABEL2ID[s] for s in sentiment_strings]
