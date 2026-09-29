"""
src/models/model_factory.py
────────────────────────────
Factory functions for loading tokenizers and models.

Design
------
- Supports BERT and XLM-R model families via a unified interface.
- Handles device selection (CUDA / MPS / CPU) automatically.
- Models are loaded from Hugging Face Hub or from local checkpoint paths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizerFast,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.config_loader import get_model_config
from src.models.absa_dataset import ABSADataset

logger = logging.getLogger(__name__)

# ── Device selection ──────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Model factory ─────────────────────────────────────────────────────────────

def load_tokenizer(model_key: str) -> PreTrainedTokenizerBase:
    """
    Load the tokenizer for a configured model key ("bert" or "xlmr").

    Parameters
    ----------
    model_key : One of the keys defined under ``models`` in config.yaml.
    """
    cfg = get_model_config(model_key)
    model_name = cfg["name"]
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def load_model(
    model_key: str,
    num_labels: int = 3,
    checkpoint_path: Union[str, Path, None] = None,
) -> Tuple[PreTrainedModel, torch.device]:
    """
    Load a sequence-classification model.

    Parameters
    ----------
    model_key       : Config key ("bert" or "xlmr").
    num_labels      : Number of sentiment classes (default 3).
    checkpoint_path : If provided, load weights from this local directory
                      instead of the Hugging Face Hub.

    Returns
    -------
    (model, device) tuple — model is already moved to device.
    """
    cfg = get_model_config(model_key)
    model_name = cfg.get("name")
    device = get_device()

    source = str(checkpoint_path) if checkpoint_path else model_name
    logger.info("Loading model from '%s' → device=%s", source, device)

    id2label = ABSADataset.ID2LABEL
    label2id = ABSADataset.LABEL2ID

    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,   # safe when re-using pretrained weights
    )
    model = model.to(device)
    return model, device


def load_tokenizer_and_model(
    model_key: str,
    checkpoint_path: Union[str, Path, None] = None,
) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel, torch.device]:
    """Convenience wrapper that returns (tokenizer, model, device)."""
    tokenizer = load_tokenizer(model_key)
    model, device = load_model(model_key, checkpoint_path=checkpoint_path)
    return tokenizer, model, device
