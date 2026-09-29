"""
src/config_loader.py
────────────────────
Loads and provides centralized access to config.yaml.
All other modules import from here rather than reading the file directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List

import yaml

# Locate config.yaml relative to the project root (one level above src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    """Read config.yaml once and cache the result."""
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {_CONFIG_PATH}")
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── Convenience accessors ─────────────────────────────────────────────────────

def get_aspects() -> List[str]:
    return load_config()["aspects"]


def get_sentiment_labels() -> List[str]:
    return load_config()["sentiment_labels"]


def get_aspect_keywords() -> Dict[str, List[str]]:
    return load_config()["aspect_keywords"]


def get_model_config(model_key: str) -> Dict[str, Any]:
    """
    Returns config for a specific model key ("bert" or "xlmr").
    """
    cfg = load_config()["models"]
    if model_key not in cfg:
        raise KeyError(f"Model key '{model_key}' not found in config. "
                       f"Available: {list(cfg.keys())}")
    return cfg[model_key]


def get_project_root() -> Path:
    return _PROJECT_ROOT
