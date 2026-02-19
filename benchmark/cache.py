"""
cache.py — Disk-based embedding cache.

Embeddings are stored as .npy files keyed by a hash of
(model_name, dataset_name, split, text_fingerprint).

This avoids re-encoding the same dataset multiple times, which is the
most expensive part of running a benchmark sweep.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(".cache/embeddings")


def _cache_key(model_name: str, dataset_name: str, texts: list[str]) -> str:
    """Deterministic key based on model, dataset, and text content."""
    text_hash = hashlib.md5(
        json.dumps(texts, ensure_ascii=False).encode()
    ).hexdigest()[:12]
    safe_model = model_name.replace("/", "__")
    safe_dataset = dataset_name.replace("/", "__")
    return f"{safe_model}__{safe_dataset}__{text_hash}"


def load_from_cache(
    model_name: str,
    dataset_name: str,
    texts: list[str],
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> np.ndarray | None:
    key = _cache_key(model_name, dataset_name, texts)
    path = cache_dir / f"{key}.npy"
    if path.exists():
        logger.info(f"Cache hit: {path}")
        return np.load(str(path))
    return None


def save_to_cache(
    embeddings: np.ndarray,
    model_name: str,
    dataset_name: str,
    texts: list[str],
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(model_name, dataset_name, texts)
    path = cache_dir / f"{key}.npy"
    np.save(str(path), embeddings)
    logger.info(f"Cached embeddings → {path}")
    return path


def encode_with_cache(
    model,
    texts: list[str],
    dataset_name: str,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode texts, using on-disk cache if available."""
    cached = load_from_cache(model.name, dataset_name, texts, cache_dir)
    if cached is not None:
        return cached
    embeddings = model.encode(texts, batch_size=batch_size, show_progress=show_progress)
    save_to_cache(embeddings, model.name, dataset_name, texts, cache_dir)
    return embeddings
