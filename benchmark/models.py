"""
models.py — Embedding model adapters.

Each adapter wraps a backend behind a unified EmbeddingModel interface
with a single method: encode(texts) -> np.ndarray.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class EmbeddingModel(abc.ABC):
    """Minimal interface every adapter must implement."""

    name: str  # human-readable identifier used in results / cache keys

    @abc.abstractmethod
    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of strings into a 2-D float32 array of shape (N, dim).
        Implementations are responsible for batching internally.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# HuggingFace / sentence-transformers adapter
# ---------------------------------------------------------------------------

@dataclass
class SentenceTransformerModel(EmbeddingModel):
    """
    Wraps any model loadable by sentence-transformers.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        device:             'cpu', 'cuda', 'mps', or None (auto-detect).
        normalize:          L2-normalize embeddings after encoding.
        encode_kwargs:      Extra kwargs forwarded to model.encode().
    """

    model_name_or_path: str
    device: Optional[str] = None
    normalize: bool = True
    encode_kwargs: dict = field(default_factory=dict)

    # set after __post_init__
    _model: object = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self.name = self.model_name_or_path

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("pip install sentence-transformers")

        logger.info(f"Loading {self.model_name_or_path} …")
        self._model = SentenceTransformer(
            self.model_name_or_path,
            device=self.device,
        )

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        self._load()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            **self.encode_kwargs,
        )
        return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# OpenAI API adapter  (optional — only needed if you want to benchmark API models)
# ---------------------------------------------------------------------------

@dataclass
class OpenAIEmbeddingModel(EmbeddingModel):
    """
    Wraps the OpenAI embeddings endpoint.

    Requires: pip install openai
    Set OPENAI_API_KEY in your environment.
    """

    model_name_or_path: str = "text-embedding-3-small"
    dimensions: Optional[int] = None   # pass None to use model default

    def __post_init__(self):
        self.name = self.model_name_or_path

    def encode(
        self,
        texts: list[str],
        batch_size: int = 512,      # OpenAI supports large batches
        show_progress: bool = True,
    ) -> np.ndarray:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        from tqdm import tqdm

        client = OpenAI()
        all_embeddings = []
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        for batch in tqdm(batches, disable=not show_progress, desc=self.name):
            kwargs = dict(input=batch, model=self.model_name_or_path)
            if self.dimensions:
                kwargs["dimensions"] = self.dimensions
            response = client.embeddings.create(**kwargs)
            vecs = [d.embedding for d in response.data]
            all_embeddings.extend(vecs)

        arr = np.array(all_embeddings, dtype=np.float32)
        # L2-normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-9)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_model(cfg: dict) -> EmbeddingModel:
    """
    Instantiate an EmbeddingModel from a plain config dict.

    Example configs:
        {"type": "sentence_transformer", "model": "BAAI/bge-small-en-v1.5"}
        {"type": "openai",               "model": "text-embedding-3-small"}
    """
    kind = cfg.get("type", "sentence_transformer")

    if kind == "sentence_transformer":
        return SentenceTransformerModel(
            model_name_or_path=cfg["model"],
            device=cfg.get("device"),
            normalize=cfg.get("normalize", True),
            encode_kwargs=cfg.get("encode_kwargs", {}),
        )
    elif kind == "openai":
        return OpenAIEmbeddingModel(
            model_name_or_path=cfg.get("model", "text-embedding-3-small"),
            dimensions=cfg.get("dimensions"),
        )
    else:
        raise ValueError(f"Unknown model type: {kind!r}")
