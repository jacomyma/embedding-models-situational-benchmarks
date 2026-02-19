"""
tasks.py — Benchmark task definitions.

Each Task subclass knows how to:
  1. Load its dataset (HuggingFace datasets or synthetic).
  2. Encode the relevant text fields using a given model (via cache).
  3. Evaluate and return a flat dict of metric_name -> float.

This mirrors the MTEB design where tasks are self-contained units.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .cache import encode_with_cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base Task
# ---------------------------------------------------------------------------

class Task(abc.ABC):
    name: str
    description: str = ""

    @abc.abstractmethod
    def run(self, model, cache_dir: Path, **kwargs) -> dict[str, float]:
        """Run the task and return {metric_name: score}."""


# ---------------------------------------------------------------------------
# Semantic Textual Similarity (STS)
# ---------------------------------------------------------------------------

@dataclass
class STSTask(Task):
    """
    Pearson & Spearman correlation between cosine similarity and human scores.

    Expects a HuggingFace dataset with columns: sentence1, sentence2, score.
    The score column is assumed to be in [0, 5] and is normalized to [0, 1].

    Default dataset: stsb (the classic STS-Benchmark).
    """

    name: str = "STS-B"
    description: str = "Semantic Textual Similarity – STS-Benchmark"
    hf_dataset: str = "sentence-transformers/stsb"
    split: str = "test"
    sentence1_col: str = "sentence1"
    sentence2_col: str = "sentence2"
    score_col: str = "score"
    score_scale: float = 5.0   # divide scores by this to get [0,1]

    def run(self, model, cache_dir: Path, **kwargs) -> dict[str, float]:
        from datasets import load_dataset
        from scipy.stats import pearsonr, spearmanr

        logger.info(f"[{self.name}] Loading dataset {self.hf_dataset} …")
        ds = load_dataset(self.hf_dataset, split=self.split)

        s1 = list(ds[self.sentence1_col])
        s2 = list(ds[self.sentence2_col])
        labels = np.array(ds[self.score_col], dtype=np.float32) / self.score_scale

        all_texts = s1 + s2
        embs = encode_with_cache(
            model, all_texts, dataset_name=self.name, cache_dir=cache_dir, **kwargs
        )
        embs1 = embs[: len(s1)]
        embs2 = embs[len(s1) :]

        # Cosine similarity (embeddings assumed normalized)
        cos_sim = np.einsum("ij,ij->i", embs1, embs2)  # element-wise dot product

        pearson = pearsonr(cos_sim, labels).statistic
        spearman = spearmanr(cos_sim, labels).statistic

        return {
            "pearson": float(pearson),
            "spearman": float(spearman),
            "main_score": float(spearman),   # convention: always set main_score
        }


# ---------------------------------------------------------------------------
# Retrieval (Information Retrieval)
# ---------------------------------------------------------------------------

@dataclass
class RetrievalTask(Task):
    """
    Passage retrieval evaluated with NDCG@10 and Recall@k.

    Expects three HuggingFace datasets (following the BEIR format):
      - corpus:  id, text (+ optional title)
      - queries: id, text
      - qrels:   query-id, corpus-id, score

    Default: a small synthetic demo dataset so the task runs without
    downloading large files. Pass your own hf_* names to override.
    """

    name: str = "Retrieval-Demo"
    description: str = "Small synthetic retrieval benchmark"
    hf_corpus: Optional[str] = None
    hf_queries: Optional[str] = None
    hf_qrels: Optional[str] = None
    top_k: int = 10

    def _load_synthetic(self):
        """Tiny in-memory dataset for smoke-testing."""
        corpus = {
            "d1": "Paris is the capital of France.",
            "d2": "Berlin is the capital of Germany.",
            "d3": "Rome is the capital of Italy.",
            "d4": "Madrid is the capital of Spain.",
            "d5": "Lisbon is the capital of Portugal.",
            "d6": "The Eiffel Tower is located in Paris.",
            "d7": "The Brandenburg Gate is in Berlin.",
            "d8": "The Colosseum is in Rome.",
        }
        queries = {
            "q1": "What is the capital of France?",
            "q2": "Famous landmarks in Germany",
        }
        qrels = {
            "q1": {"d1": 1, "d6": 1},
            "q2": {"d2": 1, "d7": 1},
        }
        return corpus, queries, qrels

    def run(self, model, cache_dir: Path, **kwargs) -> dict[str, float]:
        if self.hf_corpus:
            raise NotImplementedError(
                "BEIR-format HF loading not yet implemented in this boilerplate. "
                "Use the synthetic demo or add your own loading logic."
            )

        corpus, queries, qrels = self._load_synthetic()

        doc_ids = list(corpus.keys())
        doc_texts = [corpus[d] for d in doc_ids]
        query_ids = list(queries.keys())
        query_texts = [queries[q] for q in query_ids]

        doc_embs = encode_with_cache(
            model, doc_texts, dataset_name=f"{self.name}__corpus", cache_dir=cache_dir, **kwargs
        )
        q_embs = encode_with_cache(
            model, query_texts, dataset_name=f"{self.name}__queries", cache_dir=cache_dir, **kwargs
        )

        # Similarity matrix: (n_queries, n_docs)
        scores = q_embs @ doc_embs.T

        ndcg_scores = []
        recall_scores = []

        for i, qid in enumerate(query_ids):
            relevant = set(qrels.get(qid, {}).keys())
            if not relevant:
                continue

            ranked_doc_ids = [doc_ids[j] for j in np.argsort(-scores[i])[: self.top_k]]

            # NDCG@k
            dcg = sum(
                1.0 / np.log2(rank + 2)
                for rank, did in enumerate(ranked_doc_ids)
                if did in relevant
            )
            ideal_dcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), self.top_k)))
            ndcg_scores.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)

            # Recall@k
            hits = sum(1 for did in ranked_doc_ids if did in relevant)
            recall_scores.append(hits / len(relevant))

        ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
        recall = float(np.mean(recall_scores)) if recall_scores else 0.0

        return {
            f"ndcg@{self.top_k}": ndcg,
            f"recall@{self.top_k}": recall,
            "main_score": ndcg,
        }


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

@dataclass
class ClusteringTask(Task):
    """
    Evaluates how well embeddings cluster by class label.
    Uses V-measure (harmonic mean of homogeneity and completeness).

    Expects a HuggingFace dataset with `text` and `label` columns.
    Default: 'ag_news' (4 categories of news).
    """

    name: str = "Clustering-AG-News"
    description: str = "Clustering on AG News (4 categories)"
    hf_dataset: str = "ag_news"
    split: str = "test"
    text_col: str = "text"
    label_col: str = "label"
    max_samples: int = 2000   # subsample to keep it fast

    def run(self, model, cache_dir: Path, **kwargs) -> dict[str, float]:
        from datasets import load_dataset
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import v_measure_score

        logger.info(f"[{self.name}] Loading dataset {self.hf_dataset} …")
        ds = load_dataset(self.hf_dataset, split=self.split)

        if len(ds) > self.max_samples:
            ds = ds.shuffle(seed=42).select(range(self.max_samples))

        texts = list(ds[self.text_col])
        labels = list(ds[self.label_col])
        n_clusters = len(set(labels))

        embs = encode_with_cache(
            model, texts, dataset_name=self.name, cache_dir=cache_dir, **kwargs
        )

        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        pred_labels = kmeans.fit_predict(embs)
        v_score = v_measure_score(labels, pred_labels)

        return {
            "v_measure": float(v_score),
            "main_score": float(v_score),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, Task] = {
    "sts": STSTask(),
    "retrieval": RetrievalTask(),
    "clustering": ClusteringTask(),
}


def get_tasks(names: list[str]) -> list[Task]:
    missing = set(names) - set(TASK_REGISTRY)
    if missing:
        raise ValueError(f"Unknown tasks: {missing}. Available: {set(TASK_REGISTRY)}")
    return [TASK_REGISTRY[n] for n in names]
