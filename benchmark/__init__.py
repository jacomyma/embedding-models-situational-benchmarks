"""
embedding_benchmark — A lightweight MTEB-inspired benchmark scaffold.
"""

from .models import EmbeddingModel, SentenceTransformerModel, OpenAIEmbeddingModel, load_model
from .tasks import Task, STSTask, RetrievalTask, ClusteringTask, get_tasks, TASK_REGISTRY
from .runner import BenchmarkRunner, TaskResult, results_to_dataframe, pivot_main_scores
from .cache import encode_with_cache

__all__ = [
    "EmbeddingModel",
    "SentenceTransformerModel",
    "OpenAIEmbeddingModel",
    "load_model",
    "Task",
    "STSTask",
    "RetrievalTask",
    "ClusteringTask",
    "get_tasks",
    "TASK_REGISTRY",
    "BenchmarkRunner",
    "TaskResult",
    "results_to_dataframe",
    "pivot_main_scores",
    "encode_with_cache",
]
