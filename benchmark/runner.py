"""
runner.py — Top-level benchmark orchestrator.

Runs all requested tasks for all requested models, collects results,
and persists them to disk after each (model, task) pair so partial
runs are never lost.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import EmbeddingModel, load_model
from .tasks import Task, get_tasks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    model_name: str
    task_name: str
    metrics: dict[str, float]
    encode_time_s: float
    eval_time_s: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def main_score(self) -> float:
        return self.metrics.get("main_score", float("nan"))

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "task": self.task_name,
            **self.metrics,
            "encode_time_s": self.encode_time_s,
            "eval_time_s": self.eval_time_s,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRunner:
    """
    Args:
        model_configs:  List of model config dicts (see models.load_model).
        task_names:     List of task names from TASK_REGISTRY (or Task objects).
        output_dir:     Where to write result JSONs.
        cache_dir:      Where to cache embeddings.
        batch_size:     Encoding batch size passed to models.
        show_progress:  Show tqdm bars during encoding.
    """

    model_configs: list[dict]
    task_names: list[str]
    output_dir: Path = Path("results")
    cache_dir: Path = Path(".cache/embeddings")
    batch_size: int = 64
    show_progress: bool = True

    def run(self) -> list[TaskResult]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        tasks: list[Task] = get_tasks(self.task_names)
        all_results: list[TaskResult] = []

        for model_cfg in self.model_configs:
            model: EmbeddingModel = load_model(model_cfg)
            logger.info(f"\n{'='*60}\nModel: {model.name}\n{'='*60}")

            for task in tasks:
                logger.info(f"  Task: {task.name}")
                result = self._run_one(model, task)
                all_results.append(result)
                self._save_result(result)
                logger.info(f"  → main_score={result.main_score:.4f}  ({result.metrics})")

        self._save_summary(all_results)
        return all_results

    def _run_one(self, model: EmbeddingModel, task: Task) -> TaskResult:
        t0 = time.perf_counter()
        # Tasks call encode_with_cache internally, so we time the whole thing
        # but split encode time vs eval time by wrapping model.encode temporarily.
        encode_time_acc = [0.0]
        original_encode = model.encode

        def timed_encode(texts, **kw):
            t = time.perf_counter()
            result = original_encode(texts, **kw)
            encode_time_acc[0] += time.perf_counter() - t
            return result

        model.encode = timed_encode  # type: ignore[method-assign]
        try:
            metrics = task.run(
                model,
                cache_dir=self.cache_dir,
                batch_size=self.batch_size,
                show_progress=self.show_progress,
            )
        finally:
            model.encode = original_encode  # type: ignore[method-assign]

        total_time = time.perf_counter() - t0
        eval_time = total_time - encode_time_acc[0]

        return TaskResult(
            model_name=model.name,
            task_name=task.name,
            metrics=metrics,
            encode_time_s=round(encode_time_acc[0], 3),
            eval_time_s=round(eval_time, 3),
        )

    def _save_result(self, result: TaskResult):
        safe_model = result.model_name.replace("/", "__")
        path = self.output_dir / f"{safe_model}__{result.task_name}.json"
        path.write_text(json.dumps(result.to_dict(), indent=2))

    def _save_summary(self, results: list[TaskResult]):
        path = self.output_dir / "summary.json"
        path.write_text(json.dumps([r.to_dict() for r in results], indent=2))
        logger.info(f"\nSummary written to {path}")


# ---------------------------------------------------------------------------
# Convenience: results -> pandas DataFrame
# ---------------------------------------------------------------------------

def results_to_dataframe(results: list[TaskResult]):
    """Convert a list of TaskResults to a tidy pandas DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pip install pandas")

    rows = [r.to_dict() for r in results]
    return pd.DataFrame(rows)


def pivot_main_scores(results: list[TaskResult]):
    """Model × Task matrix of main_score values."""
    import pandas as pd

    df = results_to_dataframe(results)
    return df.pivot(index="model", columns="task", values="main_score")
