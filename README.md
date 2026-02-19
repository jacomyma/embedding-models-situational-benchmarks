# Embedding Situational Benchmarks

A lightweight, MTEB-inspired framework for comparing embedding models across tasks.

## WORK IN PROGRESS!

The code is currently being adapted to our own benchmarks.

## Structure

```
embedding_benchmark/
├── data/
│   ├── ...            — our benchmark datasets
├── benchmark/
│   ├── __init__.py    — public API
│   ├── models.py      — adapters (HuggingFace, OpenAI, …)
│   ├── tasks.py       — benchmark tasks
│   ├── cache.py       — disk caching of embeddings
│   └── runner.py      — orchestrator + result helpers
└── demo.ipynb         — interactive walkthrough
```

## Quickstart

```bash
pip install sentence-transformers datasets scipy scikit-learn pandas tqdm
```

```python
from benchmark import BenchmarkRunner
from pathlib import Path

runner = BenchmarkRunner(
    model_configs=[
        {"type": "sentence_transformer", "model": "BAAI/bge-small-en-v1.5"},
        {"type": "sentence_transformer", "model": "sentence-transformers/all-MiniLM-L6-v2"},
    ],
    task_names=["sts", "retrieval"],
    output_dir=Path("results"),
)

results = runner.run()
```

## Built-in Tasks

| Key           | Task class        | Metric         | Dataset             |
|---------------|-------------------|----------------|---------------------|
| `sts`         | `STSTask`         | Spearman ρ     | STS-Benchmark       |
| `retrieval`   | `RetrievalTask`   | NDCG@10        | Synthetic (demo)    |
| `clustering`  | `ClusteringTask`  | V-measure      | AG News (subsample) |


## Adding a Model

Add a dict to `model_configs`:

```python
# HuggingFace / sentence-transformers
{"type": "sentence_transformer", "model": "intfloat/e5-large-v2", "device": "cuda"}

# OpenAI (needs OPENAI_API_KEY)
{"type": "openai", "model": "text-embedding-3-large", "dimensions": 1024}
```


## Adding a Task

```python
from benchmark.tasks import Task, TASK_REGISTRY
from benchmark.cache import encode_with_cache
from dataclasses import dataclass
from pathlib import Path

@dataclass
class MyTask(Task):
    name: str = "My-Task"

    def run(self, model, cache_dir: Path, **kwargs) -> dict[str, float]:
        texts, labels = my_data_loader()
        embs = encode_with_cache(model, texts, self.name, cache_dir, **kwargs)
        score = my_eval_logic(embs, labels)
        return {"my_metric": score, "main_score": score}

TASK_REGISTRY["my-task"] = MyTask()
```


## Key Design Decisions

- **Cache first**: embeddings are cached to `.cache/embeddings/` keyed by model + dataset + text hash — re-running after a crash doesn't re-encode.
- **Write on completion**: each `(model, task)` result is written to `results/` immediately, so partial sweeps are recoverable.
- **No framework lock-in**: the `EmbeddingModel.encode()` interface is two lines; wrapping a new backend takes ~20 lines.
- **Config-driven**: models are plain dicts, easy to serialize, log, or generate programmatically.
