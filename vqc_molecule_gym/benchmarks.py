from pathlib import Path

from vqc_molecule_gym.schemas.task import TaskSpec
from vqc_molecule_gym.utils.io import read_jsonl_models


def benchmark_dir(benchmark_id: str, root: Path = Path("benchmarks")) -> Path:
    return root / benchmark_id


def tasks_path(benchmark_id: str, root: Path = Path("benchmarks")) -> Path:
    return benchmark_dir(benchmark_id, root) / "tasks.jsonl"


def load_tasks(benchmark_id: str, root: Path = Path("benchmarks")) -> list[TaskSpec]:
    return read_jsonl_models(tasks_path(benchmark_id, root), TaskSpec)
