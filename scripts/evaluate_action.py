#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from vqc_molecule_gym.benchmarks import load_tasks
from vqc_molecule_gym.chemistry.benchmarks import SUPPORTED_BENCHMARK_IDS
from vqc_molecule_gym.evaluators.direct_energy import DirectEnergyEvaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--benchmarks-root", default="benchmarks")
    args = parser.parse_args()

    benchmark_ids = [args.benchmark] if args.benchmark else _discover_benchmarks(Path(args.benchmarks_root))
    tasks = []
    for benchmark_id in benchmark_ids:
        path = Path(args.benchmarks_root) / benchmark_id / "tasks.jsonl"
        if path.exists():
            tasks.extend(load_tasks(benchmark_id, Path(args.benchmarks_root)))
    task_by_id = {task.task_id: task for task in tasks}
    if args.task_id not in task_by_id:
        raise SystemExit(f"Unknown task_id: {args.task_id}")

    action_payload = json.loads(args.action)
    result = DirectEnergyEvaluator().evaluate_payload(task_by_id[args.task_id], action_payload)
    print(result.model_dump_json(indent=2))


def _discover_benchmarks(root: Path) -> list[str]:
    discovered = [benchmark_id for benchmark_id in SUPPORTED_BENCHMARK_IDS if (root / benchmark_id / "tasks.jsonl").exists()]
    if discovered:
        return discovered
    return list(SUPPORTED_BENCHMARK_IDS)


if __name__ == "__main__":
    main()
