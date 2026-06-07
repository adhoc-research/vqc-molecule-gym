#!/usr/bin/env python3
import argparse
from pathlib import Path

from vqc_molecule_gym.chemistry.benchmarks import SUPPORTED_BENCHMARK_IDS, generate_benchmark
from vqc_molecule_gym.evaluators.direct_energy import attach_reference
from vqc_molecule_gym.operators.operator_pool import build_operator_pool
from vqc_molecule_gym.utils.io import write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=SUPPORTED_BENCHMARK_IDS)
    parser.add_argument("--output-root", default="benchmarks")
    args = parser.parse_args()

    tasks = [attach_reference(task) for task in generate_benchmark(args.benchmark)]
    root = Path(args.output_root) / args.benchmark
    write_jsonl(root / "tasks.jsonl", tasks)
    write_json(
        root / "references.json",
        {
            task.task_id: {
                "method": task.reference.method,
                "energy_hartree": task.reference.energy_hartree,
                "chemical_accuracy_mha": task.reference.chemical_accuracy_mha,
                "casci_energy_hartree": task.reference.casci_energy_hartree,
            }
            for task in tasks
        },
    )
    first = tasks[0]
    pool = build_operator_pool(
        first.operator_pool_id,
        num_qubits=first.active_space.qubits,
        num_electrons=first.active_space.electrons,
    )
    write_json(root / "operator_pool.json", pool.to_json())
    write_json(root / "constraints.json", first.constraints.model_dump())
    print(f"Wrote {len(tasks)} tasks to {root}")


if __name__ == "__main__":
    main()
