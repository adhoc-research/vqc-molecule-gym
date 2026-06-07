#!/usr/bin/env python3
import argparse
from pathlib import Path

from vqc_molecule_gym.benchmarks import load_tasks
from vqc_molecule_gym.evaluators.direct_energy import attach_reference
from vqc_molecule_gym.utils.io import write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-root", default="benchmarks")
    args = parser.parse_args()

    root = Path(args.output_root) / args.benchmark
    tasks = [attach_reference(task) for task in load_tasks(args.benchmark, Path(args.output_root))]
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
    print(f"Wrote references for {len(tasks)} tasks to {root / 'references.json'}")


if __name__ == "__main__":
    main()
