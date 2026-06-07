#!/usr/bin/env python3
"""Resumable bounded beam ceiling check for all benchmark tasks."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from vqc_molecule_gym.baselines.beam_search_agent import BeamSearchAgent
from vqc_molecule_gym.baselines.search_helpers import DEFAULT_REFINEMENT_DELTAS
from vqc_molecule_gym.benchmarks import load_tasks
from vqc_molecule_gym.evaluators.direct_energy import DirectEnergyEvaluator
from vqc_molecule_gym.logging.jsonl_logger import JsonlLogger
from vqc_molecule_gym.operators.operator_pool import build_operator_pool
from vqc_molecule_gym.schemas.trajectory import TrajectoryRecord

BENCHMARKS = [
    "c2h6_torsion_scan_v0",
    "h2_tiny",
    "h2o_angle_scan_v0",
    "h2o_dimer_distance_scan_v0",
    "h4_small",
    "lih_bond_scan_v0",
    "n2_bond_scan_v0",
]
ANGLE_GRID = (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3)
BEAM_WIDTH = 16
CANDIDATE_LIMIT = 64
MAX_OPERATORS = 4
REFINE_ANGLES = True
REFINEMENT_CANDIDATES = 3
REFINEMENT_DELTAS = DEFAULT_REFINEMENT_DELTAS
REFINEMENT_PASSES = 1


def existing_task_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    import json

    seen: set[str] = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = record.get("task_id")
            if isinstance(task_id, str):
                seen.add(task_id)
    return seen


def main() -> None:
    evaluator = DirectEnergyEvaluator()
    for benchmark in BENCHMARKS:
        output = Path(f"runs/beam_ceiling_check_{benchmark}.jsonl")
        logger = JsonlLogger(output)
        done = existing_task_ids(output)
        tasks = sorted(load_tasks(benchmark), key=lambda task: task.task_id)
        remaining = [task for task in tasks if task.task_id not in done]
        print(f"=== {benchmark}: {len(done)}/{len(tasks)} complete, {len(remaining)} remaining ===", flush=True)
        for task in remaining:
            agent = BeamSearchAgent(
                evaluator,
                beam_width=BEAM_WIDTH,
                max_operators=MAX_OPERATORS,
                candidate_limit=CANDIDATE_LIMIT,
                angle_grid=ANGLE_GRID,
                candidate_ranking="single_step",
                ranking_angle_grid=ANGLE_GRID,
                refine_angles=REFINE_ANGLES,
                refinement_candidates=REFINEMENT_CANDIDATES,
                refinement_deltas=REFINEMENT_DELTAS,
                refinement_passes=REFINEMENT_PASSES,
            )
            pool = build_operator_pool(
                task.operator_pool_id,
                num_qubits=task.active_space.qubits,
                num_electrons=task.active_space.electrons,
            )
            operator_ids = sorted(pool.ids)
            started = datetime.now(timezone.utc)
            print(f"Running {benchmark}/{task.task_id} at {started.isoformat()}", flush=True)
            action = agent.act(task, operator_ids)
            result = evaluator.evaluate_payload(task, action.model_dump())
            search_metadata = dict(agent.last_search_metadata)
            result.metadata = {
                **result.metadata,
                "baseline_search": {
                    **search_metadata,
                    "cached_action": False,
                    "profile": "beam_ceiling_check",
                    "objective": "delta_error_mha",
                    "refinement_enabled": REFINE_ANGLES,
                    "refinement_deltas": list(REFINEMENT_DELTAS),
                    "refinement_passes": REFINEMENT_PASSES,
                    "refinement_candidates": REFINEMENT_CANDIDATES,
                },
            }
            timestamp = datetime.now(timezone.utc)
            logger.append(
                TrajectoryRecord(
                    run_id=f"beam_ceiling_check_{benchmark}",
                    episode_id=task.task_id,
                    timestamp=timestamp,
                    agent="beam_search",
                    benchmark_id=benchmark,
                    task_id=task.task_id,
                    completion_raw=action.model_dump_json(),
                    action=action.model_dump(),
                    result=result.model_dump(),
                    software_versions={"vqc_molecule_gym": "0.1.0"},
                )
            )
            bs = result.metadata["baseline_search"]
            print(
                f"Done {benchmark}/{task.task_id}: error={result.energy_error_mha:.6f} mHa, "
                f"chem={result.chemical_accuracy}, evals={bs.get('evaluations')}, "
                f"refine_evals={bs.get('refinement_evaluations')}",
                flush=True,
            )
        print(f"Output: {output}", flush=True)


if __name__ == "__main__":
    main()
