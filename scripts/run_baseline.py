#!/usr/bin/env python3
import argparse
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vqc_molecule_gym.baselines.beam_search_agent import BeamSearchAgent
from vqc_molecule_gym.baselines.greedy_agent import GreedyAgent
from vqc_molecule_gym.baselines.random_agent import RandomAgent
from vqc_molecule_gym.baselines.search_helpers import DEFAULT_REFINEMENT_DELTAS
from vqc_molecule_gym.benchmarks import load_tasks
from vqc_molecule_gym.evaluators.direct_energy import DirectEnergyEvaluator
from vqc_molecule_gym.logging.jsonl_logger import JsonlLogger
from vqc_molecule_gym.operators.operator_pool import build_operator_pool
from vqc_molecule_gym.schemas.action import ActionSpec
from vqc_molecule_gym.schemas.trajectory import TrajectoryRecord

SEARCH_AGENTS = {"greedy", "beam_search"}
DEFAULT_ANGLE_GRID = (-0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5)
DEFAULT_CANDIDATE_RANKING = "single_step"
DEFAULT_REFINE_ANGLES = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--agent", default="random", choices=["random", "greedy", "beam", "beam_search"])
    parser.add_argument("--evaluator", default="direct_energy", choices=["direct_energy"])
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--max-operators", type=int, default=3)
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument(
        "--angle-grid",
        default=",".join(str(value) for value in DEFAULT_ANGLE_GRID),
        help="Comma-separated angle grid in radians for greedy/beam parameter search.",
    )
    parser.add_argument(
        "--candidate-ranking",
        default=DEFAULT_CANDIDATE_RANKING,
        choices=["lexicographic", "single_step"],
        help="How greedy/beam choose the candidate operators considered after --candidate-limit.",
    )
    parser.add_argument(
        "--ranking-angle-grid",
        default=None,
        help="Optional comma-separated angle grid for single-step candidate ranking. Defaults to --angle-grid.",
    )
    parser.add_argument(
        "--refine-angles",
        default=DEFAULT_REFINE_ANGLES,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable local coordinate refinement of promising non-empty search candidates.",
    )
    parser.add_argument("--refinement-candidates", type=int, default=3)
    parser.add_argument(
        "--refinement-deltas",
        default=",".join(str(value) for value in DEFAULT_REFINEMENT_DELTAS),
        help="Comma-separated positive angle deltas in radians for local refinement.",
    )
    parser.add_argument("--refinement-passes", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    try:
        args.angle_grid = _parse_angle_grid(args.angle_grid)
        args.ranking_angle_grid = _parse_angle_grid(args.ranking_angle_grid) if args.ranking_angle_grid else args.angle_grid
        args.refinement_deltas = _parse_refinement_deltas(args.refinement_deltas)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    agent_name = _normalize_agent_name(args.agent)
    tasks = sorted(load_tasks(args.benchmark), key=lambda task: task.task_id)
    counts = _balanced_counts(args.episodes, len(tasks))
    run_id = f"{args.benchmark}_{agent_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output = Path(args.output or f"runs/{run_id}.jsonl")
    logger = JsonlLogger(output)
    evaluator = DirectEnergyEvaluator()
    agent = _build_agent(agent_name, evaluator, args)

    rewards: list[float] = []
    invalid = 0
    episode_idx = 0
    action_cache: dict[str, ActionSpec] = {}
    search_metadata_cache: dict[str, dict[str, object]] = {}
    search_evaluations = 0

    for task, count in zip(tasks, counts, strict=True):
        pool = build_operator_pool(
            task.operator_pool_id,
            num_qubits=task.active_space.qubits,
            num_electrons=task.active_space.electrons,
        )
        operator_ids = sorted(pool.ids)
        for _ in range(count):
            if agent_name in SEARCH_AGENTS:
                if task.task_id not in action_cache:
                    action_cache[task.task_id] = agent.act(task, operator_ids)
                    search_metadata_cache[task.task_id] = dict(getattr(agent, "last_search_metadata", {}))
                    search_evaluations += int(getattr(agent, "last_search_evaluations", 0))
                action = action_cache[task.task_id]
            else:
                action = agent.act(task, operator_ids)
            result = evaluator.evaluate_payload(task, action.model_dump())
            if agent_name in SEARCH_AGENTS:
                result.metadata = {
                    **result.metadata,
                    "baseline_search": {
                        **search_metadata_cache.get(task.task_id, {}),
                        "cached_action": task.task_id in action_cache,
                    },
                }
            rewards.append(result.reward)
            invalid += int(not result.valid)
            episode_idx += 1
            logger.append(
                TrajectoryRecord(
                    run_id=run_id,
                    episode_id=f"ep_{episode_idx:06d}",
                    timestamp=datetime.now(timezone.utc),
                    agent=agent_name,
                    benchmark_id=args.benchmark,
                    task_id=task.task_id,
                    completion_raw=action.model_dump_json(),
                    action=action.model_dump(),
                    result=result.model_dump(),
                    software_versions={"vqc_molecule_gym": "0.1.0"},
                )
            )

    best = max(rewards) if rewards else float("nan")
    print(f"Benchmark: {args.benchmark}")
    print(f"Evaluator: {args.evaluator}")
    print(f"Agent: {agent_name}")
    print(f"Episodes: {episode_idx}")
    if agent_name in SEARCH_AGENTS:
        print(f"Search evaluations: {search_evaluations}")
    print(f"Best reward: {best:.6f}")
    print(f"Invalid action rate: {invalid / episode_idx if episode_idx else 0.0:.3f}")
    print(f"Output: {output}")


def _normalize_agent_name(agent: str) -> str:
    return "beam_search" if agent == "beam" else agent


def _build_agent(agent_name: str, evaluator: DirectEnergyEvaluator, args: argparse.Namespace) -> Any:
    if agent_name == "random":
        return RandomAgent(random.Random(args.seed))
    if agent_name == "greedy":
        return GreedyAgent(
            evaluator,
            max_operators=args.max_operators,
            candidate_limit=args.candidate_limit,
            angle_grid=getattr(args, "angle_grid", DEFAULT_ANGLE_GRID),
            candidate_ranking=getattr(args, "candidate_ranking", "lexicographic"),
            ranking_angle_grid=getattr(args, "ranking_angle_grid", getattr(args, "angle_grid", DEFAULT_ANGLE_GRID)),
            refine_angles=getattr(args, "refine_angles", False),
            refinement_candidates=getattr(args, "refinement_candidates", 3),
            refinement_deltas=getattr(args, "refinement_deltas", DEFAULT_REFINEMENT_DELTAS),
            refinement_passes=getattr(args, "refinement_passes", 1),
        )
    if agent_name == "beam_search":
        return BeamSearchAgent(
            evaluator,
            beam_width=args.beam_width,
            max_operators=args.max_operators,
            candidate_limit=args.candidate_limit,
            angle_grid=getattr(args, "angle_grid", DEFAULT_ANGLE_GRID),
            candidate_ranking=getattr(args, "candidate_ranking", "lexicographic"),
            ranking_angle_grid=getattr(args, "ranking_angle_grid", getattr(args, "angle_grid", DEFAULT_ANGLE_GRID)),
            refine_angles=getattr(args, "refine_angles", False),
            refinement_candidates=getattr(args, "refinement_candidates", 3),
            refinement_deltas=getattr(args, "refinement_deltas", DEFAULT_REFINEMENT_DELTAS),
            refinement_passes=getattr(args, "refinement_passes", 1),
        )
    raise ValueError(f"Unknown agent: {agent_name}")


def _parse_angle_grid(value: str) -> tuple[float, ...]:
    try:
        angles = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("angle grids must be comma-separated floats") from exc
    if not angles:
        raise argparse.ArgumentTypeError("angle grids must contain at least one angle")
    if any(angle < -0.5 or angle > 0.5 for angle in angles):
        raise argparse.ArgumentTypeError("angle-grid values must be in [-0.5, 0.5] radians")
    return angles


def _parse_refinement_deltas(value: str) -> tuple[float, ...]:
    deltas = _parse_angle_grid(value)
    if any(delta == 0.0 for delta in deltas):
        raise argparse.ArgumentTypeError("--refinement-deltas must not contain zero")
    return tuple(abs(delta) for delta in deltas)


def _balanced_counts(total: int, buckets: int) -> list[int]:
    base, remainder = divmod(total, buckets)
    return [base + (1 if idx < remainder else 0) for idx in range(buckets)]


if __name__ == "__main__":
    main()
