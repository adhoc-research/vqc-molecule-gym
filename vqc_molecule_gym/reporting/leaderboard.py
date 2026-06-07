from __future__ import annotations

import glob
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from vqc_molecule_gym.schemas.trajectory import TrajectoryRecord
from vqc_molecule_gym.utils.io import read_jsonl_models

ParetoScope = Literal["task", "benchmark", "both"]


@dataclass(frozen=True)
class RecordPoint:
    benchmark_id: str
    task_id: str
    run_id: str
    episode_id: str
    agent: str
    valid: bool
    reward: float | None
    energy_error_mha: float | None
    chemical_accuracy: bool
    depth: int | None
    num_operators: int | None
    action_hash: str | None


@dataclass(frozen=True)
class RunSummary:
    benchmark_id: str
    agent: str
    run_id: str
    episodes: int
    valid: int
    invalid: int
    valid_rate: float
    best_reward: float | None
    best_error_mha: float | None
    chemical_accuracy_rate: float


@dataclass(frozen=True)
class GroupSummary:
    benchmark_id: str
    agent: str
    runs: int
    episodes: int
    valid_rate: float
    best_reward: float | None
    best_error_mha: float | None
    chemical_accuracy_rate: float


@dataclass(frozen=True)
class TaskWinner:
    benchmark_id: str
    task_id: str
    run_id: str
    episode_id: str
    agent: str
    energy_error_mha: float
    reward: float | None
    depth: int | None
    num_operators: int | None
    action_hash: str | None


@dataclass(frozen=True)
class ParetoPoint:
    benchmark_id: str
    task_id: str | None
    run_id: str
    episode_id: str
    agent: str
    energy_error_mha: float
    depth: int
    reward: float | None
    num_operators: int | None
    action_hash: str | None


@dataclass(frozen=True)
class BaseComparison:
    benchmark_id: str
    task_id: str
    agent: str
    run_id: str
    episode_id: str
    base_error_mha: float
    best_nonempty_error_mha: float | None
    improvement_over_base_mha: float | None
    best_nonempty_reward: float | None


def expand_run_paths(patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.update(Path(match) for match in matches)
        else:
            path = Path(pattern)
            if path.exists():
                paths.add(path)
    return sorted(paths, key=lambda p: str(p))


def load_records(patterns: Iterable[str]) -> list[TrajectoryRecord]:
    paths = expand_run_paths(patterns)
    if not paths:
        raise FileNotFoundError("No run JSONL files matched --runs")
    records: list[TrajectoryRecord] = []
    for path in paths:
        records.extend(read_jsonl_models(path, TrajectoryRecord))
    return records


def to_point(record: TrajectoryRecord) -> RecordPoint:
    result = record.result or {}
    metrics = result.get("circuit_metrics") or {}
    return RecordPoint(
        benchmark_id=record.benchmark_id,
        task_id=record.task_id,
        run_id=record.run_id,
        episode_id=record.episode_id,
        agent=record.agent,
        valid=bool(result.get("valid", False)),
        reward=_float_or_none(result.get("reward")),
        energy_error_mha=_float_or_none(result.get("energy_error_mha")),
        chemical_accuracy=bool(result.get("chemical_accuracy", False)),
        depth=_int_or_none(metrics.get("depth")),
        num_operators=_int_or_none(metrics.get("num_operators")),
        action_hash=result.get("action_hash") if isinstance(result.get("action_hash"), str) else None,
    )


def summarize_runs(records: Iterable[TrajectoryRecord]) -> list[RunSummary]:
    groups: dict[tuple[str, str, str], list[RecordPoint]] = defaultdict(list)
    for record in records:
        groups[(record.benchmark_id, record.agent, record.run_id)].append(to_point(record))
    summaries = [_run_summary(key, points) for key, points in groups.items()]
    return sorted(summaries, key=lambda s: (s.benchmark_id, s.agent, s.run_id))


def summarize_groups(records: Iterable[TrajectoryRecord]) -> list[GroupSummary]:
    groups: dict[tuple[str, str], list[RecordPoint]] = defaultdict(list)
    run_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        key = (record.benchmark_id, record.agent)
        groups[key].append(to_point(record))
        run_ids[key].add(record.run_id)
    summaries = []
    for (benchmark_id, agent), points in groups.items():
        valid_points = [p for p in points if p.valid]
        valid_with_error = [p.energy_error_mha for p in valid_points if p.energy_error_mha is not None]
        valid_rewards = [p.reward for p in valid_points if p.reward is not None]
        summaries.append(
            GroupSummary(
                benchmark_id=benchmark_id,
                agent=agent,
                runs=len(run_ids[(benchmark_id, agent)]),
                episodes=len(points),
                valid_rate=(len(valid_points) / len(points)) if points else 0.0,
                best_reward=max(valid_rewards) if valid_rewards else None,
                best_error_mha=min(valid_with_error) if valid_with_error else None,
                chemical_accuracy_rate=(sum(p.chemical_accuracy for p in valid_points) / len(valid_points)) if valid_points else 0.0,
            )
        )
    return sorted(summaries, key=lambda s: (s.benchmark_id, s.agent))


def select_task_winners(records: Iterable[TrajectoryRecord]) -> list[TaskWinner]:
    groups: dict[tuple[str, str], list[RecordPoint]] = defaultdict(list)
    for record in records:
        point = to_point(record)
        if _is_search_leaderboard_point(point):
            groups[(point.benchmark_id, point.task_id)].append(point)
    winners: list[TaskWinner] = []
    for (_, _), points in groups.items():
        best = sorted(points, key=_best_key)[0]
        winners.append(
            TaskWinner(
                benchmark_id=best.benchmark_id,
                task_id=best.task_id,
                run_id=best.run_id,
                episode_id=best.episode_id,
                agent=best.agent,
                energy_error_mha=best.energy_error_mha or 0.0,
                reward=best.reward,
                depth=best.depth,
                num_operators=best.num_operators,
                action_hash=best.action_hash,
            )
        )
    return sorted(winners, key=lambda w: (w.benchmark_id, w.task_id, w.energy_error_mha, w.depth or 10**9))


def pareto_frontier(points: Iterable[RecordPoint], task_id: str | None = None) -> list[ParetoPoint]:
    valid = [p for p in points if _is_search_leaderboard_point(p) and p.depth is not None]
    frontier: list[RecordPoint] = []
    for point in valid:
        dominated = False
        for other in valid:
            if other is point:
                continue
            if (
                (other.energy_error_mha or math.inf) <= (point.energy_error_mha or math.inf)
                and (other.depth or 10**9) <= (point.depth or 10**9)
                and ((other.energy_error_mha != point.energy_error_mha) or (other.depth != point.depth))
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(point)
    # Collapse exact duplicate coordinates to the better reward/tie metadata row.
    by_coord: dict[tuple[float, int], RecordPoint] = {}
    for point in sorted(frontier, key=_pareto_sort_key):
        coord = (point.energy_error_mha or 0.0, point.depth or 0)
        by_coord.setdefault(coord, point)
    return [
        ParetoPoint(
            benchmark_id=p.benchmark_id,
            task_id=task_id if task_id is not None else p.task_id,
            run_id=p.run_id,
            episode_id=p.episode_id,
            agent=p.agent,
            energy_error_mha=p.energy_error_mha or 0.0,
            depth=p.depth or 0,
            reward=p.reward,
            num_operators=p.num_operators,
            action_hash=p.action_hash,
        )
        for p in sorted(by_coord.values(), key=_pareto_sort_key)
    ]


def task_pareto_frontiers(records: Iterable[TrajectoryRecord]) -> dict[tuple[str, str], list[ParetoPoint]]:
    groups: dict[tuple[str, str], list[RecordPoint]] = defaultdict(list)
    for record in records:
        point = to_point(record)
        groups[(point.benchmark_id, point.task_id)].append(point)
    return {key: pareto_frontier(points, task_id=key[1]) for key, points in sorted(groups.items())}


def benchmark_pareto_frontiers(records: Iterable[TrajectoryRecord]) -> dict[str, list[ParetoPoint]]:
    groups: dict[str, list[RecordPoint]] = defaultdict(list)
    for record in records:
        point = to_point(record)
        groups[point.benchmark_id].append(point)
    return {key: pareto_frontier(points, task_id=None) for key, points in sorted(groups.items())}


def base_comparisons(records: Iterable[TrajectoryRecord]) -> list[BaseComparison]:
    best_by_task: dict[tuple[str, str], BaseComparison] = {}
    for record in records:
        point = to_point(record)
        metadata = record.result.get("metadata") if isinstance(record.result, dict) else None
        search_metadata = metadata.get("baseline_search") if isinstance(metadata, dict) else None
        base_error = _metadata_float(search_metadata, "base_error_mha")
        best_nonempty_error = _metadata_float(search_metadata, "best_nonempty_by_error_mha")
        if best_nonempty_error is None:
            best_nonempty_error = _metadata_float(search_metadata, "best_nonempty_error_mha")
        best_nonempty_reward = _metadata_float(search_metadata, "best_nonempty_error_reward")
        if best_nonempty_reward is None:
            best_nonempty_reward = _metadata_float(search_metadata, "best_nonempty_reward")

        if base_error is None and point.num_operators == 0:
            base_error = point.energy_error_mha
        if best_nonempty_error is None and _is_search_leaderboard_point(point):
            best_nonempty_error = point.energy_error_mha
            best_nonempty_reward = point.reward
        if base_error is None:
            continue
        improvement = base_error - best_nonempty_error if best_nonempty_error is not None else None
        comparison = BaseComparison(
            benchmark_id=point.benchmark_id,
            task_id=point.task_id,
            agent=point.agent,
            run_id=point.run_id,
            episode_id=point.episode_id,
            base_error_mha=base_error,
            best_nonempty_error_mha=best_nonempty_error,
            improvement_over_base_mha=improvement,
            best_nonempty_reward=best_nonempty_reward,
        )
        key = (point.benchmark_id, point.task_id)
        current = best_by_task.get(key)
        if current is None or _base_comparison_key(comparison) < _base_comparison_key(current):
            best_by_task[key] = comparison
    return sorted(best_by_task.values(), key=lambda c: (c.benchmark_id, c.task_id))


def render_report(records: list[TrajectoryRecord], *, top_k: int = 10, pareto_scope: ParetoScope = "both") -> str:
    lines: list[str] = ["# QChem VQC Leaderboard", ""]
    lines.extend(_render_run_summaries(summarize_runs(records)))
    lines.extend(_render_group_summaries(summarize_groups(records)))
    lines.extend(_render_base_comparisons(base_comparisons(records), top_k=top_k))
    lines.extend(_render_task_winners(select_task_winners(records), top_k=top_k))
    if pareto_scope in ("task", "both"):
        lines.extend(_render_task_pareto(task_pareto_frontiers(records), top_k=top_k))
    if pareto_scope in ("benchmark", "both"):
        lines.extend(_render_benchmark_pareto(benchmark_pareto_frontiers(records), top_k=top_k))
    return "\n".join(lines).rstrip() + "\n"


def _run_summary(key: tuple[str, str, str], points: list[RecordPoint]) -> RunSummary:
    benchmark_id, agent, run_id = key
    valid_points = [p for p in points if p.valid]
    valid_with_error = [p.energy_error_mha for p in valid_points if p.energy_error_mha is not None]
    valid_rewards = [p.reward for p in valid_points if p.reward is not None]
    return RunSummary(
        benchmark_id=benchmark_id,
        agent=agent,
        run_id=run_id,
        episodes=len(points),
        valid=len(valid_points),
        invalid=len(points) - len(valid_points),
        valid_rate=(len(valid_points) / len(points)) if points else 0.0,
        best_reward=max(valid_rewards) if valid_rewards else None,
        best_error_mha=min(valid_with_error) if valid_with_error else None,
        chemical_accuracy_rate=(sum(p.chemical_accuracy for p in valid_points) / len(valid_points)) if valid_points else 0.0,
    )


def _best_key(point: RecordPoint) -> tuple[float, int, int, float, str, str]:
    return (
        point.energy_error_mha if point.energy_error_mha is not None else math.inf,
        point.depth if point.depth is not None else 10**9,
        point.num_operators if point.num_operators is not None else 10**9,
        -(point.reward if point.reward is not None else -math.inf),
        point.run_id,
        point.episode_id,
    )


def _pareto_sort_key(point: RecordPoint) -> tuple[float, int, int, float, str, str]:
    return _best_key(point)


def _is_search_leaderboard_point(point: RecordPoint) -> bool:
    return point.valid and point.energy_error_mha is not None and point.num_operators != 0


def _base_comparison_key(comparison: BaseComparison) -> tuple[float, float, str, str]:
    return (
        comparison.best_nonempty_error_mha if comparison.best_nonempty_error_mha is not None else math.inf,
        -(comparison.improvement_over_base_mha if comparison.improvement_over_base_mha is not None else -math.inf),
        comparison.run_id,
        comparison.episode_id,
    )


def _metadata_float(metadata: object, key: str) -> float | None:
    if not isinstance(metadata, dict):
        return None
    return _float_or_none(metadata.get(key))


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _hash(value: str | None) -> str:
    if not value:
        return "-"
    return value.split(":")[-1][:12]


def _render_run_summaries(summaries: list[RunSummary]) -> list[str]:
    lines = ["## Run summaries", "", "| benchmark | agent | run | episodes | valid% | best reward | best error mHa | chem acc% |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s.benchmark_id} | {s.agent} | {s.run_id} | {s.episodes} | {100*s.valid_rate:.1f} | {_fmt(s.best_reward, 6)} | {_fmt(s.best_error_mha, 3)} | {100*s.chemical_accuracy_rate:.1f} |")
    lines.append("")
    return lines


def _render_group_summaries(summaries: list[GroupSummary]) -> list[str]:
    lines = ["## Agent/benchmark summaries", "", "| benchmark | agent | runs | episodes | valid% | best reward | best error mHa | chem acc% |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s.benchmark_id} | {s.agent} | {s.runs} | {s.episodes} | {100*s.valid_rate:.1f} | {_fmt(s.best_reward, 6)} | {_fmt(s.best_error_mha, 3)} | {100*s.chemical_accuracy_rate:.1f} |")
    lines.append("")
    return lines


def _render_base_comparisons(comparisons: list[BaseComparison], *, top_k: int) -> list[str]:
    lines = ["## Base vs best non-empty", "", "| benchmark | task | agent | run | episode | base error mHa | best non-empty error mHa | improvement mHa | best non-empty reward |", "|---|---|---|---|---|---:|---:|---:|---:|"]
    for c in comparisons[:top_k]:
        lines.append(f"| {c.benchmark_id} | {c.task_id} | {c.agent} | {c.run_id} | {c.episode_id} | {c.base_error_mha:.3f} | {_fmt(c.best_nonempty_error_mha, 3)} | {_fmt(c.improvement_over_base_mha, 3)} | {_fmt(c.best_nonempty_reward, 6)} |")
    lines.append("")
    return lines


def _render_task_winners(winners: list[TaskWinner], *, top_k: int) -> list[str]:
    lines = ["## Best circuit per task", "", "| benchmark | task | agent | run | episode | error mHa | reward | depth | ops | action |", "|---|---|---|---|---|---:|---:|---:|---:|---|"]
    for w in winners[:top_k]:
        lines.append(f"| {w.benchmark_id} | {w.task_id} | {w.agent} | {w.run_id} | {w.episode_id} | {w.energy_error_mha:.3f} | {_fmt(w.reward, 6)} | {w.depth if w.depth is not None else '-'} | {w.num_operators if w.num_operators is not None else '-'} | {_hash(w.action_hash)} |")
    lines.append("")
    return lines


def _render_task_pareto(frontiers: dict[tuple[str, str], list[ParetoPoint]], *, top_k: int) -> list[str]:
    lines = ["## Per-task Pareto frontiers", ""]
    for (benchmark_id, task_id), points in frontiers.items():
        if not points:
            continue
        lines.extend([f"### {benchmark_id} / {task_id}", "", "| error mHa | depth | agent | run | episode | reward | ops | action |", "|---:|---:|---|---|---|---:|---:|---|"])
        for p in points[:top_k]:
            lines.append(f"| {p.energy_error_mha:.3f} | {p.depth} | {p.agent} | {p.run_id} | {p.episode_id} | {_fmt(p.reward, 6)} | {p.num_operators if p.num_operators is not None else '-'} | {_hash(p.action_hash)} |")
        lines.append("")
    return lines


def _render_benchmark_pareto(frontiers: dict[str, list[ParetoPoint]], *, top_k: int) -> list[str]:
    lines = ["## Per-benchmark Pareto summary", "", "| benchmark | task | error mHa | depth | agent | run | episode | reward | ops | action |", "|---|---|---:|---:|---|---|---|---:|---:|---|"]
    for benchmark_id, points in frontiers.items():
        for p in points[:top_k]:
            lines.append(f"| {benchmark_id} | {p.task_id or '-'} | {p.energy_error_mha:.3f} | {p.depth} | {p.agent} | {p.run_id} | {p.episode_id} | {_fmt(p.reward, 6)} | {p.num_operators if p.num_operators is not None else '-'} | {_hash(p.action_hash)} |")
    lines.append("")
    return lines
