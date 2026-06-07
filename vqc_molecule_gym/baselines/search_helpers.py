from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal, Sequence

from vqc_molecule_gym.baselines.types import SearchEvaluator
from vqc_molecule_gym.schemas.action import ActionSpec
from vqc_molecule_gym.schemas.result import EvalResult
from vqc_molecule_gym.schemas.task import TaskSpec

ANGLE_MIN = -0.5
ANGLE_MAX = 0.5
CandidateRanking = Literal["lexicographic", "single_step"]
DEFAULT_REFINEMENT_DELTAS: tuple[float, ...] = (0.025, 0.0125)


@dataclass(frozen=True)
class SearchEntry:
    sequence: tuple[str, ...]
    parameters: tuple[float, ...]
    result: EvalResult


@dataclass(frozen=True)
class CandidateRankingResult:
    candidates: list[str]
    entries: list[SearchEntry]
    metadata: dict[str, object]


@dataclass(frozen=True)
class RefinementResult:
    entries: list[SearchEntry]
    best_entry: SearchEntry | None
    evaluations: int
    metadata: dict[str, object]


class SearchCache:
    """Per-task evaluator cache keyed by `(operator_sequence, parameters)`."""

    def __init__(self, evaluator: SearchEvaluator, task: TaskSpec, *, shots: int) -> None:
        self.evaluator = evaluator
        self.task = task
        self.shots = shots
        self._cache: dict[tuple[tuple[str, ...], tuple[float, ...]], EvalResult] = {}
        self.base_error_mha: float | None = None

    @property
    def evaluations(self) -> int:
        return len(self._cache)

    def evaluate(self, sequence: Sequence[str], parameters: Sequence[float]) -> EvalResult:
        key = (tuple(sequence), tuple(float(value) for value in parameters))
        if key not in self._cache:
            action = ActionSpec(operator_sequence=list(key[0]), parameters=list(key[1]), shots=self.shots)
            result = self.evaluator.evaluate_payload(self.task, action.model_dump())
            if not key[0] and result.energy_error_mha is not None:
                self.base_error_mha = result.energy_error_mha
            self._annotate_energy_improvement(result)
            self._cache[key] = result
        return self._cache[key]

    def _annotate_energy_improvement(self, result: EvalResult) -> None:
        if self.base_error_mha is None or result.energy_error_mha is None:
            return
        metadata = dict(result.metadata)
        metadata["energy_improvement"] = {
            "base_error_mha": self.base_error_mha,
            "candidate_error_mha": result.energy_error_mha,
            "delta_error_mha": self.base_error_mha - result.energy_error_mha,
        }
        result.metadata = metadata

    def entry(self, sequence: Sequence[str], parameters: Sequence[float]) -> SearchEntry:
        key = (tuple(sequence), tuple(float(value) for value in parameters))
        return SearchEntry(key[0], key[1], self.evaluate(key[0], key[1]))


def sort_key(
    sequence: tuple[str, ...],
    parameters: tuple[float, ...],
    result: EvalResult,
) -> tuple[float, int, tuple[str, ...], tuple[float, ...]]:
    # Lower is better for sorting: highest reward, then shortest sequence, lexicographic sequence, then parameters.
    return (-result.reward, len(sequence), sequence, parameters)


def entry_sort_key(entry: SearchEntry) -> tuple[float, int, tuple[str, ...], tuple[float, ...]]:
    return sort_key(entry.sequence, entry.parameters, entry.result)


def delta_error_mha(entry: SearchEntry) -> float | None:
    metadata = entry.result.metadata.get("energy_improvement") if isinstance(entry.result.metadata, dict) else None
    if isinstance(metadata, dict):
        value = metadata.get("delta_error_mha")
        if isinstance(value, int | float) and math.isfinite(float(value)):
            return float(value)
    return None


def improvement_sort_key(entry: SearchEntry) -> tuple[int, float, float, int, tuple[str, ...], tuple[float, ...]]:
    delta = delta_error_mha(entry)
    reward_key, length_key, sequence_key, parameters_key = entry_sort_key(entry)
    if delta is None:
        return (1, 0.0, reward_key, length_key, sequence_key, parameters_key)
    return (0, -delta, reward_key, length_key, sequence_key, parameters_key)


def is_valid_nonempty(entry: SearchEntry) -> bool:
    return bool(entry.sequence) and entry.result.valid


def select_top_valid_nonempty(entries: Iterable[SearchEntry], limit: int) -> list[SearchEntry]:
    if limit <= 0:
        return []
    best_by_action: dict[tuple[tuple[str, ...], tuple[float, ...]], SearchEntry] = {}
    for entry in entries:
        if not is_valid_nonempty(entry):
            continue
        key = (entry.sequence, entry.parameters)
        current = best_by_action.get(key)
        if current is None or improvement_sort_key(entry) < improvement_sort_key(current):
            best_by_action[key] = entry
    return sorted(best_by_action.values(), key=improvement_sort_key)[:limit]


def best_valid_nonempty_by_reward(entries: Iterable[SearchEntry]) -> SearchEntry | None:
    valid = [entry for entry in entries if is_valid_nonempty(entry)]
    return min(valid, key=entry_sort_key) if valid else None


def best_valid_nonempty_by_improvement(entries: Iterable[SearchEntry]) -> SearchEntry | None:
    valid = [entry for entry in entries if is_valid_nonempty(entry)]
    return min(valid, key=improvement_sort_key) if valid else None


def best_valid_nonempty_by_error(entries: Iterable[SearchEntry]) -> SearchEntry | None:
    valid = [entry for entry in entries if is_valid_nonempty(entry) and entry.result.energy_error_mha is not None]
    return min(valid, key=lambda entry: (entry.result.energy_error_mha or float("inf"), entry_sort_key(entry))) if valid else None


def base_vs_nonempty_metadata(base: SearchEntry, entries: Iterable[SearchEntry]) -> dict[str, object]:
    by_reward = best_valid_nonempty_by_reward(entries)
    by_improvement = best_valid_nonempty_by_improvement(entries)
    by_error = best_valid_nonempty_by_error(entries)
    primary = by_improvement or by_error or by_reward
    base_error = base.result.energy_error_mha
    best_error = by_error.result.energy_error_mha if by_error is not None else None
    improvement = (base_error - best_error) if base_error is not None and best_error is not None else None
    return {
        "empty_sequence_evaluate_once_per_task": True,
        "empty_sequence_include_in_summary": True,
        "empty_sequence_exclude_from_search_leaderboard": True,
        "empty_sequence_exclude_from_expansion_candidates": True,
        "base_sequence": list(base.sequence),
        "base_parameters": list(base.parameters),
        "base_reward": base.result.reward,
        "base_error_mha": base_error,
        "best_nonempty_sequence": list(primary.sequence) if primary is not None else [],
        "best_nonempty_parameters": list(primary.parameters) if primary is not None else [],
        "best_nonempty_reward": primary.result.reward if primary is not None else None,
        "best_nonempty_error_mha": primary.result.energy_error_mha if primary is not None else None,
        "best_nonempty_delta_error_mha": delta_error_mha(primary) if primary is not None else None,
        "best_nonempty_by_improvement_sequence": list(by_improvement.sequence) if by_improvement is not None else [],
        "best_nonempty_by_improvement_parameters": list(by_improvement.parameters) if by_improvement is not None else [],
        "best_nonempty_by_improvement_reward": by_improvement.result.reward if by_improvement is not None else None,
        "best_nonempty_by_improvement_error_mha": by_improvement.result.energy_error_mha if by_improvement is not None else None,
        "best_nonempty_by_improvement_delta_error_mha": delta_error_mha(by_improvement) if by_improvement is not None else None,
        "best_nonempty_by_reward_sequence": list(by_reward.sequence) if by_reward is not None else [],
        "best_nonempty_by_reward_parameters": list(by_reward.parameters) if by_reward is not None else [],
        "best_nonempty_by_reward_reward": by_reward.result.reward if by_reward is not None else None,
        "best_nonempty_by_reward_error_mha": by_reward.result.energy_error_mha if by_reward is not None else None,
        "best_nonempty_error_sequence": list(by_error.sequence) if by_error is not None else [],
        "best_nonempty_error_parameters": list(by_error.parameters) if by_error is not None else [],
        "best_nonempty_error_reward": by_error.result.reward if by_error is not None else None,
        "best_nonempty_by_error_mha": best_error,
        "improvement_over_base_mha": improvement,
    }


def rank_candidates(
    operator_ids: list[str] | set[str],
    candidate_limit: int | None,
    *,
    ranking: CandidateRanking,
    ranking_angle_grid: Sequence[float],
    cache: SearchCache,
) -> CandidateRankingResult:
    if ranking not in {"lexicographic", "single_step"}:
        raise ValueError("candidate ranking must be 'lexicographic' or 'single_step'")

    pool_candidates = sorted(operator_ids)
    before = cache.evaluations
    ranked_entries: list[SearchEntry] = []

    if ranking == "lexicographic":
        selected = _limit(pool_candidates, candidate_limit)
    else:
        best_by_operator: list[tuple[str, SearchEntry | None]] = []
        for operator_id in pool_candidates:
            op_entries = [cache.entry((operator_id,), (float(angle),)) for angle in ranking_angle_grid]
            valid_entries = [entry for entry in op_entries if entry.result.valid]
            ranked_entries.extend(valid_entries)
            best_by_operator.append((operator_id, min(valid_entries, key=improvement_sort_key) if valid_entries else None))

        def ranking_sort_key(item: tuple[str, SearchEntry | None]) -> tuple[int, int, float, float, int, tuple[str, ...], tuple[float, ...], str]:
            operator_id, entry = item
            if entry is None:
                return (1, 1, 0.0, 0.0, 1, (operator_id,), (), operator_id)
            improvement_key = improvement_sort_key(entry)
            return (*improvement_key, operator_id)

        selected = [operator_id for operator_id, _ in sorted(best_by_operator, key=ranking_sort_key)]
        selected = _limit(selected, candidate_limit)

    metadata = {
        "candidate_ranking": ranking,
        "pool_candidate_count": len(pool_candidates),
        "candidate_limit": candidate_limit,
        "candidate_count": len(selected),
        "selected_candidates": list(selected),
        "ranking_angle_grid": [float(angle) for angle in ranking_angle_grid] if ranking == "single_step" else [],
        "ranking_evaluations": cache.evaluations - before,
    }
    return CandidateRankingResult(candidates=selected, entries=ranked_entries, metadata=metadata)


def refine_entries(
    entries: Iterable[SearchEntry],
    cache: SearchCache,
    *,
    max_candidates: int,
    deltas: Sequence[float] = DEFAULT_REFINEMENT_DELTAS,
    passes: int = 1,
) -> RefinementResult:
    selected = select_top_valid_nonempty(entries, max_candidates)
    before = cache.evaluations
    refined: list[SearchEntry] = []
    normalized_deltas = _normalized_refinement_deltas(deltas)

    if max_candidates <= 0 or passes <= 0 or not normalized_deltas:
        best = selected[0] if selected else None
        return RefinementResult(
            entries=selected,
            best_entry=best,
            evaluations=0,
            metadata={
                "refinement_enabled": False,
                "refinement_candidates": max_candidates,
                "refinement_selected": len(selected),
                "refinement_deltas": list(normalized_deltas),
                "refinement_passes": passes,
                "refinement_evaluations": 0,
            },
        )

    for entry in selected:
        current = entry
        for _ in range(passes):
            improved_this_pass = False
            for param_idx in range(len(current.parameters)):
                best_for_coordinate = current
                for angle in _coordinate_angles(current.parameters[param_idx], normalized_deltas):
                    parameters = list(current.parameters)
                    parameters[param_idx] = angle
                    candidate = cache.entry(current.sequence, tuple(parameters))
                    if candidate.result.valid and improvement_sort_key(candidate) < improvement_sort_key(best_for_coordinate):
                        best_for_coordinate = candidate
                if best_for_coordinate != current:
                    current = best_for_coordinate
                    improved_this_pass = True
            if not improved_this_pass:
                break
        refined.append(current)

    refined = select_top_valid_nonempty([*selected, *refined], max_candidates)
    best = refined[0] if refined else None
    evaluations = cache.evaluations - before
    return RefinementResult(
        entries=refined,
        best_entry=best,
        evaluations=evaluations,
        metadata={
            "refinement_enabled": True,
            "refinement_candidates": max_candidates,
            "refinement_selected": len(selected),
            "refinement_deltas": list(normalized_deltas),
            "refinement_passes": passes,
            "refinement_evaluations": evaluations,
            "best_refined_sequence": list(best.sequence) if best is not None else [],
            "best_refined_parameters": list(best.parameters) if best is not None else [],
            "best_refined_reward": best.result.reward if best is not None else None,
        },
    )


def _limit(candidates: list[str], candidate_limit: int | None) -> list[str]:
    if candidate_limit is None:
        return candidates
    return candidates[:candidate_limit]


def _coordinate_angles(angle: float, deltas: Sequence[float]) -> list[float]:
    values: list[float] = []
    for delta in deltas:
        for sign in (-1.0, 1.0):
            values.append(_clip_angle(angle + sign * delta))
    return _unique_preserving_order(values)


def _normalized_refinement_deltas(deltas: Sequence[float]) -> tuple[float, ...]:
    return tuple(_unique_preserving_order([abs(float(delta)) for delta in deltas if abs(float(delta)) > 0.0]))


def _clip_angle(angle: float) -> float:
    return min(ANGLE_MAX, max(ANGLE_MIN, float(angle)))


def _unique_preserving_order(values: Iterable[float]) -> list[float]:
    seen: set[float] = set()
    unique: list[float] = []
    for value in values:
        rounded = round(float(value), 12)
        if rounded in seen:
            continue
        seen.add(rounded)
        unique.append(float(rounded))
    return unique
