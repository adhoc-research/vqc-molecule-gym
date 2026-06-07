from vqc_molecule_gym.baselines.search_helpers import (
    CandidateRanking,
    DEFAULT_REFINEMENT_DELTAS,
    SearchCache,
    SearchEntry,
    base_vs_nonempty_metadata,
    improvement_sort_key,
    rank_candidates,
    refine_entries,
)
from vqc_molecule_gym.baselines.types import SearchEvaluator
from vqc_molecule_gym.schemas.action import ActionSpec
from vqc_molecule_gym.schemas.result import EvalResult
from vqc_molecule_gym.schemas.task import TaskSpec

DEFAULT_ANGLE_GRID: tuple[float, ...] = (0.1,)


class GreedyAgent:
    """Incrementally append the (operator, angle) with the best reward improvement."""

    name = "greedy"

    def __init__(
        self,
        evaluator: SearchEvaluator,
        *,
        max_operators: int | None = None,
        candidate_limit: int | None = None,
        angle_grid: list[float] | tuple[float, ...] | None = None,
        candidate_ranking: CandidateRanking = "lexicographic",
        ranking_angle_grid: list[float] | tuple[float, ...] | None = None,
        refine_angles: bool = False,
        refinement_candidates: int = 3,
        refinement_deltas: list[float] | tuple[float, ...] | None = None,
        refinement_passes: int = 1,
    ) -> None:
        self.evaluator = evaluator
        self.max_operators = max_operators
        self.candidate_limit = candidate_limit
        self.angle_grid = tuple(angle_grid) if angle_grid is not None else DEFAULT_ANGLE_GRID
        self.candidate_ranking = candidate_ranking
        self.ranking_angle_grid = tuple(ranking_angle_grid) if ranking_angle_grid is not None else self.angle_grid
        self.refine_angles = refine_angles
        self.refinement_candidates = refinement_candidates
        self.refinement_deltas = tuple(refinement_deltas) if refinement_deltas is not None else DEFAULT_REFINEMENT_DELTAS
        self.refinement_passes = refinement_passes
        self.last_search_evaluations = 0
        self.last_best_result: EvalResult | None = None
        self.last_search_metadata: dict[str, object] = {}

    def act(self, task: TaskSpec, operator_ids: list[str] | set[str]) -> ActionSpec:
        shots = min(10000, task.constraints.max_shots)
        max_depth = task.constraints.max_operators
        if self.max_operators is not None:
            max_depth = min(max_depth, self.max_operators)

        cache = SearchCache(self.evaluator, task, shots=shots)
        base = cache.entry((), ())
        current: SearchEntry | None = None
        expansion_seed = base
        ranking = rank_candidates(
            operator_ids,
            self.candidate_limit,
            ranking=self.candidate_ranking,
            ranking_angle_grid=self.ranking_angle_grid,
            cache=cache,
        )
        candidates = ranking.candidates
        nonempty_entries: list[SearchEntry] = list(ranking.entries)

        for _ in range(max_depth):
            if not candidates or not self.angle_grid:
                break
            expanded = []
            for operator_id in candidates:
                for angle in self.angle_grid:
                    sequence = expansion_seed.sequence + (operator_id,)
                    parameters = expansion_seed.parameters + (float(angle),)
                    entry = cache.entry(sequence, parameters)
                    expanded.append(entry)
                    if entry.sequence and entry.result.valid:
                        nonempty_entries.append(entry)
            valid_expanded = [entry for entry in expanded if entry.result.valid]
            if not valid_expanded:
                break
            best_entry = min(valid_expanded, key=improvement_sort_key)
            if current is not None and improvement_sort_key(best_entry) >= improvement_sort_key(current):
                break
            current = best_entry
            expansion_seed = best_entry

        best_pre_refinement = current or base
        refinement_metadata: dict[str, object] = {
            "refinement_enabled": False,
            "refinement_evaluations": 0,
        }
        if self.refine_angles:
            refinement = refine_entries(
                [*nonempty_entries, best_pre_refinement],
                cache,
                max_candidates=self.refinement_candidates,
                deltas=self.refinement_deltas,
                passes=self.refinement_passes,
            )
            refinement_metadata = refinement.metadata
            if refinement.best_entry is not None and (current is None or improvement_sort_key(refinement.best_entry) < improvement_sort_key(current)):
                current = refinement.best_entry

        final = current or base
        base_metadata = base_vs_nonempty_metadata(base, [*nonempty_entries, final])
        self.last_search_evaluations = cache.evaluations
        self.last_best_result = final.result
        self.last_search_metadata = {
            "agent": self.name,
            "evaluations": cache.evaluations,
            "max_operators": max_depth,
            "candidate_limit": self.candidate_limit,
            "candidate_count": len(candidates),
            "candidate_ranking": self.candidate_ranking,
            "angle_grid": list(self.angle_grid),
            "ranking_angle_grid": list(self.ranking_angle_grid),
            "best_pre_refinement_sequence": list(best_pre_refinement.sequence),
            "best_pre_refinement_parameters": list(best_pre_refinement.parameters),
            "best_pre_refinement_reward": best_pre_refinement.result.reward,
            "best_sequence": list(final.sequence),
            "best_parameters": list(final.parameters),
            "best_reward": final.result.reward,
            **ranking.metadata,
            **refinement_metadata,
            **base_metadata,
        }
        return ActionSpec(operator_sequence=list(final.sequence), parameters=list(final.parameters), shots=shots)


# Backward-compatible aliases for older tests/imports that used this module's helpers.
def _candidate_ids(operator_ids: list[str] | set[str], candidate_limit: int | None) -> list[str]:
    candidates = sorted(operator_ids)
    if candidate_limit is not None:
        return candidates[:candidate_limit]
    return candidates


def _sort_key(
    sequence: tuple[str, ...],
    parameters: tuple[float, ...],
    result: EvalResult,
) -> tuple[float, int, tuple[str, ...], tuple[float, ...]]:
    from vqc_molecule_gym.baselines.search_helpers import sort_key

    return sort_key(sequence, parameters, result)
