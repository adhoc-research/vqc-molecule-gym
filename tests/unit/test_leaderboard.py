import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from vqc_molecule_gym.reporting.leaderboard import (
    base_comparisons,
    benchmark_pareto_frontiers,
    load_records,
    render_report,
    select_task_winners,
    summarize_runs,
    task_pareto_frontiers,
)


def _record(run: str, episode: int, task: str, *, agent: str = "random", valid: bool = True, error: float | None = 1.0, depth: int | None = 1, reward: float = 0.5, ops: int = 1, metadata: dict[str, object] | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "valid": valid,
        "reward": reward,
        "energy_error_mha": error,
        "chemical_accuracy": bool(error is not None and error <= 1.6),
        "action_hash": f"sha256:{run}{episode:06d}",
    }
    if depth is not None:
        result["circuit_metrics"] = {"depth": depth, "num_operators": ops, "gate_count": depth + 1, "two_qubit_gate_count": 0}
    if metadata is not None:
        result["metadata"] = metadata
    return {
        "run_id": run,
        "episode_id": f"ep_{episode:06d}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "benchmark_id": "bench",
        "task_id": task,
        "prompt_hash": None,
        "completion_raw": "{}",
        "action": {"operator_sequence": [], "shots": 10000},
        "result": result,
        "software_versions": {"vqc_molecule_gym": "test"},
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_load_records_expands_globs(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "a.jsonl", [_record("r1", 1, "t1")])
    _write_jsonl(tmp_path / "b.jsonl", [_record("r2", 1, "t1")])

    records = load_records([str(tmp_path / "*.jsonl")])

    assert [r.run_id for r in records] == ["r1", "r2"]


def test_load_records_errors_for_empty_match(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_records([str(tmp_path / "none*.jsonl")])


def test_summarize_runs_handles_invalid_and_missing_metrics(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    _write_jsonl(
        path,
        [
            _record("r1", 1, "t1", valid=True, error=2.0, depth=3, reward=0.2),
            _record("r1", 2, "t1", valid=False, error=None, depth=None, reward=0.0),
        ],
    )

    summary = summarize_runs(load_records([str(path)]))[0]

    assert summary.episodes == 2
    assert summary.valid == 1
    assert summary.invalid == 1
    assert summary.best_error_mha == 2.0
    assert summary.best_reward == 0.2


def test_task_winner_tie_breaks_by_error_depth_ops_reward(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    _write_jsonl(
        path,
        [
            _record("r1", 1, "t1", error=1.0, depth=5, reward=0.9, ops=2),
            _record("r1", 2, "t1", error=1.0, depth=3, reward=0.1, ops=2),
            _record("r1", 3, "t1", error=0.9, depth=9, reward=0.1, ops=5),
            _record("r1", 4, "t1", error=0.1, depth=0, reward=1.0, ops=0),
        ],
    )

    winner = select_task_winners(load_records([str(path)]))[0]

    assert winner.episode_id == "ep_000003"
    assert winner.energy_error_mha == 0.9


def test_pareto_frontiers_per_task_and_benchmark(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    _write_jsonl(
        path,
        [
            _record("r1", 1, "t1", error=1.0, depth=10),
            _record("r1", 2, "t1", error=2.0, depth=9),
            _record("r1", 3, "t1", error=2.0, depth=11),  # dominated by ep2
            _record("r1", 4, "t2", error=0.5, depth=20),
            _record("r1", 5, "t1", error=0.1, depth=0, ops=0),  # empty/base excluded from Pareto
        ],
    )
    records = load_records([str(path)])

    per_task = task_pareto_frontiers(records)
    assert [p.episode_id for p in per_task[("bench", "t1")]] == ["ep_000001", "ep_000002"]

    per_benchmark = benchmark_pareto_frontiers(records)
    assert [p.episode_id for p in per_benchmark["bench"]] == ["ep_000004", "ep_000001", "ep_000002"]


def test_base_comparison_uses_search_metadata_and_report_section(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    _write_jsonl(
        path,
        [
            _record(
                "r1",
                1,
                "t1",
                error=28.92,
                depth=8,
                reward=0.18,
                ops=1,
                metadata={
                    "baseline_search": {
                        "base_error_mha": 46.17,
                        "best_nonempty_by_error_mha": 28.92,
                        "best_nonempty_error_reward": 0.18,
                        "improvement_over_base_mha": 17.25,
                    }
                },
            )
        ],
    )
    records = load_records([str(path)])

    comparison = base_comparisons(records)[0]
    report = render_report(records, top_k=5, pareto_scope="both")

    assert comparison.base_error_mha == 46.17
    assert comparison.best_nonempty_error_mha == 28.92
    assert comparison.improvement_over_base_mha == pytest.approx(17.25)
    assert "## Base vs best non-empty" in report
    assert "46.170" in report
    assert "28.920" in report


def test_render_report_is_compact_markdown(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    _write_jsonl(path, [_record("r1", 1, "t1", error=1.0, depth=2)])

    report = render_report(load_records([str(path)]), top_k=5, pareto_scope="both")

    assert "# QChem VQC Leaderboard" in report
    assert "## Per-task Pareto frontiers" in report
    assert "operator_sequence" not in report
    assert "sha256:" not in report
