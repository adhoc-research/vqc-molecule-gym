from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from vqc_molecule_gym.benchmarks import load_tasks
from vqc_molecule_gym.evaluators.direct_energy import DirectEnergyEvaluator
from vqc_molecule_gym.schemas.action import ActionSpec
from vqc_molecule_gym.schemas.task import TaskSpec
from vqc_molecule_gym.validators.parser import parse_completion


SYSTEM_PROMPT = """You are proposing variational quantum circuit actions for a quantum chemistry benchmark.
Return exactly one JSON object matching this schema and no extra prose:
{"operator_sequence": ["OPERATOR_ID", "..."], "shots": 10000}
Use only operator IDs listed in the task. An empty operator_sequence is valid.
Advanced agents may optionally add explicit per-operator angles as "parameters", e.g.
{"operator_sequence": ["OPERATOR_ID"], "parameters": [0.05], "shots": 10000}
Parameters must be in [-0.5, 0.5] radians and match operator_sequence length.
"""


@dataclass(frozen=True)
class QChemEnvironmentConfig:
    benchmark_id: str = "h4_small"
    evaluator_name: str = "direct_energy"
    reward_version: str = "reward_v1"
    max_turns: int = 1
    benchmark_root: Path = Path("benchmarks")


def load_environment(config: QChemEnvironmentConfig | None = None, **kwargs: Any):
    """Load the QChem benchmark as a Verifiers v1 environment.

    Supports both explicit config objects and Verifiers-style keyword loading:

    - ``load_environment(QChemEnvironmentConfig(benchmark_id="h2_tiny"))``
    - ``load_environment(benchmark_id="h2_tiny", max_turns=1)``
    """
    try:
        import verifiers as vf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("verifiers is required for load_environment") from exc

    cfg = _normalize_config(_resolve_config(config, kwargs))
    if cfg.evaluator_name != "direct_energy":
        raise ValueError(f"Unsupported evaluator_name: {cfg.evaluator_name!r}")

    tasks = load_tasks(cfg.benchmark_id, root=cfg.benchmark_root)
    source = [_task_to_row(task) for task in tasks]
    reward_fn = _make_qchem_reward(cfg)

    taskset = vf.Taskset(
        source=source,
        taskset_id=f"vqc_molecule_gym/{cfg.benchmark_id}",
        system_prompt=SYSTEM_PROMPT,
        rewards=[reward_fn],
    )
    harness = vf.Harness(max_turns=cfg.max_turns)
    return vf.Env(taskset, harness)


def _resolve_config(
    config: QChemEnvironmentConfig | None,
    kwargs: Mapping[str, Any],
) -> QChemEnvironmentConfig:
    if config is None:
        return QChemEnvironmentConfig(**dict(kwargs))
    if not isinstance(config, QChemEnvironmentConfig):
        raise TypeError("config must be a QChemEnvironmentConfig or None")
    if not kwargs:
        return config
    return replace(config, **dict(kwargs))


def _normalize_config(config: QChemEnvironmentConfig) -> QChemEnvironmentConfig:
    return replace(config, benchmark_root=Path(config.benchmark_root))


def _task_to_row(task: TaskSpec) -> dict[str, Any]:
    task_payload = task.model_dump(mode="json")
    task_json = task.model_dump_json()
    return {
        "task_id": task.task_id,
        "prompt": [{"role": "user", "content": _task_prompt(task)}],
        "qchem_task": task_payload,
        "qchem_task_json": task_json,
        "metadata": {
            "qchem_task_json": task_json,
            "benchmark_id": task.benchmark_id,
            "reward_version": "reward_v1",
        },
    }


def _task_prompt(task: TaskSpec) -> str:
    schema = ActionSpec.model_json_schema()
    return (
        f"Task ID: {task.task_id}\n"
        f"Benchmark: {task.benchmark_id}\n"
        f"Molecule: {task.molecule}\n"
        f"Basis: {task.basis}\n"
        f"Active space: {task.active_space.electrons} electrons, "
        f"{task.active_space.orbitals} orbitals, {task.active_space.qubits} qubits\n"
        f"Constraints: max_operators={task.constraints.max_operators}, "
        f"max_depth={task.constraints.max_depth}, max_shots={task.constraints.max_shots}\n"
        f"Operator pool: {task.operator_pool_id}\n"
        "Return only a JSON action. The JSON schema is:\n"
        f"{schema}"
    )


def _make_qchem_reward(config: QChemEnvironmentConfig):
    import verifiers as vf

    evaluator = DirectEnergyEvaluator()

    @vf.reward(weight=1.0)
    async def qchem_reward(task: Mapping[str, Any], state: dict[str, Any]) -> float:
        completion = _completion_text(state.get("completion"))
        state["qchem_completion_text"] = completion
        action_payload, parse_error = parse_completion(completion)
        if parse_error is not None or action_payload is None:
            state["qchem_parse_error"] = parse_error or "invalid_json"
            state["qchem_action"] = None
            return -1.0

        state["qchem_action"] = action_payload
        try:
            task_spec = _task_spec_from_vf_task(task)
            result = evaluator.evaluate_payload(task_spec, action_payload)
        except Exception as exc:  # pragma: no cover - evaluator usually returns invalid results
            state["qchem_eval_error"] = f"{type(exc).__name__}: {exc}"
            return -1.0

        state["qchem_eval"] = result.model_dump(mode="json")
        state["qchem_eval_errors"] = [error.model_dump(mode="json") for error in result.errors]
        state["qchem_reward_version"] = config.reward_version
        return float(result.reward)

    return qchem_reward


def _task_spec_from_vf_task(task: Mapping[str, Any]) -> TaskSpec:
    task_json = task.get("qchem_task_json")
    if not isinstance(task_json, str):
        metadata = task.get("metadata")
        if isinstance(metadata, Mapping):
            task_json = metadata.get("qchem_task_json")
    if isinstance(task_json, str):
        return TaskSpec.model_validate_json(task_json)

    payload = task.get("qchem_task")
    if payload is None:
        metadata = task.get("metadata")
        if isinstance(metadata, Mapping):
            payload = metadata.get("qchem_task")
    if not isinstance(payload, Mapping):
        raise ValueError("Verifiers task is missing qchem_task metadata")
    return TaskSpec.model_validate(payload)


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for message in reversed(completion):
            if not isinstance(message, Mapping):
                continue
            if message.get("role") != "assistant":
                continue
            return _message_content_text(message.get("content"))
    return "" if completion is None else str(completion)


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return "" if content is None else str(content)
