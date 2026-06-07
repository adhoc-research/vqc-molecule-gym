import pytest
from pydantic import ValidationError

from vqc_molecule_gym.chemistry.benchmarks import generate_benchmark
from vqc_molecule_gym.schemas.action import ActionSpec


def test_action_schema_is_strict() -> None:
    with pytest.raises(ValidationError):
        ActionSpec.model_validate({"operator_sequence": [], "shots": "10000"})


def test_action_parameters_are_backward_compatible_and_finite() -> None:
    legacy = ActionSpec.model_validate({"operator_sequence": ["A"], "shots": 10000})
    assert legacy.parameters == []

    parameterized = ActionSpec.model_validate({"operator_sequence": ["A"], "parameters": [0.05], "shots": 10000})
    assert parameterized.parameters == [0.05]

    with pytest.raises(ValidationError):
        ActionSpec.model_validate({"operator_sequence": ["A"], "parameters": [float("nan")], "shots": 10000})


def test_task_schema_builds_h2() -> None:
    task = generate_benchmark("h2_tiny")[0]
    assert task.active_space.qubits == 4
    assert task.constraints.max_operators == 8
