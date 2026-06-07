from vqc_molecule_gym.chemistry.benchmarks import generate_benchmark
from vqc_molecule_gym.validators.action_validator import validate_action


def test_validate_action_rejects_parameter_length_mismatch() -> None:
    task = generate_benchmark("h2_tiny")[0]
    action, errors = validate_action(
        {"operator_sequence": ["op_a", "op_b"], "parameters": [0.1], "shots": 10000},
        task,
        {"op_a", "op_b"},
        estimated_depth=0,
    )

    assert action is not None
    assert [error.type for error in errors] == ["parameter_length_mismatch"]


def test_validate_action_rejects_parameter_out_of_range() -> None:
    task = generate_benchmark("h2_tiny")[0]
    _, errors = validate_action(
        {"operator_sequence": ["op_a"], "parameters": [0.6], "shots": 10000},
        task,
        {"op_a"},
        estimated_depth=0,
    )

    assert [error.type for error in errors] == ["parameter_out_of_range"]
