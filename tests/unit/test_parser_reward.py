from vqc_molecule_gym.chemistry.benchmarks import generate_benchmark
from vqc_molecule_gym.rewards.reward_functions import reward_v1
from vqc_molecule_gym.schemas.result import CircuitMetrics
from vqc_molecule_gym.validators.parser import parse_completion


def _metrics(*, num_operators: int = 1, depth: int = 10) -> CircuitMetrics:
    return CircuitMetrics(
        num_qubits=4,
        num_operators=num_operators,
        depth=depth,
        gate_count=12,
        two_qubit_gate_count=0,
        parameter_count=num_operators,
    )


def test_parser_accepts_fenced_json() -> None:
    payload, error = parse_completion('text\n```json\n{"operator_sequence": [], "shots": 10000}\n```')
    assert error is None
    assert payload == {"operator_sequence": [], "shots": 10000}


def test_reward_v1_valid_result() -> None:
    task = generate_benchmark("h2_tiny")[0]
    reward, components = reward_v1(
        valid=True,
        energy_error_mha=1.0,
        metrics=_metrics(),
        shots=10000,
        constraints=task.constraints,
    )
    assert reward > 0
    assert set(components) == {"energy_error", "chemical_accuracy", "depth", "shots", "compactness"}


def test_reward_v1_is_monotonic_as_energy_error_decreases() -> None:
    task = generate_benchmark("h2_tiny")[0]
    kwargs = {"valid": True, "metrics": _metrics(), "shots": 10000, "constraints": task.constraints}

    high_error, _ = reward_v1(energy_error_mha=25.0, **kwargs)
    medium_error, _ = reward_v1(energy_error_mha=10.0, **kwargs)
    low_error, _ = reward_v1(energy_error_mha=2.0, **kwargs)
    near_exact, _ = reward_v1(energy_error_mha=0.1, **kwargs)

    assert high_error < medium_error < low_error < near_exact


def test_reward_v1_chemical_accuracy_threshold_bonus() -> None:
    task = generate_benchmark("h2_tiny")[0]
    threshold = task.constraints.chemical_accuracy_mha
    kwargs = {"valid": True, "metrics": _metrics(), "shots": 10000, "constraints": task.constraints}

    at_threshold, at_components = reward_v1(energy_error_mha=threshold, **kwargs)
    just_above, above_components = reward_v1(energy_error_mha=threshold + 1e-6, **kwargs)

    assert at_components["chemical_accuracy"] > 0.0
    assert above_components["chemical_accuracy"] == 0.0
    assert at_threshold > just_above


def test_reward_v1_invalid_action_is_penalized() -> None:
    task = generate_benchmark("h2_tiny")[0]
    reward, components = reward_v1(
        valid=False,
        energy_error_mha=None,
        metrics=None,
        shots=10000,
        constraints=task.constraints,
    )

    assert reward == -1.0
    assert components == {"validity": -1.0}
