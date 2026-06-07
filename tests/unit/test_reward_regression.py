import pytest

from vqc_molecule_gym.rewards.reward_functions import REWARD_VERSION, REWARD_WEIGHTS, reward_v1
from vqc_molecule_gym.schemas.result import CircuitMetrics
from vqc_molecule_gym.schemas.task import Constraints


CONSTRAINTS = Constraints(max_operators=8, max_depth=120, max_shots=100000, chemical_accuracy_mha=1.6)


def metrics(*, num_operators: int, depth: int) -> CircuitMetrics:
    return CircuitMetrics(
        num_qubits=4,
        num_operators=num_operators,
        depth=depth,
        gate_count=depth,
        two_qubit_gate_count=0,
        parameter_count=num_operators,
    )


def score(*, error_mha: float, num_operators: int, depth: int) -> float:
    reward, _ = reward_v1(
        valid=True,
        energy_error_mha=error_mha,
        metrics=metrics(num_operators=num_operators, depth=depth),
        shots=10000,
        constraints=CONSTRAINTS,
    )
    return reward


def test_reward_v1_1_0_version_and_weights_are_frozen() -> None:
    assert REWARD_VERSION == "reward_v1.1.0"
    assert REWARD_WEIGHTS == {
        "energy_error": 0.90,
        "chemical_accuracy": 0.08,
        "depth": 0.01,
        "shots": 0.005,
        "compactness": 0.005,
    }


@pytest.mark.parametrize(
    ("error_mha", "expected_unweighted_energy_score"),
    [
        (0.0, 1.0),
        (1.6, 0.0),
        (4.8, -0.5),
    ],
)
def test_reward_v1_1_0_energy_score_anchor_points(error_mha: float, expected_unweighted_energy_score: float) -> None:
    _, components = reward_v1(
        valid=True,
        energy_error_mha=error_mha,
        metrics=metrics(num_operators=1, depth=10),
        shots=10000,
        constraints=CONSTRAINTS,
    )

    assert components["energy_error"] == pytest.approx(0.90 * expected_unweighted_energy_score)


def test_reward_ordering_matches_rl_signal_regression() -> None:
    chemically_accurate_circuit = score(error_mha=1.0, num_operators=2, depth=24)
    improved_nonempty_circuit = score(error_mha=10.0, num_operators=2, depth=24)
    base_state = score(error_mha=50.0, num_operators=0, depth=0)
    worse_than_base_circuit = score(error_mha=100.0, num_operators=2, depth=24)
    invalid_action, _ = reward_v1(
        valid=False,
        energy_error_mha=None,
        metrics=None,
        shots=10000,
        constraints=CONSTRAINTS,
    )

    assert chemically_accurate_circuit > improved_nonempty_circuit
    assert improved_nonempty_circuit > base_state
    assert base_state > worse_than_base_circuit
    assert worse_than_base_circuit > invalid_action
