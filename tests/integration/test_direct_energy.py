from vqc_molecule_gym.chemistry.benchmarks import generate_benchmark
from vqc_molecule_gym.evaluators.direct_energy import DirectEnergyEvaluator, attach_reference
from vqc_molecule_gym.operators.operator_pool import build_operator_pool


def test_h2_empty_action_direct_energy() -> None:
    task = attach_reference(next(task for task in generate_benchmark("h2_tiny") if task.task_id == "h2_r0.74"))
    result = DirectEnergyEvaluator().evaluate_payload(task, {"operator_sequence": [], "shots": 10000})
    assert result.valid
    assert result.energy_hartree is not None
    assert result.energy_error_mha is not None
    assert result.circuit_metrics is not None


def test_h2_parameterized_action_direct_energy() -> None:
    task = attach_reference(next(task for task in generate_benchmark("h2_tiny") if task.task_id == "h2_r0.74"))
    pool = build_operator_pool(task.operator_pool_id, num_qubits=task.active_space.qubits, num_electrons=task.active_space.electrons)
    operator_id = sorted(pool.ids)[0]

    result = DirectEnergyEvaluator().evaluate_payload(
        task,
        {"operator_sequence": [operator_id], "parameters": [0.05], "shots": 10000},
    )

    assert result.valid
    assert result.metadata["parameters_supplied"] is True
    assert result.metadata["operator_angles_rad"] == [0.05]


def test_invalid_action_rejected_before_simulation() -> None:
    task = generate_benchmark("h2_tiny")[0]
    result = DirectEnergyEvaluator().evaluate_payload(task, {"operator_sequence": ["NOPE"], "shots": 10000})
    assert not result.valid
    assert result.errors[0].type == "unknown_operator"
