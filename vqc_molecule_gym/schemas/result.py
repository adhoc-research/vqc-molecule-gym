from typing import Any

from vqc_molecule_gym.schemas.base import StrictModel


class EvalError(StrictModel):
    type: str
    message: str


class CircuitMetrics(StrictModel):
    num_qubits: int
    num_operators: int
    depth: int
    gate_count: int
    two_qubit_gate_count: int
    parameter_count: int = 0


class EvalResult(StrictModel):
    valid: bool
    task_id: str | None = None
    action_hash: str | None = None
    evaluator: str = "direct_energy"
    backend: str = "cudaq"
    energy_hartree: float | None = None
    reference_energy_hartree: float | None = None
    energy_error_hartree: float | None = None
    energy_error_mha: float | None = None
    chemical_accuracy: bool = False
    reward: float
    reward_components: dict[str, float] = {}
    circuit_metrics: CircuitMetrics | None = None
    sampling: dict[str, int] = {}
    timing: dict[str, float] = {}
    errors: list[EvalError] = []
    metadata: dict[str, Any] = {}
