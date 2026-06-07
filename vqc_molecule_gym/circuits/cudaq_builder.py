import numpy as np
import cudaq

from vqc_molecule_gym.circuits.hartree_fock import basis_labels_little_endian, hartree_fock_occupied_qubits
from vqc_molecule_gym.operators.operator_pool import OperatorPool
from vqc_molecule_gym.schemas.result import CircuitMetrics
from vqc_molecule_gym.schemas.task import TaskSpec

FIXED_ANGLE_RAD = 0.1


def estimate_depth(operator_sequence: list[str], pool: OperatorPool) -> int:
    entries = pool.by_id()
    return sum(max(entries[operator_id].term_count, 1) for operator_id in operator_sequence if operator_id in entries)


def build_kernel(
    task: TaskSpec,
    pool: OperatorPool,
    operator_sequence: list[str],
    parameters: list[float] | None = None,
):
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(task.active_space.qubits)
    for qubit_idx in hartree_fock_occupied_qubits(task.active_space.electrons):
        kernel.x(qubits[qubit_idx])

    angles = _operator_angles(operator_sequence, parameters)
    entries = pool.by_id()
    for operator_id, angle in zip(operator_sequence, angles, strict=True):
        op = entries[operator_id].cudaq_operator
        for term in op:
            coeff = complex(term.get_coefficient()).real
            pauli_word = _full_pauli_word(term.get_pauli_word(), term.degrees, task.active_space.qubits)
            kernel.exp_pauli(float(coeff * angle), qubits, pauli_word)
    return kernel


def circuit_metrics(
    task: TaskSpec,
    pool: OperatorPool,
    operator_sequence: list[str],
    parameters: list[float] | None = None,
) -> CircuitMetrics:
    entries = pool.by_id()
    term_count = sum(entries[operator_id].term_count for operator_id in operator_sequence)
    return CircuitMetrics(
        num_qubits=task.active_space.qubits,
        num_operators=len(operator_sequence),
        depth=term_count,
        gate_count=task.active_space.electrons + term_count,
        two_qubit_gate_count=0,
        parameter_count=len(operator_sequence),
    )


def _operator_angles(operator_sequence: list[str], parameters: list[float] | None) -> list[float]:
    if parameters:
        return parameters
    return [FIXED_ANGLE_RAD] * len(operator_sequence)


def statevector_from_kernel(kernel, n_qubits: int) -> np.ndarray:
    state = cudaq.get_state(kernel)
    amplitudes = state.amplitudes(basis_labels_little_endian(n_qubits))
    return np.asarray(amplitudes, dtype=np.complex128)


def _full_pauli_word(short_word: str, degrees: list[int], n_qubits: int) -> str:
    full = ["I"] * n_qubits
    for char, idx in zip(short_word, degrees, strict=True):
        full[idx] = char
    return "".join(full)
