import time
from pathlib import Path

import numpy as np

from vqc_molecule_gym.chemistry.fermion_matrix import exact_ground_energy, fermionic_hamiltonian_matrix
from vqc_molecule_gym.chemistry.cudaqx_driver import build_molecular_data
from vqc_molecule_gym.circuits.cudaq_builder import build_kernel, circuit_metrics, estimate_depth, statevector_from_kernel
from vqc_molecule_gym.operators.operator_pool import build_operator_pool
from vqc_molecule_gym.rewards.reward_functions import REWARD_VERSION, REWARD_WEIGHTS, reward_v1
from vqc_molecule_gym.schemas.action import ActionSpec
from vqc_molecule_gym.schemas.result import EvalError, EvalResult
from vqc_molecule_gym.schemas.task import ReferenceEnergy, TaskSpec
from vqc_molecule_gym.utils.hashing import sha256_json
from vqc_molecule_gym.validators.action_validator import validate_action


class DirectEnergyEvaluator:
    def __init__(self, cache_dir: Path = Path(".cache/vqc_molecule_gym"), cudaq_target: str | None = None) -> None:
        self.cache_dir = cache_dir
        self.cudaq_target = cudaq_target
        self._problem_cache: dict[str, tuple[np.ndarray, float, float]] = {}
        self._pool_cache = {}

    def evaluate_payload(self, task: TaskSpec, action_payload: dict[str, object]) -> EvalResult:
        started = time.perf_counter()
        pool = self._load_pool(task)
        estimated_depth = estimate_depth(
            [str(op) for op in action_payload.get("operator_sequence", [])],
            pool,
        ) if isinstance(action_payload.get("operator_sequence", []), list) else 0
        action, errors = validate_action(action_payload, task, pool.ids, estimated_depth)
        if errors or action is None:
            return _invalid_result(task, action_payload, errors, started)
        return self.evaluate(task, action)

    def evaluate(self, task: TaskSpec, action: ActionSpec) -> EvalResult:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        try:
            t0 = time.perf_counter()
            hamiltonian, reference, casci_energy = self._load_problem(task)
            timings["hamiltonian_load_sec"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            pool = self._load_pool(task)
            metrics = circuit_metrics(task, pool, action.operator_sequence, action.parameters)
            if metrics.depth > task.constraints.max_depth:
                errors = [
                    EvalError(
                        type="estimated_depth_too_high",
                        message=f"Estimated depth {metrics.depth} exceeds {task.constraints.max_depth}.",
                    )
                ]
                return _invalid_result(task, action.model_dump(), errors, started)
            kernel = build_kernel(task, pool, action.operator_sequence, action.parameters)
            timings["circuit_build_sec"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            state = statevector_from_kernel(kernel, task.active_space.qubits)
            energy = float(np.vdot(state, hamiltonian @ state).real)
            timings["simulation_sec"] = time.perf_counter() - t0
        except Exception as exc:
            return _invalid_result(
                task,
                action.model_dump(),
                [EvalError(type="simulation_error", message=f"{type(exc).__name__}: {exc}")],
                started,
            )

        energy_error = abs(energy - reference)
        energy_error_mha = energy_error * 1000.0
        reward, components = reward_v1(
            valid=True,
            energy_error_mha=energy_error_mha,
            metrics=metrics,
            shots=action.shots,
            constraints=task.constraints,
        )
        timings["total_sec"] = time.perf_counter() - started
        return EvalResult(
            valid=True,
            task_id=task.task_id,
            action_hash=sha256_json(action.model_dump()),
            energy_hartree=energy,
            reference_energy_hartree=reference,
            energy_error_hartree=energy_error,
            energy_error_mha=energy_error_mha,
            chemical_accuracy=energy_error_mha <= task.constraints.chemical_accuracy_mha,
            reward=reward,
            reward_components=components,
            circuit_metrics=metrics,
            sampling={"shots": action.shots, "seed": 42},
            timing=timings,
            metadata={
                "reward_version": REWARD_VERSION,
                "reward_weights": REWARD_WEIGHTS,
                "casci_energy_hartree": casci_energy,
                "parameters_supplied": bool(action.parameters),
                "operator_angles_rad": action.parameters or [0.1] * len(action.operator_sequence),
            },
        )

    def _load_pool(self, task: TaskSpec):
        key = (task.operator_pool_id, task.active_space.qubits, task.active_space.electrons)
        if key not in self._pool_cache:
            self._pool_cache[key] = build_operator_pool(
                task.operator_pool_id,
                num_qubits=task.active_space.qubits,
                num_electrons=task.active_space.electrons,
            )
        return self._pool_cache[key]

    def _load_problem(self, task: TaskSpec) -> tuple[np.ndarray, float, float]:
        if task.task_id not in self._problem_cache:
            data = build_molecular_data(
                task.geometry.atoms,
                basis=task.basis,
                charge=task.charge,
                multiplicity=task.multiplicity,
                active_orbitals=task.active_space.orbitals,
                active_electrons=task.active_space.electrons,
            )
            hamiltonian = fermionic_hamiltonian_matrix(data.h1_spin, data.h2_spin, data.nuclear_repulsion)
            reference = exact_ground_energy(
                hamiltonian,
                task.active_space.qubits,
                task.active_space.electrons,
            )
            self._problem_cache[task.task_id] = (hamiltonian, reference, data.casci_energy)
        return self._problem_cache[task.task_id]


def attach_reference(task: TaskSpec) -> TaskSpec:
    data = build_molecular_data(
        task.geometry.atoms,
        basis=task.basis,
        charge=task.charge,
        multiplicity=task.multiplicity,
        active_orbitals=task.active_space.orbitals,
        active_electrons=task.active_space.electrons,
    )
    hamiltonian = fermionic_hamiltonian_matrix(data.h1_spin, data.h2_spin, data.nuclear_repulsion)
    exact = exact_ground_energy(hamiltonian, task.active_space.qubits, task.active_space.electrons)
    return task.model_copy(
        update={
            "reference": ReferenceEnergy(
                method="exact_diagonalization",
                energy_hartree=exact,
                chemical_accuracy_mha=task.constraints.chemical_accuracy_mha,
                casci_energy_hartree=data.casci_energy,
            )
        }
    )


def _invalid_result(
    task: TaskSpec,
    action_payload: dict[str, object],
    errors: list[EvalError],
    started: float,
) -> EvalResult:
    return EvalResult(
        valid=False,
        task_id=task.task_id,
        action_hash=sha256_json(action_payload),
        reward=-1.0,
        reward_components={"validity": -1.0},
        timing={"total_sec": time.perf_counter() - started},
        errors=errors,
    )
