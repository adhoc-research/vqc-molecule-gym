from typing import Literal

from pydantic import Field

from vqc_molecule_gym.schemas.base import StrictModel


class Geometry(StrictModel):
    kind: str
    atoms: list[tuple[str, float, float, float]]
    scan_variable: str | None = None
    scan_value: float | None = None


class ActiveSpace(StrictModel):
    electrons: int = Field(gt=0)
    orbitals: int = Field(gt=0)
    spin_orbitals: int = Field(gt=0)
    qubits: int = Field(gt=0)


class ReferenceEnergy(StrictModel):
    method: str
    energy_hartree: float
    chemical_accuracy_mha: float = 1.6
    casci_energy_hartree: float | None = None


class Constraints(StrictModel):
    max_operators: int = Field(default=8, ge=0)
    max_depth: int = Field(default=120, gt=0)
    max_shots: int = Field(default=100000, gt=0)
    chemical_accuracy_mha: float = Field(default=1.6, gt=0)
    allowed_parameter_policy: tuple[Literal["fixed"], ...] = ("fixed",)


class TaskSpec(StrictModel):
    task_id: str
    benchmark_id: str
    molecule: str
    geometry: Geometry
    basis: str
    charge: int = 0
    multiplicity: int = 1
    active_space: ActiveSpace
    mapping: Literal["jordan_wigner"] = "jordan_wigner"
    reference: ReferenceEnergy
    constraints: Constraints
    operator_pool_id: str
