from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cudaq_solvers
import numpy as np


@dataclass(frozen=True)
class MolecularData:
    h1_spatial: np.ndarray
    h2_spatial: np.ndarray
    h1_spin: np.ndarray
    h2_spin: np.ndarray
    energy_offset: float
    hf_energy: float
    reference_energy: float
    casci_energy: float
    n_spatial_orbitals: int
    n_spin_orbitals: int
    n_electrons: int
    nelec: tuple[int, int]
    cudaqx_energies: dict[str, float]

    @property
    def nuclear_repulsion(self) -> float:
        """Compatibility alias for the scalar Hamiltonian offset."""
        return self.energy_offset

    @property
    def fci_energy(self) -> float:
        """Compatibility alias for exact active-space reference energy."""
        return self.reference_energy


class CudaQXMoleculeError(RuntimeError):
    """Raised when CUDA-QX molecule generation fails with project context."""


def build_molecular_data(
    atoms: list[tuple[str, float, float, float]],
    *,
    basis: str,
    charge: int,
    multiplicity: int,
    active_orbitals: int,
    active_electrons: int | None = None,
) -> MolecularData:
    """Build active-space molecular data using CUDA-QX Solvers.

    CUDA-QX's public docs call the third argument ``spin``, but the packaged
    ``cudaq-pyscf`` backend uses PySCF semantics: ``spin = 2S = Nalpha-Nbeta``.
    For conventional multiplicity ``2S+1``, that means ``spin=multiplicity-1``.

    CUDA-QX's ``hpqrs`` is already half-weighted for
    ``0.5 * h[p,q,r,s] a_p† a_q† a_r a_s``. The local dense fermion matrix
    builder applies the 0.5 itself, so we multiply CUDA-QX two-body coefficients
    by 2 before returning them.
    """
    if multiplicity < 1:
        raise ValueError(f"multiplicity must be >= 1, got {multiplicity}")
    if active_orbitals < 1:
        raise ValueError(f"active_orbitals must be >= 1, got {active_orbitals}")

    total_electrons = _electron_count(atoms, charge)
    active_electrons = total_electrons if active_electrons is None else active_electrons
    _validate_active_space(active_electrons, active_orbitals)

    geometry = [(symbol, (float(x), float(y), float(z))) for symbol, x, y, z in atoms]
    cudaqx_spin = multiplicity - 1
    _validate_spin_electron_parity(active_electrons, cudaqx_spin, multiplicity)

    try:
        molecule = cudaq_solvers.create_molecule(
            geometry,
            basis.lower(),
            cudaqx_spin,
            charge,
            nele_cas=active_electrons,
            norb_cas=active_orbitals,
            casci=True,
        )
    except Exception as exc:  # noqa: BLE001 - CUDA-QX wraps useful errors as HTTP 500.
        raise CudaQXMoleculeError(_format_cudaqx_error(exc, geometry, basis, charge, multiplicity, cudaqx_spin, active_electrons, active_orbitals)) from exc

    h1 = np.ascontiguousarray(np.asarray(molecule.hpq, dtype=np.complex128))
    h2 = np.ascontiguousarray(2.0 * np.asarray(molecule.hpqrs, dtype=np.complex128))
    n_spin_orbitals = 2 * active_orbitals
    if h1.shape != (n_spin_orbitals, n_spin_orbitals):
        raise CudaQXMoleculeError(f"CUDA-QX returned hpq shape {h1.shape}; expected {(n_spin_orbitals, n_spin_orbitals)}")
    if h2.shape != (n_spin_orbitals, n_spin_orbitals, n_spin_orbitals, n_spin_orbitals):
        expected = (n_spin_orbitals, n_spin_orbitals, n_spin_orbitals, n_spin_orbitals)
        raise CudaQXMoleculeError(f"CUDA-QX returned hpqrs shape {h2.shape}; expected {expected}")

    energies = {str(k): float(v) for k, v in molecule.energies.items()}
    offset = _energy_offset(energies)
    casci_energy = _reference_energy(energies)
    nelec = _spin_resolved_electrons(active_electrons, cudaqx_spin)

    return MolecularData(
        h1_spatial=np.ascontiguousarray(h1[0::2, 0::2]),
        h2_spatial=np.ascontiguousarray(h2[0::2, 0::2, 0::2, 0::2]),
        h1_spin=h1,
        h2_spin=h2,
        energy_offset=offset,
        hf_energy=float(energies.get("hf_energy", np.nan)),
        reference_energy=casci_energy,
        casci_energy=casci_energy,
        n_spatial_orbitals=int(molecule.n_orbitals),
        n_spin_orbitals=n_spin_orbitals,
        n_electrons=int(molecule.n_electrons),
        nelec=nelec,
        cudaqx_energies=energies,
    )


def _energy_offset(energies: dict[str, float]) -> float:
    if "core_energy" in energies:
        return energies["core_energy"]
    if "nuclear_energy" in energies:
        return energies["nuclear_energy"]
    raise CudaQXMoleculeError(f"CUDA-QX energies missing core/nuclear scalar offset: {energies}")


def _reference_energy(energies: dict[str, float]) -> float:
    for key in ("R-CASCI", "UR-CASCI", "fci_energy", "hf_energy"):
        if key in energies:
            return energies[key]
    raise CudaQXMoleculeError(f"CUDA-QX energies missing reference energy: {energies}")


def _spin_resolved_electrons(n_electrons: int, spin: int) -> tuple[int, int]:
    n_beta = (n_electrons - spin) // 2
    n_alpha = n_electrons - n_beta
    return int(n_alpha), int(n_beta)


def _validate_active_space(active_electrons: int, active_orbitals: int) -> None:
    if active_electrons < 1:
        raise ValueError(f"active_electrons must be >= 1, got {active_electrons}")
    if active_electrons > 2 * active_orbitals:
        raise ValueError(f"active_electrons={active_electrons} exceeds capacity of {active_orbitals} spatial orbitals")


def _validate_spin_electron_parity(active_electrons: int, spin: int, multiplicity: int) -> None:
    if abs(spin) > active_electrons:
        raise ValueError(f"multiplicity={multiplicity} maps to spin={spin}, incompatible with {active_electrons} electrons")
    if (active_electrons - spin) % 2 != 0:
        raise ValueError(
            f"multiplicity={multiplicity} maps to CUDA-QX/PySCF spin={spin}, but electron count {active_electrons} "
            "has incompatible parity. CUDA-QX expects spin=Nalpha-Nbeta, not multiplicity."
        )


def _format_cudaqx_error(
    exc: BaseException,
    geometry: list[tuple[str, tuple[float, float, float]]],
    basis: str,
    charge: int,
    multiplicity: int,
    cudaqx_spin: int,
    active_electrons: int,
    active_orbitals: int,
) -> str:
    hint = ""
    if "HTTP POST Error - status code 500" in str(exc):
        hint = (
            " CUDA-QX molecule generation uses a local cudaq-pyscf HTTP helper; HTTP 500 usually means the "
            "server-side PySCF/CUDA-QX backend rejected the chemistry input. Check basis, charge/spin parity, "
            "and active-space settings. Run `uv run python scripts/debug_cudaqx_molecule.py --cli` for tracebacks."
        )
    return (
        f"CUDA-QX molecule creation failed: {type(exc).__name__}: {exc}.{hint} "
        f"Input={{geometry={geometry}, basis={basis!r}, charge={charge}, multiplicity={multiplicity}, "
        f"cudaqx_spin={cudaqx_spin}, active_electrons={active_electrons}, active_orbitals={active_orbitals}}}"
    )


_PERIODIC_Z: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
}


def _electron_count(atoms: list[tuple[str, float, float, float]], charge: int) -> int:
    try:
        return sum(_PERIODIC_Z[symbol] for symbol, *_ in atoms) - charge
    except KeyError as exc:
        raise ValueError(f"Unsupported element for local electron-count validation: {exc.args[0]}") from exc
