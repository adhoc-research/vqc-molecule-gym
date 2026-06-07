#!/usr/bin/env python3
"""Diagnose CUDA-QX/CUDA-Q Solvers molecule creation failures.

This intentionally exercises both the Python API and the packaged cudaq-pyscf
CLI. The API path often collapses server-side PySCF exceptions into HTTP 500;
the CLI path exposes the real traceback.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _versions() -> None:
    import cudaq
    import cudaq_solvers

    print("Python:", sys.version.split()[0])
    print("cudaq:", getattr(cudaq, "__version__", "unknown"))
    print("cudaq_solvers:", getattr(cudaq_solvers, "__version__", "unknown"))
    print("pyscf:", importlib.metadata.version("pyscf"))


def _api_case(name: str, geometry: list[tuple[str, tuple[float, float, float]]], *, basis: str, spin: int, charge: int, kwargs: dict[str, Any]) -> None:
    import cudaq_solvers

    print(f"\n=== API: {name} ===")
    print({"geometry": geometry, "basis": basis, "spin": spin, "charge": charge, "kwargs": kwargs})
    try:
        mol = cudaq_solvers.create_molecule(geometry, basis, spin, charge, **kwargs)
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print("ERR:", type(exc).__name__, str(exc))
        return
    print("OK:", {"n_electrons": mol.n_electrons, "n_orbitals": mol.n_orbitals, "energies": mol.energies})


def _cli_case(name: str, atoms_xyz: str, *, basis: str, spin: int, charge: int, extra: list[str] | None = None) -> None:
    script = Path(".venv/lib/python3.11/site-packages/cudaq_solvers/bin/cudaq-pyscf")
    if not script.exists():
        print("cudaq-pyscf script not found at", script)
        return
    with tempfile.TemporaryDirectory() as td:
        xyz = Path(td) / "molecule.xyz"
        xyz.write_text(atoms_xyz)
        cmd = [
            sys.executable,
            str(script),
            "--xyz",
            str(xyz),
            "--basis",
            basis,
            "--spin",
            str(spin),
            "--charge",
            str(charge),
            *(extra or []),
        ]
        print(f"\n=== CLI: {name} ===")
        print(" ".join(cmd))
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        print("exit:", result.returncode)
        print("\n".join(result.stdout.splitlines()[:80]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Also run cudaq-pyscf CLI cases to expose server-side tracebacks.")
    args = parser.parse_args()

    _versions()
    h2 = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]
    h4 = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.8)), ("H", (0.0, 0.0, 1.6)), ("H", (0.0, 0.0, 2.4))]

    cases = [
        ("H2 singlet: PySCF spin=0", h2, "sto-3g", 0, 0, {}),
        ("H2 with multiplicity=1 passed as spin: expected HTTP 500", h2, "sto-3g", 1, 0, {}),
        ("H4 singlet: PySCF spin=0", h4, "sto-3g", 0, 0, {}),
        ("H4 with multiplicity=1 passed as spin: expected HTTP 500", h4, "sto-3g", 1, 0, {}),
        ("Bad basis: expected HTTP 500", h2, "not-a-basis", 0, 0, {}),
        ("Half active space: expected HTTP 500", h2, "sto-3g", 0, 0, {"norb_cas": 2}),
        ("Valid active space", h2, "sto-3g", 0, 0, {"nele_cas": 2, "norb_cas": 2, "casci": True}),
    ]
    for name, geometry, basis, spin, charge, kwargs in cases:
        _api_case(name, geometry, basis=basis, spin=spin, charge=charge, kwargs=kwargs)

    if args.cli:
        h2_xyz = "2\nH2\nH 0 0 0\nH 0 0 0.74\n"
        _cli_case("H2 singlet", h2_xyz, basis="sto-3g", spin=0, charge=0)
        _cli_case("H2 multiplicity passed as spin", h2_xyz, basis="sto-3g", spin=1, charge=0)
        _cli_case("Bad basis", h2_xyz, basis="not-a-basis", spin=0, charge=0)
        _cli_case("Half active space", h2_xyz, basis="sto-3g", spin=0, charge=0, extra=["--norb_cas", "2"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
