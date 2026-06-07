from __future__ import annotations

import math

from vqc_molecule_gym.schemas.task import ActiveSpace, Constraints, Geometry, ReferenceEnergy, TaskSpec

DEFAULT_CONSTRAINTS = Constraints()

SUPPORTED_BENCHMARK_IDS: tuple[str, ...] = (
    "h2_tiny",
    "h4_small",
    "n2_bond_scan_v0",
    "lih_bond_scan_v0",
    "c2h6_torsion_scan_v0",
    "h2o_angle_scan_v0",
    "h2o_dimer_distance_scan_v0",
)


def h2_atoms(bond_length: float) -> list[tuple[str, float, float, float]]:
    return diatomic_atoms("H", "H", bond_length)


# Backwards-compatible alias for the existing H4 linear-chain benchmark.
def h4_atoms(spacing: float) -> list[tuple[str, float, float, float]]:
    return [("H", 0.0, 0.0, i * spacing) for i in range(4)]


def diatomic_atoms(
    atom_a: str,
    atom_b: str,
    distance: float,
) -> list[tuple[str, float, float, float]]:
    """Place a diatomic molecule on the z-axis."""
    return [(atom_a, 0.0, 0.0, 0.0), (atom_b, 0.0, 0.0, distance)]


def h2o_angle_atoms(
    angle_degrees: float,
    *,
    oh_distance: float = 0.958,
    oxygen: tuple[float, float, float] = (0.0, 0.0, 0.0),
    bisector_sign: float = 1.0,
) -> list[tuple[str, float, float, float]]:
    """Build a rigid H2O geometry in the x-z plane for an H-O-H angle."""
    ox, oy, oz = oxygen
    half_angle = math.radians(angle_degrees) / 2.0
    x = oh_distance * math.sin(half_angle)
    z = bisector_sign * oh_distance * math.cos(half_angle)
    return [
        ("O", ox, oy, oz),
        ("H", ox + x, oy, oz + z),
        ("H", ox - x, oy, oz + z),
    ]


def ethane_torsion_atoms(
    dihedral_degrees: float,
    *,
    cc_distance: float = 1.54,
    ch_distance: float = 1.09,
) -> list[tuple[str, float, float, float]]:
    """Build an approximate ethane geometry with one methyl group rotated.

    The C-C bond is on the z-axis. Hydrogens use a tetrahedral-like CH3 layout;
    the right methyl group is rotated by ``dihedral_degrees`` around the C-C axis.
    """
    radial = ch_distance * math.sqrt(8.0 / 9.0)
    axial = ch_distance / 3.0
    left_c_z = -cc_distance / 2.0
    right_c_z = cc_distance / 2.0

    atoms: list[tuple[str, float, float, float]] = [
        ("C", 0.0, 0.0, left_c_z),
        ("C", 0.0, 0.0, right_c_z),
    ]

    for angle in (0.0, 120.0, 240.0):
        theta = math.radians(angle)
        atoms.append(("H", radial * math.cos(theta), radial * math.sin(theta), left_c_z - axial))

    for angle in (dihedral_degrees, dihedral_degrees + 120.0, dihedral_degrees + 240.0):
        theta = math.radians(angle)
        atoms.append(("H", radial * math.cos(theta), radial * math.sin(theta), right_c_z + axial))

    return atoms


def water_dimer_distance_atoms(
    oxygen_oxygen_distance: float,
    *,
    oh_distance: float = 0.958,
    angle_degrees: float = 104.5,
) -> list[tuple[str, float, float, float]]:
    """Build a simple rigid water dimer for an O...O distance scan.

    This is a compact-MVP hydrogen-bond-like orientation, not a paper-final
    optimized water-dimer structure. The first water accepts along +z; the
    second water is translated by the requested O...O distance and has its
    hydrogens pointing back toward the first monomer.
    """
    monomer_a = h2o_angle_atoms(angle_degrees, oh_distance=oh_distance, bisector_sign=-1.0)
    monomer_b = h2o_angle_atoms(
        angle_degrees,
        oh_distance=oh_distance,
        oxygen=(0.0, 0.0, oxygen_oxygen_distance),
        bisector_sign=-1.0,
    )
    return monomer_a + monomer_b


def generate_benchmark(benchmark_id: str) -> list[TaskSpec]:
    if benchmark_id == "h2_tiny":
        return [
            _task(
                benchmark_id="h2_tiny",
                task_id=f"h2_r{r:.2f}",
                molecule="H2",
                atoms=h2_atoms(r),
                scan_value=r,
                electrons=2,
                orbitals=2,
                operator_pool_id="h2_sto3g_uccsd_pool_v0",
            )
            for r in [0.50, 0.74, 1.00, 1.50, 2.00]
        ]
    if benchmark_id == "h4_small":
        return [
            _task(
                benchmark_id="h4_small",
                task_id=f"h4_linear_r{r:.2f}_sto3g_cas4e4o",
                molecule="H4",
                atoms=h4_atoms(r),
                scan_value=r,
                electrons=4,
                orbitals=4,
                operator_pool_id="h4_sto3g_uccsd_pool_v0",
            )
            for r in [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        ]
    if benchmark_id == "n2_bond_scan_v0":
        return [
            _task(
                benchmark_id=benchmark_id,
                task_id=f"n2_r{r:.2f}_sto3g_cas6e6o",
                molecule="N2",
                atoms=diatomic_atoms("N", "N", r),
                scan_value=r,
                electrons=6,
                orbitals=6,
                operator_pool_id="n2_sto3g_cas6e6o_uccsd_pool_v0",
                geometry_kind="diatomic_bond_scan",
                scan_variable="n_n_distance_angstrom",
            )
            for r in [0.80, 0.95, 1.10, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00]
        ]
    if benchmark_id == "lih_bond_scan_v0":
        return [
            _task(
                benchmark_id=benchmark_id,
                task_id=f"lih_r{r:.2f}_sto3g_cas4e4o",
                molecule="LiH",
                atoms=diatomic_atoms("Li", "H", r),
                scan_value=r,
                electrons=4,
                orbitals=4,
                operator_pool_id="lih_sto3g_cas4e4o_uccsd_pool_v0",
                geometry_kind="diatomic_bond_scan",
                scan_variable="li_h_distance_angstrom",
            )
            for r in [1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 2.40, 3.00]
        ]
    if benchmark_id == "c2h6_torsion_scan_v0":
        return [
            _task(
                benchmark_id=benchmark_id,
                task_id=f"c2h6_phi{int(phi):03d}_sto3g_cas2e2o",
                molecule="C2H6",
                atoms=ethane_torsion_atoms(phi),
                scan_value=float(phi),
                electrons=2,
                orbitals=2,
                operator_pool_id="c2h6_sto3g_cas2e2o_uccsd_pool_v0",
                geometry_kind="torsion_scan",
                scan_variable="h_c_c_h_dihedral_degrees",
            )
            for phi in [0, 30, 60, 90, 120, 150, 180]
        ]
    if benchmark_id == "h2o_angle_scan_v0":
        return [
            _task(
                benchmark_id=benchmark_id,
                task_id=f"h2o_angle{_format_scan_value(angle)}_sto3g_cas4e4o",
                molecule="H2O",
                atoms=h2o_angle_atoms(angle),
                scan_value=float(angle),
                electrons=4,
                orbitals=4,
                operator_pool_id="h2o_sto3g_cas4e4o_uccsd_pool_v0",
                geometry_kind="bond_angle_scan",
                scan_variable="h_o_h_angle_degrees",
            )
            for angle in [80, 90, 100, 104.5, 110, 120, 130]
        ]
    if benchmark_id == "h2o_dimer_distance_scan_v0":
        return [
            _task(
                benchmark_id=benchmark_id,
                task_id=f"h2o_dimer_r{r:.2f}_sto3g_cas4e4o",
                molecule="(H2O)2",
                atoms=water_dimer_distance_atoms(r),
                scan_value=r,
                electrons=4,
                orbitals=4,
                operator_pool_id="h2o_dimer_sto3g_cas4e4o_uccsd_pool_v0",
                geometry_kind="intermolecular_distance_scan",
                scan_variable="oxygen_oxygen_distance_angstrom",
            )
            for r in [2.40, 2.60, 2.80, 3.00, 3.20, 3.50, 4.00]
        ]
    raise ValueError(f"Unknown benchmark: {benchmark_id}")


def _task(
    *,
    benchmark_id: str,
    task_id: str,
    molecule: str,
    atoms: list[tuple[str, float, float, float]],
    scan_value: float,
    electrons: int,
    orbitals: int,
    operator_pool_id: str,
    geometry_kind: str = "linear_chain",
    scan_variable: str = "bond_length_angstrom",
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        benchmark_id=benchmark_id,
        molecule=molecule,
        geometry=Geometry(
            kind=geometry_kind,
            atoms=atoms,
            scan_variable=scan_variable,
            scan_value=scan_value,
        ),
        basis="STO-3G",
        charge=0,
        multiplicity=1,
        active_space=ActiveSpace(
            electrons=electrons,
            orbitals=orbitals,
            spin_orbitals=2 * orbitals,
            qubits=2 * orbitals,
        ),
        mapping="jordan_wigner",
        reference=ReferenceEnergy(method="exact_diagonalization", energy_hartree=0.0),
        constraints=DEFAULT_CONSTRAINTS,
        operator_pool_id=operator_pool_id,
    )


def _format_scan_value(value: float) -> str:
    return f"{value:g}"
