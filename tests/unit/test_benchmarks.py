from vqc_molecule_gym.chemistry.benchmarks import SUPPORTED_BENCHMARK_IDS, generate_benchmark
from vqc_molecule_gym.schemas.task import TaskSpec


def test_supported_benchmark_ids_include_compact_mvp_scans() -> None:
    assert SUPPORTED_BENCHMARK_IDS == (
        "h2_tiny",
        "h4_small",
        "n2_bond_scan_v0",
        "lih_bond_scan_v0",
        "c2h6_torsion_scan_v0",
        "h2o_angle_scan_v0",
        "h2o_dimer_distance_scan_v0",
    )


def test_compact_mvp_benchmark_counts_and_metadata() -> None:
    expected = {
        "n2_bond_scan_v0": (9, "N2", "diatomic_bond_scan", "n_n_distance_angstrom", 6, 6),
        "lih_bond_scan_v0": (8, "LiH", "diatomic_bond_scan", "li_h_distance_angstrom", 4, 4),
        "c2h6_torsion_scan_v0": (7, "C2H6", "torsion_scan", "h_c_c_h_dihedral_degrees", 2, 2),
        "h2o_angle_scan_v0": (7, "H2O", "bond_angle_scan", "h_o_h_angle_degrees", 4, 4),
        "h2o_dimer_distance_scan_v0": (
            7,
            "(H2O)2",
            "intermolecular_distance_scan",
            "oxygen_oxygen_distance_angstrom",
            4,
            4,
        ),
    }
    for benchmark_id, (count, molecule, geometry_kind, scan_variable, electrons, orbitals) in expected.items():
        tasks = generate_benchmark(benchmark_id)
        assert len(tasks) == count
        assert len({task.task_id for task in tasks}) == count
        for task in tasks:
            assert isinstance(TaskSpec.model_validate(task.model_dump()), TaskSpec)
            assert task.benchmark_id == benchmark_id
            assert task.molecule == molecule
            assert task.geometry.kind == geometry_kind
            assert task.geometry.scan_variable == scan_variable
            assert task.geometry.scan_value is not None
            assert task.active_space.electrons == electrons
            assert task.active_space.orbitals == orbitals
            assert task.active_space.spin_orbitals == 2 * orbitals
            assert task.active_space.qubits == 2 * orbitals
            assert task.reference.method == "exact_diagonalization"


def test_compact_mvp_scan_values() -> None:
    expected_values = {
        "n2_bond_scan_v0": [0.80, 0.95, 1.10, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00],
        "lih_bond_scan_v0": [1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 2.40, 3.00],
        "c2h6_torsion_scan_v0": [0, 30, 60, 90, 120, 150, 180],
        "h2o_angle_scan_v0": [80, 90, 100, 104.5, 110, 120, 130],
        "h2o_dimer_distance_scan_v0": [2.40, 2.60, 2.80, 3.00, 3.20, 3.50, 4.00],
    }
    for benchmark_id, values in expected_values.items():
        assert [task.geometry.scan_value for task in generate_benchmark(benchmark_id)] == [float(value) for value in values]


def test_geometry_atom_counts() -> None:
    expected_counts = {
        "n2_bond_scan_v0": 2,
        "lih_bond_scan_v0": 2,
        "c2h6_torsion_scan_v0": 8,
        "h2o_angle_scan_v0": 3,
        "h2o_dimer_distance_scan_v0": 6,
    }
    for benchmark_id, atom_count in expected_counts.items():
        assert all(len(task.geometry.atoms) == atom_count for task in generate_benchmark(benchmark_id))
