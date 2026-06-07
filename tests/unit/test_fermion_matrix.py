from vqc_molecule_gym.chemistry.cudaqx_driver import build_molecular_data
from vqc_molecule_gym.chemistry.fermion_matrix import exact_ground_energy, fermionic_hamiltonian_matrix


def test_h2_fermion_matrix_matches_cudaqx_casci() -> None:
    data = build_molecular_data(
        [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)],
        basis="STO-3G",
        charge=0,
        multiplicity=1,
        active_orbitals=2,
        active_electrons=2,
    )
    matrix = fermionic_hamiltonian_matrix(data.h1_spin, data.h2_spin, data.nuclear_repulsion)
    exact = exact_ground_energy(matrix, 4, 2)
    assert data.nelec == (1, 1)
    assert abs(exact - data.casci_energy) < 1e-9
