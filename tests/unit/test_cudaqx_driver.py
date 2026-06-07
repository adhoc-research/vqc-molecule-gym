import pytest

from vqc_molecule_gym.chemistry.cudaqx_driver import build_molecular_data


def test_cudaqx_driver_maps_singlet_multiplicity_to_spin_zero() -> None:
    data = build_molecular_data(
        [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)],
        basis="STO-3G",
        charge=0,
        multiplicity=1,
        active_orbitals=2,
        active_electrons=2,
    )
    assert data.nelec == (1, 1)
    assert data.n_spin_orbitals == 4


def test_cudaqx_driver_rejects_incompatible_spin_parity_before_http_call() -> None:
    with pytest.raises(ValueError, match="CUDA-QX/PySCF spin=1"):
        build_molecular_data(
            [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)],
            basis="STO-3G",
            charge=0,
            multiplicity=2,
            active_orbitals=2,
            active_electrons=2,
        )
