import pytest

from vqc_molecule_gym.curricula import CURRICULUM_TIERS, canonical_benchmark_id, curriculum_benchmark_ids


def test_curriculum_tiers_are_separated_for_rl_training() -> None:
    assert CURRICULUM_TIERS == {
        "easy_curriculum": (
            "h2_tiny_v0",
            "lih_bond_scan_v0",
            "c2h6_torsion_scan_v0",
        ),
        "medium_curriculum": (
            "h2o_angle_scan_v0",
            "h2o_dimer_distance_scan_v0",
        ),
        "hard_curriculum": (
            "h4_small_v0",
            "n2_bond_scan_v0",
        ),
    }


def test_curriculum_ids_resolve_to_existing_benchmark_artifacts() -> None:
    assert curriculum_benchmark_ids("easy_curriculum") == (
        "h2_tiny",
        "lih_bond_scan_v0",
        "c2h6_torsion_scan_v0",
    )
    assert curriculum_benchmark_ids("hard_curriculum") == ("h4_small", "n2_bond_scan_v0")
    assert canonical_benchmark_id("h2o_angle_scan_v0") == "h2o_angle_scan_v0"


def test_curriculum_can_return_public_v0_names() -> None:
    assert curriculum_benchmark_ids("easy_curriculum", canonical=False)[0] == "h2_tiny_v0"
    assert curriculum_benchmark_ids("hard_curriculum", canonical=False)[0] == "h4_small_v0"


def test_unknown_curriculum_tier_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown curriculum tier"):
        curriculum_benchmark_ids("not_a_tier")
