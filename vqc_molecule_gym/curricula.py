"""Benchmark curriculum tiers for RL training.

Hard benchmarks are useful evaluation tasks, but PPO should start from the easy
or medium curricula where dense improvement signal is more reliable.
"""
from __future__ import annotations

CURRICULUM_TIERS: dict[str, tuple[str, ...]] = {
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

# Public curriculum names keep explicit v0 labels for H2/H4. Existing benchmark
# artifacts currently use these canonical directory IDs.
BENCHMARK_ALIASES: dict[str, str] = {
    "h2_tiny_v0": "h2_tiny",
    "h4_small_v0": "h4_small",
}


def canonical_benchmark_id(benchmark_id: str) -> str:
    return BENCHMARK_ALIASES.get(benchmark_id, benchmark_id)


def curriculum_benchmark_ids(tier: str, *, canonical: bool = True) -> tuple[str, ...]:
    if tier not in CURRICULUM_TIERS:
        raise ValueError(f"Unknown curriculum tier: {tier}")
    benchmark_ids = CURRICULUM_TIERS[tier]
    if canonical:
        return tuple(canonical_benchmark_id(benchmark_id) for benchmark_id in benchmark_ids)
    return benchmark_ids
