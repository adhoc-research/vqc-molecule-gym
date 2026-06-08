# vqc-molecule-gym

> **Project page**: [adhoc-research.github.io/vqc-molecule-gym](https://adhoc-research.github.io/vqc-molecule-gym/)

VQC Molecule Gym is an RL environment for molecular VQC circuit proposals.
An agent submits structured JSON actions, the environment validates them, evaluates
them with real chemistry/quantum simulation dependencies, computes reward and
metrics, and logs eval-ready trajectories.

## Quick Start

```bash
uv sync --extra dev
uv run python scripts/generate_tasks.py --benchmark h2_tiny
uv run python scripts/generate_tasks.py --benchmark lih_bond_scan_v0
uv run python scripts/precompute_references.py --benchmark h2_tiny
uv run python scripts/evaluate_action.py --task-id h2_r0.74 --action '{"operator_sequence": [], "shots": 10000}'
uv run python scripts/evaluate_action.py --task-id h2_r0.74 --action '{"operator_sequence": ["OPERATOR_ID"], "parameters": [0.05], "shots": 10000}'
uv run python scripts/run_baseline.py --benchmark h4_small --agent random --episodes 14
uv run python scripts/run_baseline.py --benchmark h2_tiny --agent greedy --episodes 3
uv run python scripts/run_baseline.py --benchmark h2_tiny --agent beam_search --episodes 3 --angle-grid=-0.5,-0.25,-0.1,-0.05,0.05,0.1,0.25,0.5
uv run python scripts/make_leaderboard.py --runs 'runs/*.jsonl' --output reports/leaderboard.md
uv run pytest
```

The MVP is real-backend only: CUDA-Q and CUDA-QX must be installed and importable. PySCF remains an installed dependency because CUDA-QX's local molecule backend uses it internally.

## Action format

Simple agents submit an operator sequence and shots:

```json
{"operator_sequence": ["OPERATOR_ID", "..."], "shots": 10000}
```

Advanced agents may optionally provide one variational angle per operator:

```json
{"operator_sequence": ["OPERATOR_ID"], "parameters": [0.05], "shots": 10000}
```

Parameters are radians, must be finite, must match `operator_sequence` length when supplied, and are limited to `[-0.5, 0.5]`. If parameters are omitted, legacy fixed-angle circuit construction is used for non-empty sequences.

Greedy and beam-search baselines use this advanced parameter format by default with the deterministic angle grid `[-0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5]`. The latest parameterized sweep report is `reports/leaderboard_parameterized_search_20260604.md`.

## Supported generated benchmarks

Core/debug benchmarks:

- `h2_tiny` — H2 bond-length scan.
- `h4_small` — linear H4 bond-length scan, CAS(4e,4o); this is the paper-aligned MVP H4 scan.

Compact MVP `*_v0` scans:

- `n2_bond_scan_v0` — N2 diatomic bond scan over `[0.80, 0.95, 1.10, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00]` Å.
- `lih_bond_scan_v0` — LiH diatomic bond scan over `[1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 2.40, 3.00]` Å.
- `c2h6_torsion_scan_v0` — ethane torsion scan over `[0, 30, 60, 90, 120, 150, 180]` degrees.
- `h2o_angle_scan_v0` — water H-O-H angle scan over `[80, 90, 100, 104.5, 110, 120, 130]` degrees with fixed O-H distance 0.958 Å.
- `h2o_dimer_distance_scan_v0` — water-dimer O...O distance scan over `[2.40, 2.60, 2.80, 3.00, 3.20, 3.50, 4.00]` Å.

The `*_v0` scans are compact MVP grids for environment validation, dashboards, and baseline sweeps. They are intentionally not paper-final grids. Larger molecules use reduced active spaces so exact diagonalization can remain the primary core-reference path where feasible; approximate/precomputed reference tiers should not be mixed into the official core leaderboard.
