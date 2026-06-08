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

## Post-training a Nemotron model

The environment is exposed to the [Verifiers](https://github.com/PrimeIntellect-ai/verifiers)
RL stack through `vqc_molecule_gym.envs.verifiers_adapter:load_environment`, which builds a
`vf.Env` whose single reward is the shaped environment reward (`reward_v1`: validity, energy
error, circuit depth, shot cost). Post-training a Nemotron policy means having the model emit
one JSON action per task (see [Action format](#action-format)) and optimizing it against that
reward with GRPO. The action JSON is parsed and scored by the same real-backend evaluator used
by the baselines, so the policy is trained on true chemical-accuracy signal, not a proxy.

### 1. Install the RL stack

The base `verifiers` wheel (pinned `>=0.1.14`) ships the environment adapter and `vf-eval`, but
the trainer and inference server live in the optional `verifiers-rl` package (`vf-rl`,
`vf-train`, and `vf-vllm` raise an explicit "install verifiers-rl" error until it is present):

```bash
uv add verifiers-rl vllm
```

`prime-rl` (Prime Intellect's GRPO trainer) is an equivalent training backend and ships the same
`prime-rl` entry point.

### 2. Build the benchmark and references

Rollouts need generated tasks and precomputed exact references for whichever benchmark you train
on. For the smallest curriculum:

```bash
uv run python scripts/generate_tasks.py --benchmark h2_tiny
uv run python scripts/precompute_references.py --benchmark h2_tiny
```

Any benchmark from [Supported generated benchmarks](#supported-generated-benchmarks) works the
same way (`lih_bond_scan_v0`, `h4_small`, ...).

### 3. Serve the Nemotron policy

Start an OpenAI-compatible endpoint for the checkpoint you are post-training (any `nvidia/Nemotron-*`
model on Hugging Face). Either of the following works; see `--help` for GPU/parallelism flags:

```bash
MODEL=nvidia/Nemotron-<variant>
uv run vf-vllm --model "$MODEL" --port 8000     # verifiers-rl inference server
# or, plain vLLM:
# uv run vllm serve "$MODEL" --port 8000
```

### 4. Smoke-test rollouts and reward

Before spending GPU hours on GRPO, confirm the policy returns parseable actions and the
evaluator scores them end to end. The env-id is the importable module that exposes
`load_environment`:

```bash
export OPENAI_API_KEY=local   # vLLM ignores the value but a key var must be set
uv run vf-eval vqc_molecule_gym.envs.verifiers_adapter \
  -a '{"benchmark_id": "h2_tiny", "max_turns": 1}' \
  -m "$MODEL" -b http://localhost:8000/v1 -k OPENAI_API_KEY \
  -n 5 -r 4 -s
```

`-a` passes environment args (forwarded as keyword arguments to `load_environment`; the adapter
accepts `benchmark_id`, `max_turns`, `reward_version`, and `benchmark_root`), `-m`/`-b`/`-k` point
at the served model, and `-n`/`-r` set examples and rollouts per example. Finite per-rollout
rewards that vary across actions, with no `qchem_parse_error`/`qchem_eval_error` in the saved
state, mean the loop is wired correctly (the reward is `-1.0` only for unparseable or invalid
actions; valid actions score on a smooth scale that is `+1` at zero energy error and asymptotes
to `-1` for large errors).

### 5. Run GRPO post-training

Launch the trainer against the same env-id and the Nemotron checkpoint:

```bash
uv run vf-rl <config.toml>     # or: uv run prime-rl <config.toml>
```

In the trainer config set the environment id to `vqc_molecule_gym.envs.verifiers_adapter` with
`env_args = { benchmark_id = "h2_tiny" }`, the policy/model to your Nemotron checkpoint, and keep
the reward as the environment default. The full GRPO config schema (batch size, learning rate,
vLLM colocation, etc.) is owned by `verifiers-rl`/`prime-rl`; see `vf-rl --help` and the upstream
docs for the exact keys.

### NVIDIA NeMo RL backend

To post-train with NVIDIA's native stack (NeMo RL / NeMo-Gym) instead of `verifiers-rl`, point the
rollouts at a NeMo-Gym server and select the matching client with
`--api-client-type nemorl_chat_completions` (e.g. on `vf-eval`). The environment, action schema,
and reward are unchanged — only the inference/training backend differs.
