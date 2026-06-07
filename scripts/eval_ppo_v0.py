#!/usr/bin/env python3
"""Evaluate a trained PPO agent and compare against baselines.

Usage
-----
    # Compare PPO checkpoint vs random / greedy / beam
    uv run python scripts/eval_ppo_v0.py --checkpoint runs/ppo_v0_*/checkpoint_*.pt

    # Run all baselines for comparison
    uv run python scripts/eval_ppo_v0.py --checkpoint <path> --baselines random greedy beam
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vqc_molecule_gym.baselines.beam_search_agent import BeamSearchAgent
from vqc_molecule_gym.baselines.greedy_agent import GreedyAgent
from vqc_molecule_gym.baselines.random_agent import RandomAgent
from vqc_molecule_gym.benchmarks import load_tasks
from vqc_molecule_gym.curricula import canonical_benchmark_id
from vqc_molecule_gym.evaluators.direct_energy import DirectEnergyEvaluator
from vqc_molecule_gym.operators.operator_pool import build_operator_pool
from vqc_molecule_gym.rl import PPOConfig, QChemPPOPolicy
from vqc_molecule_gym.rl.observation import build_observation, obs_dim
from vqc_molecule_gym.schemas.action import ActionSpec

# ═══════════════════════════════════════════════════════════════════════════
#  PPO agent evaluator
# ═══════════════════════════════════════════════════════════════════════════


class PPOEvaluator:
    """Greedy evaluation of a trained PPO policy on a task.

    At each step the policy picks the action with the highest logit (argmax).
    """

    def __init__(
        self,
        policy: QChemPPOPolicy,
        config: PPOConfig,
        device: torch.device,
    ) -> None:
        self.policy = policy
        self.config = config
        self.device = device
        self.policy.eval()

    def act(
        self,
        task: Any,
        operator_ids: list[str],
    ) -> ActionSpec:
        """Run the policy greedily on one task, returning the final action."""
        evaluator = DirectEnergyEvaluator()
        param_bins = self.config.parameter_bins
        max_ops = self.config.max_operators
        op_ids = tuple(sorted(operator_ids))
        num_actions = 1 + len(op_ids) * len(param_bins)

        # Base evaluation
        shots = self.config.shots
        base_result = evaluator.evaluate(
            task,
            ActionSpec(operator_sequence=[], shots=shots),
        )
        current_reward = float(base_result.reward)

        seq: list[str] = []
        param_bin_idxs: list[int] = []
        step = 0
        done = False

        while not done and step <= max_ops:
            obs = build_observation(
                task=task,
                step_index=step,
                operator_sequence=tuple(seq),
                parameter_bins=tuple(param_bin_idxs),
                operator_ids=op_ids,
                parameter_bin_values=param_bins,
                max_operators=max_ops,
                base_result=base_result,
                current_result=None,
                current_reward=current_reward,
                previous_reward=current_reward,
            )

            # Build action mask
            mask = np.ones(num_actions, dtype=np.float32)
            if self.config.disable_stop_at_step_0 and step == 0:
                mask[0] = 0.0
            if step >= max_ops:
                mask[1:] = 0.0
                mask[0] = 1.0
            if not self.config.allow_repeated_operators:
                used = set(seq)
                for idx, oid in enumerate(op_ids):
                    if oid in used:
                        start = 1 + idx * len(param_bins)
                        mask[start:start + len(param_bins)] = 0.0
            if mask.sum() == 0:
                mask[0] = 1.0

            # Policy forward
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            out = self.policy(obs_t, action=None, action_mask=mask_t)
            logits = out["logits"]
            logits = logits.masked_fill(mask_t == 0, float("-inf"))
            action = int(logits.argmax(dim=-1).item())

            if action == 0:
                done = True
            else:
                flat = action - 1
                op_idx = flat // len(param_bins)
                bin_idx = flat % len(param_bins)
                seq.append(op_ids[op_idx])
                param_bin_idxs.append(bin_idx)
                step += 1

                # Re-evaluate for next obs
                result = evaluator.evaluate(
                    task,
                    ActionSpec(
                        operator_sequence=list(seq),
                        parameters=[param_bins[bin_idx]],
                        shots=shots,
                    ),
                )
                current_reward = float(result.reward)

                if step >= max_ops:
                    done = True

        return ActionSpec(
            operator_sequence=seq,
            parameters=[param_bins[b] for b in param_bin_idxs],
            shots=shots,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation runner
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_agent(
    agent_name: str,
    agent,
    evaluator: DirectEnergyEvaluator,
    tasks: list[Any],
    operator_ids_by_task: dict[str, list[str]],
    n_episodes: int = 1,
) -> dict[str, Any]:
    """Run an agent on a list of tasks and return aggregated metrics.

    For search-based agents (greedy, beam), each task is evaluated once and
    the result is re-used across episodes (deterministic).  Random has one
    independent episode per task.
    """
    rewards: list[float] = []
    chem_accs: list[bool] = []
    valid_actions: list[bool] = []
    lengths: list[int] = []
    errors_mha: list[float] = []

    for task in tasks:
        op_ids = operator_ids_by_task.get(task.task_id, [])
        if not op_ids:
            continue

        for episode in range(n_episodes):
            if agent_name == "random":
                # Re-sample for each episode
                action = agent.act(task, op_ids)
            else:
                # Deterministic: cache first action and reuse
                if episode == 0:
                    action = agent.act(task, op_ids)
                # Reuse cached action

            result = evaluator.evaluate_payload(task, action.model_dump())
            rewards.append(float(result.reward))
            chem_accs.append(bool(result.chemical_accuracy))
            valid_actions.append(bool(result.valid))
            lengths.append(len(action.operator_sequence))
            if result.energy_error_mha is not None:
                errors_mha.append(float(result.energy_error_mha))

    return {
        "agent": agent_name,
        "n_episodes": len(rewards),
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "std_reward": float(np.std(rewards)) if rewards else float("nan"),
        "median_reward": float(np.median(rewards)) if rewards else float("nan"),
        "min_reward": float(np.min(rewards)) if rewards else float("nan"),
        "max_reward": float(np.max(rewards)) if rewards else float("nan"),
        "chem_accuracy_rate": float(np.mean(chem_accs)) if chem_accs else 0.0,
        "valid_action_rate": float(np.mean(valid_actions)) if valid_actions else 0.0,
        "mean_seq_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_error_mha": float(np.mean(errors_mha)) if errors_mha else float("nan"),
    }


def build_baseline_agents(
    evaluator: DirectEnergyEvaluator,
    config: PPOConfig,
    seed: int = 42,
) -> dict[str, Any]:
    """Create baseline agent instances."""
    agents: dict[str, Any] = {}

    # Random
    agents["random"] = RandomAgent(random.Random(seed))

    # Greedy
    agents["greedy"] = GreedyAgent(
        evaluator,
        max_operators=config.max_operators,
        angle_grid=config.parameter_bins,
        candidate_ranking="single_step",
        refine_angles=False,
    )

    # Beam search
    agents["beam"] = BeamSearchAgent(
        evaluator,
        beam_width=2,
        max_operators=config.max_operators,
        angle_grid=config.parameter_bins,
        candidate_ranking="single_step",
        refine_angles=False,
    )

    return agents


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO vs baselines")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PPO checkpoint .pt file")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["h2_tiny_v0", "lih_bond_scan_v0", "h4_small_v0", "n2_bond_scan_v0"])
    parser.add_argument("--baselines", nargs="+",
                        default=["random", "greedy", "beam"],
                        choices=["random", "greedy", "beam"])
    parser.add_argument("--n-episodes", type=int, default=1,
                        help="Episodes per task (random uses more; search-based use 1)")
    parser.add_argument("--random-episodes", type=int, default=10,
                        help="Episodes per task for random agent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="JSONL output path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", {})

    # Reconstruct PPOConfig from saved dict
    config = PPOConfig(
        max_operators=ckpt_config.get("max_operators", 4),
        parameter_bins=tuple(ckpt_config.get("parameter_bins", [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3])),
        allow_repeated_operators=ckpt_config.get("allow_repeated_operators", True),
        disable_stop_at_step_0=ckpt_config.get("disable_stop_at_step_0", True),
        hidden_sizes=tuple(ckpt_config.get("hidden_sizes", [256, 256])),
        activation=ckpt_config.get("activation", "tanh"),
    )

    # Build policy
    evaluator = DirectEnergyEvaluator()

    # We need to know action dim — load a sample task to get operator pool
    sample_bid = canonical_benchmark_id(args.benchmarks[0])
    sample_tasks = load_tasks(sample_bid)
    sample_task = sample_tasks[0]
    sample_pool = build_operator_pool(
        sample_task.operator_pool_id,
        num_qubits=sample_task.active_space.qubits,
        num_electrons=sample_task.active_space.electrons,
    )
    sample_num_ops = len(sample_pool.ids)
    action_dim = 1 + sample_num_ops * len(config.parameter_bins)

    policy = QChemPPOPolicy(
        obs_dim=obs_dim(config.max_operators),
        action_dim=action_dim,
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
    ).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    print(f"Loaded checkpoint from step {ckpt.get('step', '?')}")

    ppo_agent = PPOEvaluator(policy, config, device)

    # Build baseline agents
    baseline_agents = build_baseline_agents(evaluator, config, seed=args.seed)

    # Run on each benchmark
    results: list[dict[str, Any]] = []

    for bid in args.benchmarks:
        canonical = canonical_benchmark_id(bid)
        tasks = load_tasks(canonical)
        benchmark_ids = [t.task_id for t in tasks]
        print(f"\n{'=' * 60}")
        print(f"Benchmark: {bid} ({len(tasks)} tasks)")
        print(f"{'=' * 60}")

        # Collect operator IDs per task
        op_ids_by_task: dict[str, list[str]] = {}
        for task in tasks:
            pool = build_operator_pool(
                task.operator_pool_id,
                num_qubits=task.active_space.qubits,
                num_electrons=task.active_space.electrons,
            )
            op_ids_by_task[task.task_id] = sorted(pool.ids)

        # PPO
        print(f"  PPO (greedy decode)...", end=" ", flush=True)
        ppo_result = evaluate_agent("ppo", ppo_agent, evaluator, tasks, op_ids_by_task, 1)
        ppo_result["benchmark"] = bid
        ppo_result["n_tasks"] = len(tasks)
        results.append(ppo_result)
        print(f"reward={ppo_result['mean_reward']:.4f}, "
              f"chem_acc={ppo_result['chem_accuracy_rate']:.2%}, "
              f"valid={ppo_result['valid_action_rate']:.2%}")

        # Baselines
        for baseline_name in args.baselines:
            agent = baseline_agents[baseline_name]
            n_eps = args.random_episodes if baseline_name == "random" else args.n_episodes
            print(f"  {baseline_name}...", end=" ", flush=True)
            bl_result = evaluate_agent(baseline_name, agent, evaluator, tasks, op_ids_by_task, n_eps)
            bl_result["benchmark"] = bid
            bl_result["n_tasks"] = len(tasks)
            results.append(bl_result)
            print(f"reward={bl_result['mean_reward']:.4f}, "
                  f"chem_acc={bl_result['chem_accuracy_rate']:.2%}, "
                  f"valid={bl_result['valid_action_rate']:.2%}")

    # Summary table
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"{'Benchmark':<25} {'Agent':<12} {'Reward':>8} {'Chem%':>8} {'Valid%':>8} {'Length':>8}")
    print("-" * 69)
    for r in sorted(results, key=lambda x: (x["benchmark"], x["agent"])):
        print(f"{r['benchmark']:<25} {r['agent']:<12} "
              f"{r['mean_reward']:>8.4f} "
              f"{r['chem_accuracy_rate']:>7.0%} "
              f"{r['valid_action_rate']:>7.0%} "
              f"{r['mean_seq_length']:>8.2f}")

    # Save results
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
