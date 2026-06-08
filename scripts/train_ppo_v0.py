#!/usr/bin/env python3
"""Train a PPO agent on the QChem molecule gym.

Uses pure PyTorch + Gymnasium (no PufferLib vectorisation for v0).
The training loop implements a standard PPO clipped surrogate objective with
Generalised Advantage Estimation (GAE). Action masking is supported.

Usage
-----
    ./venv/bin/python scripts/train_ppo_v0.py \\
        --curriculum easy_curriculum \\
        --total-timesteps 200_000 \\
        --eval-frequency 50_000
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from vqc_molecule_gym.benchmarks import load_tasks
from vqc_molecule_gym.curricula import canonical_benchmark_id, curriculum_benchmark_ids
from vqc_molecule_gym.evaluators.direct_energy import DirectEnergyEvaluator
from vqc_molecule_gym.operators.operator_pool import build_operator_pool
from vqc_molecule_gym.rl import PPOConfig, QChemPPOEnv, QChemPPOPolicy
from vqc_molecule_gym.rl.observation import obs_dim
from vqc_molecule_gym.rl.observation import build_observation, obs_dim
from vqc_molecule_gym.schemas.action import ActionSpec
from vqc_molecule_gym.utils.io import write_jsonl
from vqc_molecule_gym.utils.seeds import seed_everything


# ═══════════════════════════════════════════════════════════════════════════
#  Argument parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO v0 for QChem molecule gym")

    # Environment
    parser.add_argument("--curriculum", default="easy_curriculum")
    parser.add_argument("--max-operators", type=int, default=4)
    parser.add_argument("--parameter-bins", nargs="+", type=float,
                        default=[-0.3, -0.2, -0.1, 0.1, 0.2, 0.3])
    parser.add_argument("--allow-repeated-operators", default=True,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--disable-stop-at-step-0", default=True,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--shots", type=int, default=10_000)

    # Policy
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--activation", default="tanh", choices=["tanh", "relu"])

    # Training
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    # Eval / logging
    parser.add_argument("--eval-frequency", type=int, default=50_000)
    parser.add_argument("--eval-benchmarks", nargs="+",
                        default=["h2_tiny_v0", "lih_bond_scan_v0",
                                 "h4_small_v0", "n2_bond_scan_v0"])
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--stop-bonus-weight", type=float, default=0.1)

    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PPOConfig:
    return PPOConfig(
        curriculum=args.curriculum,
        eval_benchmarks=tuple(args.eval_benchmarks),
        max_operators=args.max_operators,
        parameter_bins=tuple(args.parameter_bins),
        allow_repeated_operators=args.allow_repeated_operators,
        disable_stop_at_step_0=args.disable_stop_at_step_0,
        shots=args.shots,
        hidden_sizes=tuple(args.hidden_sizes),
        activation=args.activation,
        learning_rate=args.learning_rate,
        total_timesteps=args.total_timesteps,
        num_envs=1,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
        eval_frequency=args.eval_frequency,
        log_dir=args.log_dir,
        seed=args.seed,
        stop_bonus_weight=args.stop_bonus_weight,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Rollout buffer
# ═══════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """Stores one episode of experience for PPO updates."""

    def __init__(self) -> None:
        self.obs: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.action_masks: list[np.ndarray | None] = []

    def clear(self) -> None:
        for lst in [self.obs, self.actions, self.rewards, self.dones,
                     self.log_probs, self.values, self.action_masks]:
            lst.clear()

    def __len__(self) -> int:
        return len(self.obs)

    def compute_gae(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """GAE: returns (advantages, returns)."""
        values_np = np.array(self.values + [last_value])
        rewards_np = np.array(self.rewards)
        dones_np = np.array(self.dones, dtype=np.float32)
        advantages = np.zeros_like(rewards_np, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards_np))):
            delta = (
                rewards_np[t]
                + gamma * values_np[t + 1] * (1.0 - dones_np[t])
                - values_np[t]
            )
            gae = delta + gamma * gae_lambda * (1.0 - dones_np[t]) * gae
            advantages[t] = gae
        returns = advantages + np.array(self.values)
        return advantages, returns


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation (greedy decode on eval benchmarks)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    policy: QChemPPOPolicy,
    config: PPOConfig,
    device: torch.device,
    num_episodes: int = 10,
) -> dict[str, float]:
    """Greedy evaluation (argmax) across eval benchmarks."""
    policy.eval()
    evaluator = DirectEnergyEvaluator()
    all_rewards: list[float] = []
    all_chem_acc: list[bool] = []
    all_lengths: list[int] = []

    # Build a temporary env to get the fixed max action dimension
    import copy
    tmp_config = dataclasses.replace(
        config, curriculum="easy_curriculum", eval_benchmarks=())
    tmp_env = QChemPPOEnv(tmp_config)
    tmp_env.reset()
    max_num_actions = int(tmp_env.action_space.n)
    
    # Collect unique benchmark IDs (training + eval-only)
    benchmark_ids = list(curriculum_benchmark_ids(config.curriculum))
    for bid in config.eval_benchmarks:
        if bid not in benchmark_ids:
            benchmark_ids.append(bid)

    for bid in benchmark_ids:
        try:
            tasks = load_tasks(canonical_benchmark_id(bid),
                               root=Path(config.benchmark_root))
        except Exception:
            continue
        if not tasks:
            continue

        rng = np.random.default_rng(config.seed)
        selected = rng.choice(tasks, min(num_episodes, len(tasks)),
                              replace=False)

        for task in selected:
            pool = build_operator_pool(
                task.operator_pool_id,
                num_qubits=task.active_space.qubits,
                num_electrons=task.active_space.electrons,
            )
            operator_ids = tuple(sorted(pool.ids))
            param_bins = config.parameter_bins
            num_valid_actions = 1 + len(operator_ids) * len(param_bins)
            max_ops = config.max_operators

            # Empty base
            base_result = evaluator.evaluate(
                task, ActionSpec(operator_sequence=[], shots=config.shots))
            current_reward = float(base_result.reward)
            seq: list[str] = []
            params: list[int] = []
            step = 0
            done = False

            while not done and step <= max_ops:
                obs = build_observation(
                    task=task, step_index=step,
                    operator_sequence=tuple(seq),
                    parameter_bins=tuple(params),
                    operator_ids=operator_ids,
                    parameter_bin_values=param_bins,
                    max_operators=max_ops,
                    base_result=base_result,
                    current_result=None,
                    current_reward=current_reward,
                    previous_reward=current_reward,
                )
                # Action mask — must use max_num_actions to match policy
                mask = np.ones(max_num_actions, dtype=np.float32)
                # Block actions beyond this task's valid range
                if num_valid_actions < max_num_actions:
                    mask[num_valid_actions:] = 0.0
                if config.disable_stop_at_step_0 and step == 0:
                    mask[0] = 0.0
                if step >= max_ops:
                    mask[1:] = 0.0
                    mask[0] = 1.0
                if not config.allow_repeated_operators:
                    used = set(seq)
                    for idx, oid in enumerate(operator_ids):
                        if oid in used:
                            s = 1 + idx * len(param_bins)
                            mask[s:s + len(param_bins)] = 0.0

                obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                        device=device).unsqueeze(0)
                mask_t = torch.as_tensor(mask, dtype=torch.float32,
                                         device=device).unsqueeze(0)
                out = policy(obs_t, action=None, action_mask=mask_t)
                logits = out["logits"].masked_fill(mask_t == 0, float("-inf"))
                action = int(logits.argmax(dim=-1).item())

                if action == 0:  # STOP
                    done = True
                    final_r = current_reward + config.stop_bonus_weight * current_reward
                    all_rewards.append(final_r)
                    all_lengths.append(len(seq))
                else:
                    flat = action - 1
                    op_idx = flat // len(param_bins)
                    bin_idx = flat % len(param_bins)
                    seq.append(operator_ids[op_idx])
                    params.append(bin_idx)
                    step += 1

                    result = evaluator.evaluate(
                        task,
                        ActionSpec(operator_sequence=list(seq),
                                   parameters=[param_bins[b] for b in params],
                                   shots=config.shots),
                    )
                    new_r = float(result.reward)
                    current_reward = new_r
                    if step >= max_ops:
                        done = True
                        final_r = current_reward + config.stop_bonus_weight * current_reward
                        all_rewards.append(final_r)
                        all_chem_acc.append(bool(result.chemical_accuracy))
                        all_lengths.append(len(seq))

    policy.train()
    if not all_rewards:
        return {}
    return {
        "eval_mean_reward": float(np.mean(all_rewards)),
        "eval_median_reward": float(np.median(all_rewards)),
        "eval_min_reward": float(np.min(all_rewards)),
        "eval_max_reward": float(np.max(all_rewards)),
        "eval_chem_accuracy_rate": float(np.mean(all_chem_acc)) if all_chem_acc else 0.0,
        "eval_mean_seq_length": float(np.mean(all_lengths)),
        "eval_num_episodes": len(all_rewards),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PPO update
# ═══════════════════════════════════════════════════════════════════════════

def ppo_update(
    policy: QChemPPOPolicy,
    optimizer: optim.Adam,
    buffer: RolloutBuffer,
    advantages: np.ndarray,
    returns: np.ndarray,
    config: PPOConfig,
    device: torch.device,
) -> dict[str, float]:
    """Run one PPO update on the buffer data.

    Returns dict of mean losses.
    """
    # Normalise advantages with robust std clipping
    adv_std = advantages.std()
    if adv_std > 1e-6:
        adv = (advantages - advantages.mean()) / (adv_std + 1e-8)
    else:
        adv = advantages - advantages.mean()  # no scaling if std is tiny

    # Build tensors
    obs_t = torch.as_tensor(np.stack(buffer.obs), dtype=torch.float32,
                            device=device)
    act_t = torch.as_tensor(np.array(buffer.actions), dtype=torch.long,
                            device=device)
    old_logp_t = torch.as_tensor(np.array(buffer.log_probs),
                                 dtype=torch.float32, device=device)
    adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
    ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

    masks_t = None
    if buffer.action_masks[0] is not None:
        masks_t = torch.as_tensor(np.stack(buffer.action_masks),
                                  dtype=torch.float32, device=device)

    dataset_size = len(buffer)
    indices = np.arange(dataset_size)
    losses_policy: list[float] = []
    losses_value: list[float] = []
    losses_entropy: list[float] = []
    kl_vals: list[float] = []

    for epoch in range(config.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, config.batch_size):
            idx = indices[start:start + config.batch_size]

            batch_masks = masks_t[idx] if masks_t is not None else None
            out = policy(obs_t[idx], action=act_t[idx],
                         action_mask=batch_masks)
            new_logp = out["log_prob"]
            entropy = out["entropy"]
            new_value = out["value"]

            ratio = torch.exp(new_logp - old_logp_t[idx])
            surr1 = ratio * adv_t[idx]
            surr2 = torch.clamp(ratio, 1.0 - config.clip_epsilon,
                                1.0 + config.clip_epsilon) * adv_t[idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            v_pred = new_value
            v_unclipped = (v_pred - ret_t[idx]).pow(2)
            v_clipped = old_logp_t[idx] + (v_pred - old_logp_t[idx]).clamp(
                -config.clip_epsilon, config.clip_epsilon)
            v_clipped_loss = (v_clipped - ret_t[idx]).pow(2)
            value_loss = 0.5 * torch.max(v_unclipped, v_clipped_loss).mean()

            # Entropy bonus with minimum threshold to prevent collapse
            entropy_mean = entropy.mean()
            min_entropy = 0.5  # nat minimum; below this we boost entropy coef
            entropy_bonus = entropy_mean
            if entropy_mean < min_entropy:
                entropy_bonus = entropy_mean + (min_entropy - entropy_mean) * 2.0
            entropy_loss = -entropy_bonus

            loss = (policy_loss
                    + config.value_coef * value_loss
                    + config.entropy_coef * entropy_loss)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(),
                                     config.max_grad_norm)
            optimizer.step()

            losses_policy.append(float(policy_loss.item()))
            losses_value.append(float(value_loss.item()))
            losses_entropy.append(float(entropy_mean.item()))
            with torch.no_grad():
                kl_vals.append(float(
                    ((ratio - 1.0) - ratio.log()).mean().item()))

    return {
        "policy_loss": float(np.mean(losses_policy)),
        "value_loss": float(np.mean(losses_value)),
        "entropy": float(np.mean(losses_entropy)),
        "approx_kl": float(np.mean(kl_vals)),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Main training loop
# ═══════════════════════════════════════════════════════════════════════════

def train(config: PPOConfig) -> dict[str, Any]:
    """Run PPO training, return final metrics."""
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Environment setup ─────────────────────────────────────────────────
    env = QChemPPOEnv(config)
    obs, info = env.reset()
    obs_dim_val = obs_dim(config.max_operators)
    action_dim = int(env.action_space.n)
    print(f"Observation dim: {obs_dim_val}")
    print(f"Action dim: {action_dim}")

    # ── Policy / optimiser ────────────────────────────────────────────────
    policy = QChemPPOPolicy(
        obs_dim=obs_dim_val,
        action_dim=action_dim,
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate,
                           eps=1e-5)
    print(f"Parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # ── Logging ───────────────────────────────────────────────────────────
    run_id = f"ppo_v0_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path(config.log_dir) / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "metrics.jsonl"

    with open(log_dir / "config.json", "w") as f:
        d = dataclasses.asdict(config)
        d = {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}
        json.dump(d, f, indent=2)

    # ── Training variables ────────────────────────────────────────────────
    global_step = 0
    episode_idx = 0
    buffer = RolloutBuffer()
    episode_rewards: list[float] = []
    episode_lengths: list[float] = []
    episode_chem_acc: list[bool] = []
    mask_compliance: list[bool] = []
    loss_history: list[dict[str, float]] = []
    eval_history: list[dict[str, Any]] = []

    t_start = time.perf_counter()

    # ── Main loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {config.total_timesteps:,} timesteps...")
    while global_step < config.total_timesteps:
        # ── Forward pass ──────────────────────────────────────────────────
        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                device=device).unsqueeze(0)
        mask = info.get("action_mask")
        mask_t = (torch.as_tensor(mask, dtype=torch.float32,
                                  device=device).unsqueeze(0)
                  if mask is not None else None)

        out = policy(obs_t, action=None, action_mask=mask_t)
        dist = out["dist"]
        value = float(out["value"].item())
        action = int(dist.sample().item())
        log_prob = float(dist.log_prob(
            torch.tensor(action, device=device)).item())

        if mask is not None:
            mask_compliance.append(bool(mask[action] > 0.5))

        # ── Step env ──────────────────────────────────────────────────────
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.obs.append(obs)
        buffer.actions.append(action)
        buffer.rewards.append(reward)
        buffer.dones.append(done)
        buffer.log_probs.append(log_prob)
        buffer.values.append(value)
        buffer.action_masks.append(mask)
        obs = next_obs

        if done:
            global_step += len(buffer)
            episode_rewards.append(float(np.sum(buffer.rewards)))
            episode_lengths.append(len(buffer))
            episode_idx += 1

            # Chemical accuracy from final info
            ca = info.get("chemical_accuracy", False)
            episode_chem_acc.append(bool(ca))

            # Compute GAE
            last_val = 0.0  # bootstrap with 0 for terminal
            adv, ret = buffer.compute_gae(last_val, config.gamma,
                                          config.gae_lambda)

            # PPO update
            losses = ppo_update(policy, optimizer, buffer, adv, ret,
                                config, device)
            loss_history.append(losses)

            # ── Progress ──────────────────────────────────────────────────
            if episode_idx % 5 == 0:
                elapsed = time.perf_counter() - t_start
                sps = global_step / max(elapsed, 1e-6)
                r_mean = float(np.mean(episode_rewards[-50:])) if episode_rewards else 0.0
                l_mean = float(np.mean(episode_lengths[-50:])) if episode_lengths else 0.0
                mc = float(np.mean(mask_compliance[-200:])) if mask_compliance else 0.0
                ent = losses.get("entropy", 0.0)
                print(
                    f"[{global_step:>8,} steps | "
                    f"{episode_idx:>5} eps | "
                    f"{sps:>7.1f} sps | "
                    f"kl={losses['approx_kl']:.4f} | "
                    f"ent={ent:.3f}]  "
                    f"reward={r_mean:+.4f}  "
                    f"len={l_mean:.1f}  "
                    f"comp={mc:.2f}"
                )

            # ── Periodic evaluation ──────────────────────────────────────
            if global_step % config.eval_frequency < len(buffer):
                eval_metrics = evaluate(policy, config, device,
                                        num_episodes=10)
                eval_history.append({"step": global_step, **eval_metrics})
                if eval_metrics:
                    print(f"\n  ── Evaluation @ step {global_step} ──")
                    for k, v in eval_metrics.items():
                        print(f"    {k}: {v:.4f}")
                    print()

                # Save checkpoint
                ckpt_path = log_dir / f"checkpoint_{global_step:08d}.pt"
                torch.save({
                    "step": global_step,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": dataclasses.asdict(config),
                    "action_dim": action_dim,
                    "obs_dim": obs_dim_val,
                    "eval_metrics": eval_metrics,
                }, ckpt_path)
                print(f"  Checkpoint: {ckpt_path}\n")

            # ── Log metrics row ──────────────────────────────────────────
            metrics_row = {
                "step": global_step,
                "episode": episode_idx,
                "mean_reward": float(np.mean(episode_rewards[-100:])),
                "mean_length": float(np.mean(episode_lengths[-100:])),
                "chem_acc_rate": float(np.mean(episode_chem_acc[-100:])),
                "mask_compliance": float(np.mean(mask_compliance[-200:])),
                **losses,
            }
            write_jsonl(metrics_path, [metrics_row])

            # Reset buffer
            buffer.clear()
            obs, info = env.reset()

    # ── Final evaluation ──────────────────────────────────────────────────
    print("\nRunning final evaluation...")
    final_eval = evaluate(policy, config, device, num_episodes=50)
    if final_eval:
        print("\n  ── Final Evaluation ──")
        for k, v in final_eval.items():
            print(f"    {k}: {v:.4f}")

    # Save final model
    final_path = log_dir / "model_final.pt"
    torch.save({
        "step": global_step,
        "model_state_dict": policy.state_dict(),
        "config": dataclasses.asdict(config),
        "eval_metrics": final_eval,
    }, final_path)
    print(f"\nModel: {final_path}")

    elapsed = time.perf_counter() - t_start
    print(f"Done in {elapsed:.1f}s ({global_step/elapsed:.1f} sps)")
    print(f"Logs: {log_dir}")

    return {
        "run_id": run_id,
        "log_dir": str(log_dir),
        "total_timesteps": global_step,
        "total_episodes": episode_idx,
        "elapsed": elapsed,
        "sps": global_step / max(elapsed, 1e-6),
        "final_eval": final_eval,
    }


# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    train(config)
    print("\nDone.")


if __name__ == "__main__":
    main()
