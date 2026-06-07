"""Fixed-size feature-vector observation builder for the PPO environment."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from vqc_molecule_gym.schemas.result import EvalResult
from vqc_molecule_gym.schemas.task import TaskSpec


# ── Feature block sizes ────────────────────────────────────────────────────
TASK_FEATURES = 5       # active_electrons, active_orbitals, num_qubits,
                        # base_error_mha, max_operators
STEP_FEATURE = 1        # step_index / max_operators
SEQ_FEATURES_PER_OPS = 2  # one ID + one param-bin per past operator
CIRCUIT_FEATURES = 3    # depth / max_depth, error / base_error, improvement
REWARD_FEATURE = 1      # current reward


def obs_dim(max_operators: int) -> int:
    """Return the total flattened observation dimension for a given max_ops."""
    return (
        TASK_FEATURES
        + STEP_FEATURE
        + max_operators * SEQ_FEATURES_PER_OPS
        + CIRCUIT_FEATURES
        + REWARD_FEATURE
    )


def observation_space(max_operators: int) -> spaces.Box:
    """Create the Box observation space for a given max_operators."""
    dim = obs_dim(max_operators)
    return spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)


def build_observation(
    *,
    task: TaskSpec,
    step_index: int,
    operator_sequence: tuple[str, ...],
    parameter_bins: tuple[int, ...],
    operator_ids: tuple[str, ...],
    parameter_bin_values: tuple[float, ...],
    max_operators: int,
    base_result: EvalResult | None,
    current_result: EvalResult | None,
    current_reward: float,
    previous_reward: float,
) -> np.ndarray:
    """Build a fixed-size numeric observation vector for the policy.

    All values are normalised to approximately [-1, 1] or [0, 1].

    Returns
    -------
    np.ndarray of shape ``(obs_dim(max_operators),)`` with dtype float32.
    """
    parts: list[float] = []

    # ── 1. Task features (normalised) ────────────────────────────────────────
    active_electrons = float(task.active_space.electrons) / 20.0
    active_orbitals = float(task.active_space.orbitals) / 20.0
    num_qubits = float(task.active_space.qubits) / 20.0
    base_error_mha = (
        max(0.0, base_result.energy_error_mha) / 10.0
        if base_result is not None and base_result.energy_error_mha is not None
        else 0.0
    )
    max_ops_norm = float(max_operators) / 10.0
    parts.extend([
        _clip01(active_electrons),
        _clip01(active_orbitals),
        _clip01(num_qubits),
        _clip01(base_error_mha),
        _clip01(max_ops_norm),
    ])

    # ── 2. Step index (normalised) ───────────────────────────────────────────
    step_norm = float(step_index) / max(max_operators, 1)
    parts.append(_clip01(step_norm))

    # ── 3. Previous operator IDs and parameter bins (padded) ─────────────────
    num_ids = len(operator_ids)
    num_bins = len(parameter_bin_values)
    for i in range(max_operators):
        if i < len(operator_sequence):
            op_id = operator_sequence[i]
            op_idx = float(operator_ids.index(op_id)) / max(num_ids - 1, 1)
            bin_idx = float(parameter_bins[i]) / max(num_bins - 1, 1)
            parts.append(_clip01(op_idx))
            parts.append(_clip01(bin_idx))
        else:
            parts.append(-1.0)  # padding sentinel
            parts.append(-1.0)

    # ── 4. Circuit metrics (normalised) ──────────────────────────────────────
    depth = (
        float(current_result.circuit_metrics.depth) / max(task.constraints.max_depth, 1)
        if current_result is not None and current_result.circuit_metrics is not None
        else 0.0
    )
    error = (
        max(0.0, current_result.energy_error_mha) / 10.0
        if current_result is not None and current_result.energy_error_mha is not None
        else 0.0
    )
    improvement = (
        (previous_reward - current_reward)
        if current_result is not None and base_result is not None
        else 0.0
    )
    parts.extend([
        _clip01(depth),
        _clip01(error),
        float(np.clip(improvement, -1.0, 1.0)),
    ])

    # ── 5. Current reward ────────────────────────────────────────────────────
    parts.append(float(np.clip(current_reward, -1.0, 1.0)))

    return np.array(parts, dtype=np.float32)


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))
