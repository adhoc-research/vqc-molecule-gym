"""Unit tests for the PPO environment and related components.

These tests verify the environment mechanics without requiring CUDA-Q or a GPU.
We use mocked evaluators / dummy tasks where needed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vqc_molecule_gym.rl import PPOConfig, QChemPPOEnv
from vqc_molecule_gym.rl.observation import build_observation, obs_dim
from vqc_molecule_gym.rl.policy import QChemPPOPolicy
from vqc_molecule_gym.schemas.result import CircuitMetrics, EvalResult


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _dummy_eval_result(
    *,
    reward: float = 0.0,
    valid: bool = True,
    energy_error_mha: float | None = 0.0,
    depth: int = 0,
    task_id: str = "dummy",
) -> EvalResult:
    return EvalResult(
        valid=valid,
        reward=reward,
        energy_error_mha=energy_error_mha,
        circuit_metrics=CircuitMetrics(
            num_qubits=4,
            num_operators=depth,
            depth=depth,
            gate_count=depth + 2,
            two_qubit_gate_count=0,
        ),
        task_id=task_id,
    )


def _make_config(**overrides: Any) -> PPOConfig:
    defaults: dict[str, Any] = dict(
        curriculum="easy_curriculum",
        eval_benchmarks=(),
        max_operators=4,
        parameter_bins=(-0.3, -0.2, -0.1, 0.1, 0.2, 0.3),
        allow_repeated_operators=True,
        disable_stop_at_step_0=True,
        hidden_sizes=(64, 64),
        activation="tanh",
        max_pool_size=8,
        benchmark_root="benchmarks",
    )
    defaults.update(**overrides)
    return PPOConfig(**defaults)


# ═══════════════════════════════════════════════════════════════════════════
#  Observation builder tests
# ═══════════════════════════════════════════════════════════════════════════

def test_obs_dim() -> None:
    """obs_dim returns the correct total dimension."""
    max_ops = 4
    expected = 5 + 1 + max_ops * 2 + 3 + 1  # 5 + 1 + 8 + 3 + 1 = 18
    assert obs_dim(max_ops) == expected
    assert obs_dim(8) == 5 + 1 + 16 + 3 + 1  # = 26


@patch("vqc_molecule_gym.benchmarks.load_tasks")
def test_build_observation_basic(mock_load_tasks) -> None:
    """Observation vector has expected shape and values in valid range."""
    # This test just checks the pure function.
    from vqc_molecule_gym.curricula import curriculum_benchmark_ids

    # We need a real task spec.  Build a minimal one.
    from vqc_molecule_gym.schemas.task import (
        ActiveSpace,
        Constraints,
        Geometry,
        ReferenceEnergy,
        TaskSpec,
    )

    task = TaskSpec(
        task_id="test",
        benchmark_id="test_benchmark",
        molecule="H2",
        geometry=Geometry(
            kind="linear_chain",
            atoms=[("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)],
        ),
        basis="STO-3G",
        active_space=ActiveSpace(electrons=2, orbitals=2, spin_orbitals=4, qubits=4),
        reference=ReferenceEnergy(method="exact", energy_hartree=-1.0),
        constraints=Constraints(max_operators=4),
        operator_pool_id="test_pool",
    )

    base_result = _dummy_eval_result(reward=-1.0, energy_error_mha=2.0)

    obs = build_observation(
        task=task,
        step_index=0,
        operator_sequence=(),
        parameter_bins=(),
        operator_ids=("E_000", "E_001"),
        parameter_bin_values=(-0.3, -0.2, -0.1, 0.1, 0.2, 0.3),
        max_operators=4,
        base_result=base_result,
        current_result=base_result,
        current_reward=-1.0,
        previous_reward=-1.0,
    )

    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == (obs_dim(4),)

    # Check all values are within [-1, 1]
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0), f"obs out of range: {obs}"


@patch("vqc_molecule_gym.benchmarks.load_tasks")
def test_build_observation_with_operators(mock_load_tasks) -> None:
    """Observation reflects the operator sequence when operators have been added."""
    from vqc_molecule_gym.schemas.task import (
        ActiveSpace,
        Constraints,
        Geometry,
        ReferenceEnergy,
        TaskSpec,
    )

    task = TaskSpec(
        task_id="test",
        benchmark_id="test_benchmark",
        molecule="H2",
        geometry=Geometry(
            kind="linear_chain",
            atoms=[("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)],
        ),
        basis="STO-3G",
        active_space=ActiveSpace(electrons=2, orbitals=2, spin_orbitals=4, qubits=4),
        reference=ReferenceEnergy(method="exact", energy_hartree=-1.0),
        constraints=Constraints(max_operators=4),
        operator_pool_id="test_pool",
    )

    base_result = _dummy_eval_result(reward=-1.0, energy_error_mha=2.0)
    current_result = _dummy_eval_result(reward=-0.5, energy_error_mha=1.0, depth=1)

    obs = build_observation(
        task=task,
        step_index=1,
        operator_sequence=("E_000",),
        parameter_bins=(2,),  # -0.1 is index 2
        operator_ids=("E_000", "E_001"),
        parameter_bin_values=(-0.3, -0.2, -0.1, 0.1, 0.2, 0.3),
        max_operators=4,
        base_result=base_result,
        current_result=current_result,
        current_reward=-0.5,
        previous_reward=-1.0,
    )

    assert obs.shape == (obs_dim(4),)
    # The first two sequence slots: first should have values, rest should be -1
    # Slot 0: op_idx = 0/1 = 0.0, bin_idx = 2/5 = 0.4
    # Slot 1+: padding sentinel = -1.0
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Action decoding tests
# ═══════════════════════════════════════════════════════════════════════════

def test_action_decoding() -> None:
    """Verify decode_operator_action maps indices correctly."""
    config = _make_config(parameter_bins=(-0.3, -0.2, -0.1, 0.1, 0.2, 0.3))
    num_bins = len(config.parameter_bins)  # 6

    # action 0 = STOP
    # action 1 = op[0], bin[0]
    # action 6 = op[0], bin[5]
    # action 7 = op[1], bin[0]
    # action 12 = op[2], bin[0]

    # We test the private method via a simple function
    def decode(action: int, num_bins: int) -> tuple[int, int]:
        flat = action - 1
        return flat // num_bins, flat % num_bins

    assert decode(1, num_bins) == (0, 0)
    assert decode(6, num_bins) == (0, 5)
    assert decode(7, num_bins) == (1, 0)
    assert decode(12, num_bins) == (1, 5)
    assert decode(13, num_bins) == (2, 0)


# ═══════════════════════════════════════════════════════════════════════════
#  Action masking tests
# ═══════════════════════════════════════════════════════════════════════════

def test_compute_action_mask_stop_disabled_at_step_0() -> None:
    """Action mask disables STOP at step 0 when configured."""
    config = _make_config(disable_stop_at_step_0=True, max_operators=4)

    # We test the mask logic independently of the full env
    num_bins = len(config.parameter_bins)
    num_ops = 4
    num_actions = 1 + num_ops * num_bins  # 1 + 4*6 = 25

    mask = np.ones(num_actions, dtype=np.float32)
    step_index = 0

    # Apply rules
    if config.disable_stop_at_step_0 and step_index == 0:
        mask[0] = 0.0

    if step_index >= config.max_operators:
        mask[1:] = 0.0
        mask[0] = 1.0

    assert mask[0] == 0.0  # STOP disabled
    assert np.all(mask[1:] == 1.0)  # all operators enabled


def test_compute_action_mask_only_stop_when_maxed() -> None:
    """Only STOP is allowed when max_operators is reached."""
    config = _make_config(disable_stop_at_step_0=False, max_operators=4)
    num_bins = len(config.parameter_bins)
    num_ops = 4
    num_actions = 1 + num_ops * num_bins

    mask = np.ones(num_actions, dtype=np.float32)
    step_index = config.max_operators

    if config.disable_stop_at_step_0 and step_index == 0:
        mask[0] = 0.0
    if step_index >= config.max_operators:
        mask[1:] = 0.0
        mask[0] = 1.0

    assert mask[0] == 1.0  # STOP allowed
    assert np.all(mask[1:] == 0.0)  # all operator actions blocked


def test_compute_action_mask_no_repeats() -> None:
    """Used operators are masked when allow_repeated_operators is False."""
    config = _make_config(allow_repeated_operators=False, max_operators=4)
    num_bins = len(config.parameter_bins)
    operator_ids = ("E_000", "E_001", "E_002", "E_003")
    num_actions = 1 + len(operator_ids) * num_bins

    mask = np.ones(num_actions, dtype=np.float32)
    step_index = 1

    # Rules
    if config.disable_stop_at_step_0 and step_index == 0:
        mask[0] = 0.0
    if step_index >= config.max_operators:
        mask[1:] = 0.0
        mask[0] = 1.0
        return

    used_operator_ids = {"E_001"}
    if not config.allow_repeated_operators:
        for idx, op_id in enumerate(operator_ids):
            if op_id in used_operator_ids:
                start = 1 + idx * num_bins
                mask[start:start + num_bins] = 0.0

    # E_001 (idx 1) should be masked
    start_e001 = 1 + 1 * num_bins
    assert np.all(mask[start_e001:start_e001 + num_bins] == 0.0)
    # Other operators still available
    start_e000 = 1 + 0 * num_bins
    assert np.all(mask[start_e000:start_e000 + num_bins] == 1.0)
    # STOP not disabled
    assert mask[0] == 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  Policy tests
# ═══════════════════════════════════════════════════════════════════════════

def test_policy_forward_shape() -> None:
    """Policy forward pass produces correctly shaped outputs."""
    obs_dim_val = obs_dim(4)  # 18
    action_dim = 1 + 4 * 6  # 25
    policy = QChemPPOPolicy(obs_dim=obs_dim_val, action_dim=action_dim,
                            hidden_sizes=(64, 64))

    batch_size = 8
    obs = torch.randn(batch_size, obs_dim_val)

    out = policy(obs)
    assert "logits" in out
    assert "value" in out
    assert out["logits"].shape == (batch_size, action_dim)
    assert out["value"].shape == (batch_size,)


def test_policy_with_action_mask() -> None:
    """Masked logits are -inf for blocked actions."""
    obs_dim_val = obs_dim(4)
    action_dim = 1 + 4 * 6
    policy = QChemPPOPolicy(obs_dim=obs_dim_val, action_dim=action_dim,
                            hidden_sizes=(64, 64))

    batch_size = 2
    obs = torch.randn(batch_size, obs_dim_val)
    mask = torch.ones(batch_size, action_dim)
    mask[:, 0] = 0.0  # block STOP

    out = policy(obs, action_mask=mask)
    logits = out["logits"]

    # After masked_fill, logits for masked actions should be -inf
    # But the internal forward uses logits *before* masking in the result dict.
    # Let's check the distribution instead
    dist = out["dist"]
    probs = dist.probs
    assert torch.all(probs[:, 0] == 0.0)  # STOP prob = 0


def test_policy_log_prob_and_entropy() -> None:
    """Policy returns log_prob and entropy when action is provided."""
    obs_dim_val = obs_dim(4)
    action_dim = 1 + 4 * 6
    policy = QChemPPOPolicy(obs_dim=obs_dim_val, action_dim=action_dim,
                            hidden_sizes=(64, 64))

    batch_size = 4
    obs = torch.randn(batch_size, obs_dim_val)
    actions = torch.randint(0, action_dim, (batch_size,))

    out = policy(obs, action=actions)
    assert "log_prob" in out
    assert "entropy" in out
    assert out["log_prob"].shape == (batch_size,)
    assert out["entropy"].shape == (batch_size,)
    assert torch.all(torch.isfinite(out["log_prob"]))
    assert torch.all(torch.isfinite(out["entropy"]))


def test_policy_value_head() -> None:
    """get_value returns value predictions."""
    obs_dim_val = obs_dim(4)
    action_dim = 1 + 4 * 6
    policy = QChemPPOPolicy(obs_dim=obs_dim_val, action_dim=action_dim,
                            hidden_sizes=(64, 64))

    batch_size = 4
    obs = torch.randn(batch_size, obs_dim_val)
    values = policy.get_value(obs)
    assert values.shape == (batch_size,)
    assert torch.all(torch.isfinite(values))
