"""Gymnasium environment for stepwise circuit construction with dense prefix rewards.

The agent builds a circuit one operator at a time. At each step it chooses either
STOP (action 0) or an (operator_id, parameter_bin) pair from a discrete action
space. After every added operator the partial circuit is evaluated and the agent
receives a dense reward equal to the improvement in the overall reward function
over the previous partial circuit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from vqc_molecule_gym.baselines.search_helpers import SearchCache
from vqc_molecule_gym.benchmarks import load_tasks
from vqc_molecule_gym.curricula import canonical_benchmark_id, curriculum_benchmark_ids
from vqc_molecule_gym.evaluators.direct_energy import DirectEnergyEvaluator
from vqc_molecule_gym.operators.operator_pool import OperatorPool, build_operator_pool
from vqc_molecule_gym.rl.config import PPOConfig
from vqc_molecule_gym.rl.observation import build_observation, observation_space
from vqc_molecule_gym.schemas.action import ActionSpec
from vqc_molecule_gym.schemas.result import EvalResult
from vqc_molecule_gym.schemas.task import TaskSpec


class QChemPPOEnv(gym.Env):
    """Stepwise circuit-building environment for PPO.

    At each step the agent picks one discrete action:
    - ``0`` = STOP (terminate episode)
    - ``1 .. num_actions-1`` = (operator_id, parameter_bin) to append to the circuit

    After every operator addition the current partial circuit is evaluated and a
    dense prefix reward is computed as the improvement over the previous step.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: PPOConfig,
        *,
        benchmark_ids: tuple[str, ...] | None = None,
    ) -> None:
        self.config = config
        self._benchmark_ids = benchmark_ids or curriculum_benchmark_ids(config.curriculum)
        self._tasks: list[TaskSpec] | None = None  # loaded lazily in reset
        self._evaluator = DirectEnergyEvaluator()

        # Set observation space (Box) — dimension depends on max_operators.
        self.observation_space = observation_space(config.max_operators)

        # Pre-compute the maximum pool size across all tasks so we can fix the
        # action space dimension once (different tasks have different pool sizes).
        self._max_operator_count = self._compute_max_pool_size()
        self._max_num_actions = 1 + self._max_operator_count * len(config.parameter_bins)
        self.action_space = gym.spaces.Discrete(self._max_num_actions)

        # ── Per-episode state ────────────────────────────────────────────────
        self._task: TaskSpec | None = None
        self._pool: OperatorPool | None = None
        self._operator_ids: tuple[str, ...] = ()
        self._parameter_bin_values: tuple[float, ...] = ()
        self._parameter_bins: list[int] = []  # bin indices chosen
        self._operator_sequence: list[str] = []
        self._step_index: int = 0
        self._done: bool = False
        self._base_result: EvalResult | None = None
        self._current_result: EvalResult | None = None
        self._last_reward: float = 0.0  # reward of the previous (or base) circuit
        self._action_mask: np.ndarray | None = None
        self._search_cache: SearchCache | None = None

    # ── Public API ───────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode on a randomly sampled task from the curriculum.

        Returns
        -------
        observation : np.ndarray
            Feature vector for the empty-circuit state.
        info : dict
            Contains ``action_mask``, ``task_id``, ``num_operators``, etc.
        """
        super().reset(seed=seed)

        # ── Sample a task ────────────────────────────────────────────────────────
        tasks = self._load_tasks()
        rng = np.random.default_rng(self._get_rng_seed())
        idx = int(rng.integers(0, len(tasks)))
        self._task = tasks[idx]

        # ── Load operator pool ───────────────────────────────────────────────────
        self._pool = build_operator_pool(
            self._task.operator_pool_id,
            num_qubits=self._task.active_space.qubits,
            num_electrons=self._task.active_space.electrons,
        )
        self._operator_ids = tuple(sorted(self._pool.ids))
        self._parameter_bin_values = tuple(self.config.parameter_bins)

        # ── Action space is fixed (from __init__) — we just track the actual
        #    number of valid actions for masking below.
        self._num_valid_actions = 1 + len(self._operator_ids) * len(self._parameter_bin_values)

        # ── Evaluate empty circuit (base) ────────────────────────────────────────
        self._search_cache = SearchCache(
            self._evaluator, self._task, shots=self.config.shots
        )
        base_entry = self._search_cache.entry((), ())
        self._base_result = base_entry.result
        self._current_result = self._base_result
        self._last_reward = float(self._base_result.reward)

        # ── Reset episode state ──────────────────────────────────────────────────
        self._operator_sequence = []
        self._parameter_bins = []
        self._step_index = 0
        self._done = False

        # ── Build action mask ────────────────────────────────────────────────────
        self._action_mask = self._compute_action_mask()

        # ── Build initial observation ────────────────────────────────────────────
        obs = self._make_obs()
        info = self._make_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one action and return the next state, reward, termination flags and info.

        Parameters
        ----------
        action : int
            Discrete action index. 0 = STOP, > 0 = (operator_id, param_bin).

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        if self._done:
            raise RuntimeError("Episode already done; call reset() first.")

        if self._action_mask is not None and self._action_mask[action] == 0:
            # Invalid action — treat as STOP with penalty.
            reward = -1.0
            terminated = True
            self._done = True
            obs = self._make_obs()
            return obs, reward, terminated, False, self._make_info()

        # ── Decode action ────────────────────────────────────────────────────────
        if action == 0:
            # STOP
            if self._operator_sequence:
                final_result = self._evaluate_current()
                final_reward = float(final_result.reward)
            else:
                final_reward = self._last_reward

            step_reward = self._last_reward  # reward for keeping existing circuit
            step_reward += self.config.stop_bonus_weight * final_reward
            self._done = True
            obs = self._make_obs()
            return obs, step_reward, True, False, self._make_info()

        # ── Operator action ──────────────────────────────────────────────────────
        op_idx, bin_idx = self._decode_operator_action(action)
        operator_id = self._operator_ids[op_idx]
        parameter = self._parameter_bin_values[bin_idx]

        self._operator_sequence.append(operator_id)
        self._parameter_bins.append(bin_idx)
        self._step_index += 1

        before_reward = self._last_reward
        current_result = self._evaluate_current()
        after_reward = float(current_result.reward)

        # Dense prefix reward: improvement over previous step
        step_reward = after_reward - before_reward

        self._last_reward = after_reward
        self._current_result = current_result

        # ── Check termination ────────────────────────────────────────────────────
        terminated = False
        if self._step_index >= self.config.max_operators:
            # Max operators reached — next step only STOP is allowed; we let the
            # agent take one more STOP step.  We signal truncation only if the
            # agent has maxed out *and* we haven't given the STOP chance yet.
            # For v0 we force-trigger STOP logic.
            step_reward += self.config.stop_bonus_weight * after_reward
            terminated = True
            self._done = True

        # ── Build mask for the next step ─────────────────────────────────────────
        self._action_mask = self._compute_action_mask()

        obs = self._make_obs()
        return obs, step_reward, terminated, False, self._make_info()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_max_pool_size(self) -> int:
        """Return the maximum number of operators across all tasks in the curriculum."""
        max_ops = 0
        for task in self._load_tasks():
            pool = build_operator_pool(
                task.operator_pool_id,
                num_qubits=task.active_space.qubits,
                num_electrons=task.active_space.electrons,
            )
            max_ops = max(max_ops, len(pool.ids))
        return max_ops

    def _compute_action_mask(self) -> np.ndarray:
        """Return a binary mask of shape ``(max_num_actions,)`` for the next step.

        1 = allowed, 0 = blocked.
        """
        num_actions = int(self.action_space.n)
        mask = np.ones(num_actions, dtype=np.float32)

        # Rule: actions beyond the current task's valid range are always blocked
        if self._num_valid_actions < num_actions:
            mask[self._num_valid_actions:] = 0.0

        # Rule: disable STOP at step 0
        if self.config.disable_stop_at_step_0 and self._step_index == 0:
            mask[0] = 0.0

        # Rule: if max operators reached, only STOP allowed
        if self._step_index >= self.config.max_operators:
            mask[1:] = 0.0
            mask[0] = 1.0  # ensure STOP is always allowed when maxed out
            return mask

        # Rule: if operator repeats are disallowed, mask used operator IDs
        if not self.config.allow_repeated_operators:
            used_operator_ids = set(self._operator_sequence)
            num_bins = len(self._parameter_bin_values)
            for idx, op_id in enumerate(self._operator_ids):
                if op_id in used_operator_ids:
                    start = 1 + idx * num_bins
                    mask[start : start + num_bins] = 0.0

        # Rule: if mask is all zeros prevent deadlock - allow at least STOP
        if mask.sum() == 0:
            mask[0] = 1.0

        return mask

    def _decode_operator_action(self, action: int) -> tuple[int, int]:
        """Convert a flat discrete action index to (operator_idx, param_bin_idx)."""
        num_bins = len(self._parameter_bin_values)
        flat = action - 1  # shift past STOP
        op_idx = flat // num_bins
        bin_idx = flat % num_bins
        return int(op_idx), int(bin_idx)

    def _evaluate_current(self) -> EvalResult:
        """Evaluate the current partial circuit using the search cache."""
        seq = tuple(self._operator_sequence)
        params = tuple(
            self._parameter_bin_values[b] for b in self._parameter_bins
        )
        entry = self._search_cache.entry(seq, params)
        return entry.result

    def _make_obs(self) -> np.ndarray:
        """Build the observation vector from the current state."""
        return build_observation(
            task=self._task,
            step_index=self._step_index,
            operator_sequence=tuple(self._operator_sequence),
            parameter_bins=tuple(self._parameter_bins),
            operator_ids=self._operator_ids,
            parameter_bin_values=self._parameter_bin_values,
            max_operators=self.config.max_operators,
            base_result=self._base_result,
            current_result=self._current_result,
            current_reward=self._last_reward,
            previous_reward=self._last_reward,
        )

    def _make_info(self) -> dict[str, Any]:
        """Build the info dict for the current step."""
        return {
            "action_mask": self._action_mask.copy() if self._action_mask is not None else None,
            "task_id": self._task.task_id if self._task is not None else None,
            "num_operators": len(self._operator_sequence),
            "operator_sequence": list(self._operator_sequence),
            "parameter_bins": list(self._parameter_bins),
            "step_index": self._step_index,
            "reward": self._last_reward,
            "num_actions": int(self.action_space.n),
            "num_valid_actions": self._num_valid_actions,
        }

    def _load_tasks(self) -> list[TaskSpec]:
        """Load and cache all tasks for the configured benchmarks."""
        if self._tasks is not None:
            return self._tasks
        tasks: list[TaskSpec] = []
        for bid in self._benchmark_ids:
            canonical = canonical_benchmark_id(bid)
            tasks.extend(load_tasks(canonical, root=Path(self.config.benchmark_root)))
        self._tasks = tasks
        return self._tasks

    def _get_rng_seed(self) -> int | None:
        """Return a fresh seed derived from the env's internal RNG."""
        if self.np_random is None:
            return None
        return int(self.np_random.integers(0, 2**31 - 1))
