from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PPOConfig:
    """Hyperparameters and settings for PPO training on the QChem molecule gym."""

    # -- Environment -----------------------------------------------------------
    curriculum: str = "easy_curriculum"
    eval_benchmarks: tuple[str, ...] = (
        "h2_tiny_v0",
        "lih_bond_scan_v0",
        "h4_small_v0",
        "n2_bond_scan_v0",
    )
    max_operators: int = 4
    parameter_bins: tuple[float, ...] = (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3)
    allow_repeated_operators: bool = True
    disable_stop_at_step_0: bool = True
    shots: int = 10_000

    # -- Policy network --------------------------------------------------------
    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: Literal["tanh", "relu"] = "tanh"

    # -- PPO hyperparameters ---------------------------------------------------
    learning_rate: float = 3e-4
    total_timesteps: int = 1_000_000
    num_envs: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.05
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    batch_size: int = 64

    # -- Evaluation / logging --------------------------------------------------
    eval_frequency: int = 50_000
    log_dir: str = "runs"
    seed: int = 42

    # -- Reward ----------------------------------------------------------------
    stop_bonus_weight: float = 0.1

    # -- Misc ------------------------------------------------------------------
    max_pool_size: int = 64  # padded observation dimension for operator count
    benchmark_root: str = "benchmarks"  # will be used as Path(..) in env

    def __post_init__(self) -> None:
        if self.max_operators < 1:
            raise ValueError("max_operators must be at least 1")
        if not self.parameter_bins:
            raise ValueError("parameter_bins must not be empty")
