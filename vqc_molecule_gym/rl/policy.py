"""Simple MLP actor-critic policy for the QChem PPO agent.

Architecture::

    observation vector
           ↓
    MLP trunk (256 → 256, Tanh)
           ↓
    ┌─────────────────┬──────────────────┐
    │ policy_head      │ value_head       │
    │ Linear → logits  │ Linear → scalar  │
    └─────────────────┴──────────────────┘
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class QChemPPOPolicy(nn.Module):
    """MLP actor-critic policy that outputs action logits and a state value.

    This can be used with PufferLib or wrapped as needed.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
        activation: Literal["tanh", "relu"] = "tanh",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        act_fn: type[nn.Module] = nn.Tanh if activation == "tanh" else nn.ReLU

        # ── Trunk ──────────────────────────────────────────────────────────
        trunk_layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            trunk_layers.append(nn.Linear(prev, h))
            trunk_layers.append(act_fn())
            prev = h
        self.trunk = nn.Sequential(*trunk_layers)

        # ── Heads ──────────────────────────────────────────────────────────
        self.policy_head = nn.Linear(prev, action_dim)
        self.value_head = nn.Linear(prev, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Policy head: smaller init for stable start
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute logits, value, and optionally log-prob of the given action.

        Parameters
        ----------
        obs : shape ``(batch, obs_dim)``
        action : shape ``(batch,)`` or ``None``
        action_mask : shape ``(batch, action_dim)``, optional
            1 = allowed, 0 = blocked.  Logits of blocked actions are set to
            -inf before the softmax.

        Returns
        -------
        dict with keys ``logits``, ``value``, and optionally ``log_prob``,
        ``entropy``, ``dist``.
        """
        features = self.trunk(obs)
        logits = self.policy_head(features)  # (batch, action_dim)
        value = self.value_head(features)  # (batch, 1)

        result: dict[str, torch.Tensor] = {
            "logits": logits,
            "value": value.squeeze(-1),
        }

        # ── Mask logits (if provided) ──────────────────────────────────────
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float("-inf"))

        # ── Categorical distribution ───────────────────────────────────────
        dist = torch.distributions.Categorical(logits=logits)
        result["dist"] = dist  # type: ignore[assignment]

        if action is not None:
            result["log_prob"] = dist.log_prob(action)
            result["entropy"] = dist.entropy()

        return result

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the state value for a batch of observations (no grad)."""
        with torch.no_grad():
            features = self.trunk(obs)
            return self.value_head(features).squeeze(-1)
