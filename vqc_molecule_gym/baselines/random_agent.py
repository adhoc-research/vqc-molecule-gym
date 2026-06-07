import random

from vqc_molecule_gym.schemas.action import ActionSpec
from vqc_molecule_gym.schemas.task import TaskSpec


class RandomAgent:
    name = "random"

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def act(self, task: TaskSpec, operator_ids: list[str]) -> ActionSpec:
        max_ops = task.constraints.max_operators
        length = self.rng.randint(0, max_ops)
        sequence = [self.rng.choice(operator_ids) for _ in range(length)] if operator_ids else []
        return ActionSpec(operator_sequence=sequence, shots=min(10000, task.constraints.max_shots))
