from typing import Protocol

from vqc_molecule_gym.schemas.result import EvalResult
from vqc_molecule_gym.schemas.task import TaskSpec


class SearchEvaluator(Protocol):
    """Evaluator contract used by search baselines.

    Any evaluator with this method can be used by greedy/beam agents. Keeping the
    contract small lets unit tests inject lightweight fake evaluators without
    importing or running CUDA-Q.
    """

    def evaluate_payload(self, task: TaskSpec, action_payload: dict[str, object]) -> EvalResult:
        ...
