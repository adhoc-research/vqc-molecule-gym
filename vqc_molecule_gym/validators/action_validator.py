from pydantic import ValidationError

from vqc_molecule_gym.schemas.action import ActionSpec
from vqc_molecule_gym.schemas.result import EvalError
from vqc_molecule_gym.schemas.task import TaskSpec


def validate_action(
    action_payload: dict[str, object],
    task: TaskSpec,
    operator_ids: set[str],
    estimated_depth: int,
) -> tuple[ActionSpec | None, list[EvalError]]:
    errors: list[EvalError] = []
    try:
        action = ActionSpec.model_validate(action_payload)
    except ValidationError as exc:
        return None, [
            EvalError(type="schema_error", message=err["msg"])
            for err in exc.errors(include_url=False)
        ]

    if len(action.operator_sequence) > task.constraints.max_operators:
        errors.append(
            EvalError(
                type="too_many_operators",
                message=f"Expected at most {task.constraints.max_operators} operators.",
            )
        )

    if action.shots > task.constraints.max_shots:
        errors.append(
            EvalError(
                type="too_many_shots",
                message=f"Expected at most {task.constraints.max_shots} shots.",
            )
        )

    unknown = [operator_id for operator_id in action.operator_sequence if operator_id not in operator_ids]
    if unknown:
        errors.append(
            EvalError(
                type="unknown_operator",
                message=f"Unknown operators: {', '.join(unknown)}.",
            )
        )

    if action.parameters and len(action.parameters) != len(action.operator_sequence):
        errors.append(
            EvalError(
                type="parameter_length_mismatch",
                message="Expected exactly one parameter per operator when parameters are provided.",
            )
        )

    out_of_range = [value for value in action.parameters if value < -0.5 or value > 0.5]
    if out_of_range:
        errors.append(
            EvalError(
                type="parameter_out_of_range",
                message="Parameters must be in [-0.5, 0.5] radians.",
            )
        )

    if estimated_depth > task.constraints.max_depth:
        errors.append(
            EvalError(
                type="estimated_depth_too_high",
                message=f"Estimated depth {estimated_depth} exceeds {task.constraints.max_depth}.",
            )
        )

    return action, errors
