import math

from pydantic import Field, field_validator

from vqc_molecule_gym.schemas.base import StrictModel


class ActionSpec(StrictModel):
    operator_sequence: list[str] = Field(default_factory=list)
    parameters: list[float] = Field(default_factory=list)
    shots: int = Field(gt=0)

    @field_validator("parameters")
    @classmethod
    def parameters_must_be_finite(cls, parameters: list[float]) -> list[float]:
        if not all(math.isfinite(value) for value in parameters):
            raise ValueError("parameters must be finite floats")
        return parameters
