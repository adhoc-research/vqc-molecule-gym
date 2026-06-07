from datetime import datetime
from typing import Any

from vqc_molecule_gym.schemas.base import StrictModel


class TrajectoryRecord(StrictModel):
    run_id: str
    episode_id: str
    timestamp: datetime
    agent: str
    benchmark_id: str
    task_id: str
    prompt_hash: str | None = None
    completion_raw: str
    action: dict[str, Any] | None = None
    result: dict[str, Any]
    software_versions: dict[str, str]
