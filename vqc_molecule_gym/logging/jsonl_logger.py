import json
from pathlib import Path

from pydantic import BaseModel


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: BaseModel | dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            if isinstance(record, BaseModel):
                f.write(record.model_dump_json() + "\n")
            else:
                f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
