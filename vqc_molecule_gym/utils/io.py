import json
from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel, TypeAdapter

T = TypeVar("T", bound=BaseModel)


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def read_jsonl_models(path: Path, model_type: type[T]) -> list[T]:
    adapter = TypeAdapter(model_type)
    rows: list[T] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(adapter.validate_json(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            if isinstance(row, BaseModel):
                f.write(row.model_dump_json() + "\n")
            else:
                f.write(json.dumps(row, sort_keys=True) + "\n")
