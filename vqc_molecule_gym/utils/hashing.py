import hashlib
import json
from typing import Any


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()
