import json
import re
from typing import Any


FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
FINAL_ACTION_RE = re.compile(r"Final action:\s*(\{.*\})", re.DOTALL | re.IGNORECASE)


def parse_completion(completion: str) -> tuple[dict[str, Any] | None, str | None]:
    candidates = [completion.strip()]
    for pattern in (FENCED_JSON_RE, FINAL_ACTION_RE):
        match = pattern.search(completion)
        if match:
            candidates.insert(0, match.group(1).strip())

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, None
        return None, "completion_json_not_object"
    return None, "invalid_json"
