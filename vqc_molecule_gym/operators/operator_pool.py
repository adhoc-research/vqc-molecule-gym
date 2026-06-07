from dataclasses import dataclass
from typing import Any

import cudaq_solvers


@dataclass(frozen=True)
class OperatorEntry:
    operator_id: str
    cudaq_operator: Any
    serialized: str
    term_count: int

    def to_json(self) -> dict[str, object]:
        return {
            "operator_id": self.operator_id,
            "kind": "uccsd_pool_operator",
            "description": self.serialized,
            "term_count": self.term_count,
        }


@dataclass(frozen=True)
class OperatorPool:
    pool_id: str
    entries: list[OperatorEntry]

    @property
    def ids(self) -> set[str]:
        return {entry.operator_id for entry in self.entries}

    def by_id(self) -> dict[str, OperatorEntry]:
        return {entry.operator_id: entry for entry in self.entries}

    def to_json(self) -> dict[str, object]:
        return {"operator_pool_id": self.pool_id, "operators": [entry.to_json() for entry in self.entries]}


def build_operator_pool(pool_id: str, *, num_qubits: int, num_electrons: int) -> OperatorPool:
    raw_ops = cudaq_solvers.get_operator_pool(
        "uccsd",
        num_qubits=num_qubits,
        num_electrons=num_electrons,
    )
    sorted_ops = sorted(raw_ops, key=lambda op: repr(op.serialize()))
    entries = [
        OperatorEntry(
            operator_id=f"E_{idx:03d}",
            cudaq_operator=op,
            serialized=repr(op.serialize()),
            term_count=int(op.term_count),
        )
        for idx, op in enumerate(sorted_ops)
    ]
    return OperatorPool(pool_id=pool_id, entries=entries)
