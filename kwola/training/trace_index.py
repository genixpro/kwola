"""Indexed views over immutable recorded traces used during one training step."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

TraceRecord = tuple[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class TraceIndex:
    ordered: Sequence[TraceRecord]
    by_step: Mapping[str, tuple[TraceRecord, ...]]
    by_position: Mapping[tuple[str, int], TraceRecord]

    @classmethod
    def build(cls, traces: Sequence[TraceRecord]) -> "TraceIndex":
        grouped: dict[str, list[TraceRecord]] = {}
        positions: dict[tuple[str, int], TraceRecord] = {}
        for record in traces:
            trace = record[1]
            step_id = str(trace["step_id"])
            grouped.setdefault(step_id, []).append(record)
            positions[(step_id, int(trace["index"]))] = record
        return cls(
            traces,
            {step_id: tuple(records) for step_id, records in grouped.items()},
            positions,
        )

    def step(self, trace: Mapping[str, Any]) -> tuple[TraceRecord, ...]:
        return self.by_step[str(trace["step_id"])]

    def next_samples(self, selected: Sequence[TraceRecord]) -> tuple[list[TraceRecord], list[bool]]:
        samples = []
        validity = []
        for current in selected:
            trace = current[1]
            next_trace = self.by_position.get((str(trace["step_id"]), int(trace["index"]) + 1))
            samples.append(next_trace or current)
            validity.append(next_trace is not None)
        return samples, validity


def trace_order(item: tuple[str, Mapping[str, Any]]) -> tuple[str, int]:
    return str(item[1]["step_id"]), int(item[1]["index"])


def cache_payload(trace_ids: Sequence[str]) -> dict[str, Any]:
    return {"trace_ids": list(trace_ids)}
