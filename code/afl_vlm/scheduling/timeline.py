"""Stable virtual ordering, with optional causally selected refresh events."""

from __future__ import annotations

import heapq
from typing import Any


class EventTimeline:
    def __init__(self, plan: list[Any]) -> None:
        self.heap: list[Any] = []
        for event in plan:
            self.push(event.start_time, 2, "start", event)
            self.push(event.arrival_time, 0, "arrival", event)

    def push(self, time: float, priority: int, kind: str, event: Any) -> None:
        heapq.heappush(
            self.heap,
            (time, priority, event.client_id, event.local_round, event.event_id, kind, event),
        )

    def refresh(self, event: Any, step: int, total_steps: int) -> float:
        if not 0 < step < total_steps:
            raise ValueError("Refresh must be strictly inside the local optimizer-step interval")
        time = event.start_time + step / total_steps * event.estimated_train_time
        self.push(time, 1, "refresh", event)
        return time

    def __iter__(self):
        while self.heap:
            time, priority, _, _, _, kind, event = heapq.heappop(self.heap)
            yield time, priority, kind, event
