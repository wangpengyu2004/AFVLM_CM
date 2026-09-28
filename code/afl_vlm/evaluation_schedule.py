"""Method-independent periodic evaluation accounting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PeriodicEvaluationSchedule:
    """Advance evaluation milestones without assuming one update per aggregation."""

    unit: str
    interval: int
    next_threshold: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> PeriodicEvaluationSchedule:
        configured_unit = str(
            config.get("interval_unit", "incorporated_client_updates")
        )
        if configured_unit == "incorporated_client_updates":
            interval = int(config.get("eval_every_client_updates", 0))
        elif configured_unit == "server_updates":
            # Migrate immutable pre-change profiles to the fair protocol while
            # retaining their configured numeric interval.
            interval = int(config.get("eval_every_server_updates", 0))
        else:
            raise ValueError(f"Unsupported evaluation interval unit: {configured_unit}")
        if interval <= 0:
            raise ValueError("Periodic evaluation interval must be positive")
        return cls(
            unit="incorporated_client_updates",
            interval=interval,
            next_threshold=interval,
        )

    def observe(self, count: int) -> tuple[int, ...]:
        """Return newly crossed milestones, evaluating one current model once."""

        if count < 0:
            raise ValueError("Evaluation progress count cannot be negative")
        crossed: list[int] = []
        while count >= self.next_threshold:
            crossed.append(self.next_threshold)
            self.next_threshold += self.interval
        return tuple(crossed)


def evaluation_progress_count(
    server: Any,
    evaluation_scope: str,
    unit: str,
) -> int:
    """Return the comparable progress counter for the configured protocol."""

    if unit == "server_updates":
        return int(server.version)
    if unit != "incorporated_client_updates":
        raise ValueError(f"Unsupported evaluation interval unit: {unit}")
    if evaluation_scope == "client_local_mean":
        # Local-only updates are incorporated into client models without a
        # global mutation, so each received local result advances its budget.
        return int(server.received_updates)
    return int(server.accepted_updates)
