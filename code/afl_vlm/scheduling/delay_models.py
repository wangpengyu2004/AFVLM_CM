"""Deterministic virtual delay implementations."""

from __future__ import annotations

from collections.abc import Mapping
from random import Random
from typing import Any

from afl_vlm.scheduling.base import DelayModel


class FixedDelay(DelayModel):
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def duration(self, client_id: str, task: str, local_round: int, rng: Random) -> float:
        return self.value


class PerTaskDelay(DelayModel):
    def __init__(self, values: Mapping[str, float]) -> None:
        self.values = {str(key): float(value) for key, value in values.items()}

    def duration(self, client_id: str, task: str, local_round: int, rng: Random) -> float:
        return self.values[task]


class PerClientDelay(DelayModel):
    def __init__(self, values: Mapping[str, float]) -> None:
        self.values = {str(key): float(value) for key, value in values.items()}

    def duration(self, client_id: str, task: str, local_round: int, rng: Random) -> float:
        return self.values[client_id]


class ScriptedDelay(DelayModel):
    def __init__(self, values: Mapping[str, list[float]]) -> None:
        self.values = {str(key): [float(item) for item in value] for key, value in values.items()}

    def duration(self, client_id: str, task: str, local_round: int, rng: Random) -> float:
        sequence = self.values[client_id]
        if local_round >= len(sequence):
            raise ValueError(f"No scripted duration for {client_id} round {local_round}")
        return sequence[local_round]


class CombinedDelay(DelayModel):
    def __init__(self, *models: DelayModel) -> None:
        self.models = models

    def duration(self, client_id: str, task: str, local_round: int, rng: Random) -> float:
        return sum(model.duration(client_id, task, local_round, rng) for model in self.models)


def create_delay_model(
    config: Mapping[str, Any], client_defaults: Mapping[str, float]
) -> DelayModel:
    kind = str(config.get("type", "per_client"))
    if kind == "fixed":
        return FixedDelay(float(config["value"]))
    if kind == "per_task":
        return PerTaskDelay(config["values"])
    if kind == "per_client":
        values = config.get("values") or client_defaults
        return PerClientDelay(values)
    if kind == "scripted":
        return ScriptedDelay(config["values"])
    raise ValueError(f"Unknown delay model: {kind}")
