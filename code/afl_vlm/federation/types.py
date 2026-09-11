"""Typed records shared by clients, servers, schedulers, and methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from afl_vlm.models.base import LoRAState


@dataclass(frozen=True, slots=True)
class ClientSpec:
    id: str
    task: str
    virtual_train_time: float


@dataclass(frozen=True, slots=True)
class ScheduledEvent:
    event_id: int
    client_id: str
    task: str
    local_round: int
    start_time: float
    finish_time: float
    virtual_duration: float


@dataclass(slots=True)
class Update:
    update_id: str
    client_id: str
    task: str
    local_round: int
    download_version: int
    base_state: LoRAState
    delta: LoRAState
    seed: int
    sample_ids_hash: str
    losses: list[float]
    optimizer_steps: int
    update_kind: str = "normal"
    pair_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ServerMutation:
    new_state: LoRAState
    applied_weight: float
    contributing_update_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ApplyResult:
    applied: bool
    applied_weight: float
    receive_version: int
    training_version: int
    resulting_version: int
    contributing_update_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClientContext:
    client_id: str
    task: str
    local_round: int
    download_version: int
    seed: int


@dataclass(frozen=True, slots=True)
class ServerContext:
    global_state: LoRAState
    version: int
    expected_clients: int
