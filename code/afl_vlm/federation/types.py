"""Typed records shared by clients, servers, schedules, and algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from afl_vlm.models.base import LoRAState


@dataclass(frozen=True, slots=True)
class ClientSpec:
    id: str
    task: str
    dataset: str
    num_samples: int
    speed_factor: float
    network_delay: float
    initial_availability: float
    base_training_cost: float


@dataclass(frozen=True, slots=True)
class ScheduledEvent:
    event_id: int
    client_id: str
    task: str
    dataset: str
    local_round: int
    start_time: float
    arrival_time: float
    speed_factor: float
    estimated_train_time: float
    network_delay: float
    local_steps: int
    group_id: int | None = None


@dataclass(slots=True)
class Update:
    update_id: str
    client_id: str
    task: str
    dataset: str
    num_samples: int
    local_round: int
    base_version: int
    arrival_time: float
    base_state: LoRAState
    local_state: LoRAState
    delta: LoRAState
    seed: int
    sample_ids_hash: str
    losses: list[float]
    optimizer_steps: int
    mean_gradient: LoRAState | None = None
    group_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ServerMutation:
    new_state: LoRAState
    applied_weight: float
    contributing_update_ids: list[str]
    increment_version: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ApplyResult:
    applied: bool
    applied_weight: float
    receive_version: int
    base_version: int
    resulting_version: int
    contributing_update_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClientContext:
    client_id: str
    task: str
    dataset: str
    num_samples: int
    local_round: int
    base_version: int
    arrival_time: float
    seed: int
    fresh_global_state: LoRAState | None = None
    fresh_global_version: int | None = None


@dataclass(frozen=True, slots=True)
class ServerContext:
    global_state: LoRAState
    version: int
    expected_clients: int


@dataclass(frozen=True, slots=True)
class MethodCapabilities:
    mode: str
    requires_task_id: bool = False
    requires_staleness: bool = False
    requires_buffer: bool = False
    requires_client_speed: bool = False
    requires_custom_scheduler: bool = False
    requires_local_step_control: bool = False
    requires_custom_adapter: bool = False
    requires_fresh_global_during_local_training: bool = False
    requires_local_state: bool = False
