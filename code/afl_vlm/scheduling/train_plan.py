"""Deterministic system profiles and plan-then-replay virtual schedules."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from afl_vlm.federation.types import ClientSpec, ScheduledEvent


def load_system_profile(path: str | Path) -> list[ClientSpec]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    return [
        ClientSpec(
            id=str(item["client_id"]),
            task=str(item["task"]),
            dataset=str(item["dataset"]),
            num_samples=int(item["num_samples"]),
            speed_factor=float(item["speed_factor"]),
            network_delay=float(item["network_delay"]),
            initial_availability=float(item["initial_availability"]),
            base_training_cost=float(item["base_training_cost"]),
            task_compute_factor=float(item.get("task_compute_factor", 1.0)),
        )
        for item in payload["clients"]
    ]


def _duration(client: ClientSpec, steps: int) -> float:
    return client.base_training_cost * client.task_compute_factor * steps / client.speed_factor


def optimizer_steps_for_epochs(
    num_samples: int,
    local_epochs: int,
    batch_size: int,
    gradient_accumulation: int,
) -> int:
    """Return the optimizer steps implied by epoch-controlled local training."""
    values = (num_samples, local_epochs, batch_size, gradient_accumulation)
    if any(value <= 0 for value in values):
        raise ValueError("Sample count and local-training controls must be positive")
    micro_batches = math.ceil(num_samples / batch_size)
    return local_epochs * math.ceil(micro_batches / gradient_accumulation)


def optimizer_steps_for_distributed_epochs(
    num_samples: int,
    local_epochs: int,
    per_device_batch_size: int,
    gradient_accumulation: int,
    world_size: int,
) -> int:
    """Return DDP steps with a padded DistributedSampler-style partition."""
    values = (
        num_samples,
        local_epochs,
        per_device_batch_size,
        gradient_accumulation,
        world_size,
    )
    if any(value <= 0 for value in values):
        raise ValueError("Distributed local-training controls must be positive")
    samples_per_rank = math.ceil(num_samples / world_size)
    micro_batches_per_rank = math.ceil(samples_per_rank / per_device_batch_size)
    return local_epochs * math.ceil(micro_batches_per_rank / gradient_accumulation)


def build_train_plan(
    clients: Iterable[ClientSpec],
    rounds: int,
    local_epochs: int,
    batch_size: int,
    gradient_accumulation: int,
    mode: str,
    method_params: Mapping[str, Any] | None = None,
    *,
    optimizer_step_cost_scale: float = 1.0,
) -> list[ScheduledEvent]:
    """Build jobs whose ordinary workload is derived only from local epochs.

    ``local_steps`` is expected-work metadata for ordinary methods.  FedCompass
    is the sole current scheduling variant that replaces it with a bounded,
    compute-aware step allocation.

    ``optimizer_step_cost_scale`` converts an algorithm-controlled physical
    step back to the persisted system profile's nominal cost units. Reference
    DDP uses a larger record budget per step; this is a unit conversion, not a
    change to speed_factor or network_delay. Ordinary replay keeps the default.
    """
    specs = sorted(clients, key=lambda item: item.id)
    params = dict(method_params or {})
    if mode == "semi_asynchronous":
        from afl_vlm.scheduling.compass import build_compass_plan

        return build_compass_plan(specs, rounds, params, optimizer_step_cost_scale)
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if not math.isfinite(optimizer_step_cost_scale) or optimizer_step_cost_scale <= 0:
        raise ValueError("optimizer_step_cost_scale must be finite and positive")
    nominal_steps = {
        item.id: optimizer_steps_for_epochs(
            item.num_samples, local_epochs, batch_size, gradient_accumulation
        )
        for item in specs
    }
    events: list[ScheduledEvent] = []
    available = {item.id: item.initial_availability for item in specs}
    for local_round in range(rounds):
        if mode == "synchronous":
            round_start = max(available.values())
        else:
            round_start = 0.0
        provisional: list[ScheduledEvent] = []
        for client in specs:
            steps = nominal_steps[client.id]
            start = round_start if mode == "synchronous" else available[client.id]
            train_time = _duration(client, steps) * optimizer_step_cost_scale
            arrival = start + train_time + client.network_delay
            provisional.append(
                ScheduledEvent(
                    event_id=-1,
                    client_id=client.id,
                    task=client.task,
                    dataset=client.dataset,
                    local_round=local_round,
                    start_time=start,
                    arrival_time=arrival,
                    speed_factor=client.speed_factor,
                    estimated_train_time=train_time,
                    network_delay=client.network_delay,
                    local_steps=steps,
                    task_compute_factor=client.task_compute_factor,
                )
            )
        provisional.sort(key=lambda item: (item.arrival_time, item.client_id))
        for item in provisional:
            events.append(item)
            available[item.client_id] = item.arrival_time
        if mode == "synchronous":
            barrier = max(item.arrival_time for item in provisional)
            available = {item.id: barrier for item in specs}
    events.sort(key=lambda item: (item.arrival_time, item.client_id, item.local_round))
    result = []
    for index, item in enumerate(events):
        values = {name: getattr(item, name) for name in item.__dataclass_fields__}
        values["event_id"] = index
        result.append(ScheduledEvent(**values))
    return result


def event_records(events: list[ScheduledEvent]) -> list[dict[str, Any]]:
    group_sizes: dict[int, int] = {}
    for event in events:
        if event.group_id is not None:
            group_sizes[event.group_id] = group_sizes.get(event.group_id, 0) + 1
    return [
        {
            "event_id": item.event_id,
            "client_id": item.client_id,
            "task": item.task,
            "dataset": item.dataset,
            "local_round": item.local_round,
            "start_time": item.start_time,
            "arrival_time": item.arrival_time,
            "speed_factor": item.speed_factor,
            "estimated_train_time": item.estimated_train_time,
            "network_delay": item.network_delay,
            "local_steps": item.local_steps,
            "group_id": item.group_id,
            "group_size": group_sizes.get(item.group_id) if item.group_id is not None else None,
            "task_compute_factor": item.task_compute_factor,
            **({"scheduler_metadata": item.metadata} if item.metadata else {}),
        }
        for item in events
    ]


def load_train_plan(path: str | Path) -> list[ScheduledEvent]:
    payload = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    fields = set(ScheduledEvent.__dataclass_fields__)
    return [
        ScheduledEvent(
            **{
                **{key: value for key, value in row.items() if key in fields},
                "metadata": row.get("scheduler_metadata", row.get("metadata", {})),
            }
        )
        for row in payload
    ]


def mean_staleness(values: list[int]) -> float:
    return sum(values) / len(values) if values else math.nan
