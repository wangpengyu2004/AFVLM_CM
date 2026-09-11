"""Plan-then-replay virtual event scheduler."""

from __future__ import annotations

import heapq
import random
from collections.abc import Iterable

from afl_vlm.federation.types import ClientSpec, ScheduledEvent
from afl_vlm.scheduling.base import DelayModel


def build_schedule(
    clients: Iterable[ClientSpec], upload_quota: int, delay_model: DelayModel, seed: int
) -> list[ScheduledEvent]:
    specs = {client.id: client for client in clients}
    if upload_quota <= 0:
        raise ValueError("upload_quota must be positive")
    queue: list[tuple[float, str, int, float]] = []
    for client in sorted(specs.values(), key=lambda item: item.id):
        rng = random.Random(f"{seed}:timing:{client.id}:0")
        duration = delay_model.duration(client.id, client.task, 0, rng)
        if duration <= 0:
            raise ValueError("Virtual durations must be positive")
        heapq.heappush(queue, (duration, client.id, 0, 0.0))
    events: list[ScheduledEvent] = []
    while queue:
        finish_time, client_id, local_round, start_time = heapq.heappop(queue)
        spec = specs[client_id]
        events.append(
            ScheduledEvent(
                event_id=len(events),
                client_id=client_id,
                task=spec.task,
                local_round=local_round,
                start_time=start_time,
                finish_time=finish_time,
                virtual_duration=finish_time - start_time,
            )
        )
        next_round = local_round + 1
        if next_round < upload_quota:
            rng = random.Random(f"{seed}:timing:{client_id}:{next_round}")
            duration = delay_model.duration(client_id, spec.task, next_round, rng)
            if duration <= 0:
                raise ValueError("Virtual durations must be positive")
            heapq.heappush(queue, (finish_time + duration, client_id, next_round, finish_time))
    return events


def schedule_records(events: list[ScheduledEvent]) -> list[dict[str, object]]:
    return [
        {
            "event_id": event.event_id,
            "client_id": event.client_id,
            "task": event.task,
            "local_round": event.local_round,
            "start_time": event.start_time,
            "finish_time": event.finish_time,
            "virtual_duration": event.virtual_duration,
        }
        for event in events
    ]


def build_sync_schedule(
    clients: Iterable[ClientSpec], upload_quota: int, delay_model: DelayModel, seed: int
) -> list[ScheduledEvent]:
    """Barrier schedule used only by the synchronous FedAvg reference."""
    specs = sorted(clients, key=lambda item: item.id)
    events: list[ScheduledEvent] = []
    round_start = 0.0
    for local_round in range(upload_quota):
        round_events: list[ScheduledEvent] = []
        for client in specs:
            rng = random.Random(f"{seed}:timing:{client.id}:{local_round}")
            duration = delay_model.duration(client.id, client.task, local_round, rng)
            round_events.append(
                ScheduledEvent(
                    event_id=-1,
                    client_id=client.id,
                    task=client.task,
                    local_round=local_round,
                    start_time=round_start,
                    finish_time=round_start + duration,
                    virtual_duration=duration,
                )
            )
        round_events.sort(key=lambda item: (item.finish_time, item.client_id))
        for event in round_events:
            events.append(
                ScheduledEvent(
                    event_id=len(events),
                    client_id=event.client_id,
                    task=event.task,
                    local_round=event.local_round,
                    start_time=event.start_time,
                    finish_time=event.finish_time,
                    virtual_duration=event.virtual_duration,
                )
            )
        round_start = max(event.finish_time for event in round_events)
    return events


def shuffled_task_delays(clients: list[ClientSpec]) -> list[ClientSpec]:
    """Assign sorted delays round-robin across tasks while preserving the multiset."""
    tasks = sorted({client.task for client in clients})
    by_task = {
        task: sorted([c for c in clients if c.task == task], key=lambda c: c.id) for task in tasks
    }
    ordered_slots = [
        client
        for index in range(max(map(len, by_task.values())))
        for task in tasks
        for client in by_task[task][index : index + 1]
    ]
    delays = sorted(client.virtual_train_time for client in clients)
    reassigned = {client.id: delay for client, delay in zip(ordered_slots, delays, strict=True)}
    return [ClientSpec(client.id, client.task, reassigned[client.id]) for client in clients]
