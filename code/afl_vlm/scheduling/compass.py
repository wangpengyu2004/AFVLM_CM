"""Virtual-clock port of the author FedCompass compute-aware scheduler.

Reference: APPFL/FedCompass af96db2204795b0b0addfdec219b2a541f4b1ab3,
CompassScheduler._record_info/_join_group/_create_group/_group_aggregation.
No round barrier: jobs are assigned on completion/group release. Timers are
virtual events, never wall-clock sleeps. Network delay is included in measured
seconds/step, as in the source. Intrinsic profiles are not regenerated.

The replay Plan compiles each aggregation packet at its release time. Metadata
keeps the actual communication arrival and original scheduling group, distinct
from the aggregation packet ID. This represents timed buffering without
introducing artificial client uploads. End-of-budget residual flushing is a
disclosed finite-budget extension. Late updates are queued rather than
overwritten by the author's per-client dictionary.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import replace
from typing import Any

from afl_vlm.federation.types import ClientSpec, ScheduledEvent


def build_compass_plan(
    clients: list[ClientSpec],
    rounds: int,
    params: dict[str, Any],
    cost_scale: float,
) -> list[ScheduledEvent]:
    minimum = int(params.get("min_local_steps", 20))
    maximum = int(params.get("max_local_steps", 100))
    momentum = float(params.get("speed_momentum", 0.9))
    deadline_factor = float(params.get("latest_time_factor", 1.5))
    if not clients or rounds <= 0 or not 0 < minimum <= maximum:
        raise ValueError("FedCompass requires clients, positive rounds, and 0 < min <= max")
    if not 0 <= momentum <= 1 or not math.isfinite(deadline_factor) or deadline_factor < 1:
        raise ValueError("Invalid FedCompass speed_momentum/latest_time_factor")
    if not math.isfinite(cost_scale) or cost_scale <= 0:
        raise ValueError("FedCompass optimizer-step cost scale must be positive")
    specs = {c.id: c for c in clients}
    if len(specs) != len(clients):
        raise ValueError("Duplicate FedCompass client ID")
    for c in clients:
        if not all(
            math.isfinite(value)
            for value in (
                c.speed_factor,
                c.base_training_cost,
                c.task_compute_factor,
                c.network_delay,
                c.initial_availability,
            )
        ):
            raise ValueError("FedCompass client properties must be finite")
        if min(c.speed_factor, c.base_training_cost, c.task_compute_factor) <= 0:
            raise ValueError("FedCompass client costs/speeds must be positive")
        if c.network_delay < 0 or c.initial_availability < 0:
            raise ValueError("FedCompass network delay/availability must be nonnegative")
    info: dict[str, dict[str, Any]] = {}
    groups: dict[int, dict[str, Any]] = {}
    general: list[ScheduledEvent] = []
    queue: list[tuple[float, int, int, Any]] = []
    result: list[ScheduledEvent] = []
    group_counter = packet_counter = counter = 0
    last_release = -math.inf

    def enqueue(when: float, kind: int, payload: Any) -> None:
        nonlocal counter
        counter += 1
        heapq.heappush(queue, (when, kind, counter, payload))

    def launch(cid: str, local_round: int, now: float, steps: int, gid: int | None) -> None:
        spec = specs[cid]
        duration = (
            spec.base_training_cost
            * spec.task_compute_factor
            * steps
            * cost_scale
            / spec.speed_factor
        )
        arrival = now + duration + spec.network_delay
        job = ScheduledEvent(
            -1,
            cid,
            spec.task,
            spec.dataset,
            local_round,
            now,
            arrival,
            spec.speed_factor,
            duration,
            spec.network_delay,
            steps,
            gid,
            spec.task_compute_factor,
        )
        enqueue(arrival, 0, job)  # completion precedes an exactly coincident timer

    def assign(cid: str, local_round: int, now: float) -> None:
        nonlocal group_counter
        if local_round >= rounds:
            return
        speed = info[cid]["speed"]
        chosen, steps = None, -1
        for gid, group in groups.items():
            candidate = math.floor((group["expected"] - now) / speed)
            if minimum <= candidate <= maximum and candidate >= steps:
                chosen, steps = gid, candidate
        if chosen is None:
            steps = -1
            for group in groups.values():
                if now < group["deadline"]:
                    owners = list(group["waiting"]) + [j.client_id for j in group["arrived"]]
                    fastest = min(info[owner]["speed"] for owner in owners)
                    candidate = math.floor((group["deadline"] + fastest * maximum - now) / speed)
                    if candidate <= maximum:
                        steps = max(steps, candidate)
            steps = maximum if steps < 0 else max(minimum, steps)
            chosen = group_counter
            group_counter += 1
            groups[chosen] = {
                "waiting": set(),
                "arrived": [],
                "expected": now + steps * speed,
                "deadline": now + steps * speed * deadline_factor,
            }
            enqueue(groups[chosen]["deadline"], 1, chosen)
        groups[chosen]["waiting"].add(cid)
        launch(cid, local_round, now, steps, chosen)

    def emit(jobs: list[ScheduledEvent], now: float, reason: str) -> float:
        nonlocal last_release, packet_counter
        release = max(now, math.nextafter(last_release, math.inf))
        last_release = release
        for job in jobs:
            result.append(
                replace(
                    job,
                    arrival_time=release,
                    group_id=packet_counter,
                    metadata={
                        "scheduling_group_id": job.group_id,
                        "communication_arrival_time": job.arrival_time,
                        "aggregation_release_time": release,
                        "aggregation_reason": reason,
                    },
                )
            )
        packet_counter += 1
        return release

    def flush(gid: int, now: float, reason: str) -> None:
        group = groups.get(gid)
        if group is None or not group["arrived"]:
            return
        arrived = list(group["arrived"])
        release = emit([*general, *arrived], now, reason)
        general.clear()
        group["arrived"] = []
        group["expected"] = group["deadline"] = 0.0
        for job in sorted(arrived, key=lambda j: (info[j.client_id]["speed"], j.client_id)):
            assign(job.client_id, job.local_round + 1, math.nextafter(release, math.inf))
        if not group["waiting"]:
            groups.pop(gid, None)

    for spec in sorted(clients, key=lambda c: c.id):
        launch(spec.id, 0, spec.initial_availability, maximum, None)
    last_time = 0.0
    while queue:
        now, kind, _, payload = heapq.heappop(queue)
        last_time = max(last_time, now)
        if kind == 1:
            flush(payload, now, "group_deadline")
            continue
        job = payload
        measured = (job.arrival_time - job.start_time) / job.local_steps
        previous = info.get(job.client_id)
        info[job.client_id] = {
            "speed": measured
            if previous is None
            else (1 - momentum) * previous["speed"] + momentum * measured
        }
        if job.group_id is None:
            release = emit([job], now, "initial_single_update")
            assign(job.client_id, job.local_round + 1, math.nextafter(release, math.inf))
            continue
        group = groups.get(job.group_id)
        if group is None:
            raise RuntimeError("FedCompass completion references a missing scheduling group")
        group["waiting"].remove(job.client_id)
        if now > group["deadline"]:
            # Preserve every late update; a per-client dictionary can overwrite.
            general.append(job)
            if not group["waiting"] and not group["arrived"]:
                groups.pop(job.group_id)
            assign(job.client_id, job.local_round + 1, now)
        else:
            group["arrived"].append(job)
            if not group["waiting"]:
                flush(job.group_id, now, "group_complete")
    if general:
        emit(list(general), last_time, "finite_budget_residual_flush")
    if len(result) != len(clients) * rounds:
        raise RuntimeError("FedCompass Plan lost or duplicated a local job")
    result.sort(key=lambda e: (e.arrival_time, e.client_id, e.local_round))
    return [replace(event, event_id=index) for index, event in enumerate(result)]
