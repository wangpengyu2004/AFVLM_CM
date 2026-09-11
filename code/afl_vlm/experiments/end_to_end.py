"""Equal-quota end-to-end asynchronous trace and task-delay shuffle control."""

from __future__ import annotations

from collections import Counter, deque
from typing import Any

from afl_vlm.evaluation.probes import evaluate_tasks
from afl_vlm.experiments.common import batch_for, probe_all, train_config_for, update_record
from afl_vlm.federation.server import FederatedServer
from afl_vlm.scheduling.delay_models import CombinedDelay, create_delay_model
from afl_vlm.scheduling.virtual_event import (
    build_schedule,
    build_sync_schedule,
    schedule_records,
    shuffled_task_delays,
)


def _capture_download(context: Any, server: FederatedServer) -> dict[str, Any]:
    state = server.state
    current = context.model.snapshot_trainable()
    context.model.load_trainable(state)
    try:
        probes = evaluate_tasks(context.model, context.tasks, context.probe_ids)
    finally:
        context.model.load_trainable(current)
    return {"state": state, "version": server.version, "probes": probes}


def run(context: Any, params: dict[str, Any], scenario: str) -> list[dict[str, Any]]:
    specs = list(context.client_specs)
    if scenario == "task_shuffled":
        specs = shuffled_task_delays(specs)
    elif scenario != "task_correlated":
        raise ValueError(f"Unknown delay scenario: {scenario}")
    defaults = {client.id: client.virtual_train_time for client in specs}
    timing = context.config["timing"]
    train_delay = create_delay_model(timing["train_delay"], defaults)
    network_delay = create_delay_model(timing["network_delay"], {key: 0.0 for key in defaults})
    delay_model = CombinedDelay(train_delay, network_delay)
    upload_quota = int(context.config["clients"]["upload_quota"])
    if context.method.name == "fedavg_sync":
        schedule = build_sync_schedule(specs, upload_quota, delay_model, context.seed)
    else:
        schedule = build_schedule(specs, upload_quota, delay_model, context.seed)
    context.writer.write_json("schedule.json", schedule_records(schedule))
    server = FederatedServer(context.model, context.method, len(context.clients))
    downloads: dict[tuple[str, int], dict[str, Any]] = {
        (client.id, 0): _capture_download(context, server) for client in specs
    }
    arrivals: deque[str] = deque(maxlen=int(context.config["evaluation"]["window_size"]))
    quota_counts: Counter[str] = Counter()
    max_fast_mass = 0.0
    for event in schedule:
        key = (event.client_id, event.local_round)
        if key not in downloads:
            downloads[key] = _capture_download(context, server)
        download = downloads[key]
        client = context.clients[event.client_id]
        update = client.train(
            context.model,
            context.method,
            download["state"],
            int(download["version"]),
            event.local_round,
            batch_for(context, client, event.local_round),
            train_config_for(
                context,
                context.derived_seed("training", event.client_id, event.local_round),
            ),
        )
        receive_version = server.version
        context.writer.append("updates.jsonl", update_record(update, server.state))
        results = server.receive(update)
        quota_counts[event.client_id] += 1
        arrivals.append(event.task)
        window_counts = dict(Counter(arrivals))
        fast_mass = window_counts.get("fast", 0) / len(arrivals)
        max_fast_mass = max(max_fast_mass, fast_mass)
        applied_weight = sum(result.applied_weight for result in results if result.applied)
        context.writer.append(
            "events.jsonl",
            {
                "event_id": event.event_id,
                "virtual_time": event.finish_time,
                "client_id": event.client_id,
                "task": event.task,
                "local_round": event.local_round,
                "download_version": update.download_version,
                "download_probe_metrics": download["probes"],
                "receive_version": receive_version,
                "staleness": receive_version - update.download_version,
                "virtual_duration": event.virtual_duration,
                "applied_weight": applied_weight,
                "recent_window_task_counts": window_counts,
                "window_mass_fast": fast_mass,
            },
        )
        probes = probe_all(context)
        context.writer.append(
            "probes.jsonl",
            {
                "event_id": event.event_id,
                "server_version": server.version,
                "virtual_time": event.finish_time,
                "task_metrics": probes,
            },
        )
        next_round = event.local_round + 1
        if next_round < upload_quota and context.method.name != "fedavg_sync":
            downloads[(event.client_id, next_round)] = _capture_download(context, server)
    tail_results = server.finish()
    expected_counts = {client.id: upload_quota for client in specs}
    if dict(quota_counts) != expected_counts:
        raise RuntimeError(
            f"Equal upload quota violated: {dict(quota_counts)} != {expected_counts}"
        )
    final_metrics = evaluate_tasks(context.model, context.tasks, context.final_ids, mode="final")
    row = {
        "experiment": "trace",
        "scenario": scenario,
        "received_updates": server.received_updates,
        "applied_client_updates": server.applied_updates,
        "server_versions": server.version,
        "optimizer_steps": sum(
            update_quota * context.train_config.local_steps
            for update_quota in quota_counts.values()
        ),
        "max_window_mass_fast": max_fast_mass,
        "tail_server_applications": len(tail_results),
        "method_extra_cost": 0.0,
        "final_task_metrics": final_metrics,
        "upload_counts": dict(quota_counts),
        "delay_multiset": sorted(defaults.values()),
    }
    return [row]
