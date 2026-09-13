"""Single formal AFVLM-CM training entry point."""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from afl_vlm.config import resolved_public_config, validate_config
from afl_vlm.data.afvlm_cm import AFVLMDataModule
from afl_vlm.federation.client import FederatedClient
from afl_vlm.federation.server import FederatedServer
from afl_vlm.methods.registry import create_method
from afl_vlm.models.base import TrainConfig, state_norm
from afl_vlm.models.registry import create_model
from afl_vlm.scheduling.train_plan import (
    build_train_plan,
    event_records,
    load_system_profile,
    load_train_plan,
    mean_staleness,
)


def _seed(seed: int, *parts: Any) -> int:
    import hashlib

    return int.from_bytes(
        hashlib.sha256(":".join(map(str, (seed, *parts))).encode()).digest()[:4], "big"
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _append(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def evaluate_method(
    model: Any, method: Any, server_state: dict[str, Any], tasks: dict[str, Any], split: str
) -> dict[str, Any]:
    """Run the conventional primary evaluation for one server checkpoint.

    Aggregated methods always use the current server state. Local-only has no
    global model, so its client states are evaluated on their task's common
    held-out split and then macro-averaged within that task.
    """
    sample_ids = {
        task: [sample.id for sample in adapter.load_split(split)] for task, adapter in tasks.items()
    }
    original = model.snapshot_trainable()
    states = (
        method.evaluation_states(server_state)
        if method.evaluation_scope == "client_local_mean"
        else {"global": server_state}
    )
    try:
        if set(states) == {"global"}:
            model.load_trainable(states["global"])
            metrics = {}
            for task, adapter in tasks.items():
                model.set_evaluation_context(task, None)
                metrics[task] = model.evaluate(adapter, sample_ids[task], "final")
            return metrics
        by_task: dict[str, list[dict[str, float]]] = defaultdict(list)
        for client_id, state in states.items():
            task = client_id.split("/", 1)[0]
            model.load_trainable(state)
            model.set_evaluation_context(task, client_id)
            by_task[task].append(model.evaluate(tasks[task], sample_ids[task], "final"))
        return {
            task: {metric: sum(row[metric] for row in rows) / len(rows) for metric in rows[0]}
            for task, rows in by_task.items()
        }
    finally:
        model.load_trainable(original)


def execute(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    method = create_method(config["method"]["name"], config["method"].get("params", {}))
    method.validate_runtime()
    import numpy as np

    run = config["run"]
    seed = int(run["seed"])
    random.seed(seed)
    np.random.seed(seed)
    output = Path(str(config["output"]["directory"])).resolve()
    if (
        output.exists()
        and any(output.iterdir())
        and not bool(config["output"].get("overwrite", False))
    ):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("events.jsonl", "updates.jsonl", "task_metrics.jsonl"):
        (output / name).write_text("", encoding="utf-8")
    (output / "train.log").write_text(
        f"AFVLM-CM run start\nmethod={method.name}\nseed={seed}\n", encoding="utf-8"
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_public_config(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    data_module = AFVLMDataModule(config["dataset"])
    tasks = data_module.tasks
    partitions = list(data_module.clients.values())
    profiles = load_system_profile(config["federation"]["system_profile"])
    profile_path = Path(str(config["federation"]["system_profile"])).resolve()
    import hashlib

    _write_json(
        output / "system_profile.reference.json",
        {
            "path": str(profile_path),
            "sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        },
    )
    profile_by_id = {item.id: item for item in profiles}
    detected = {item.client_id for item in partitions}
    if set(profile_by_id) != detected:
        raise ValueError("System profile client IDs do not exactly match the immutable partition")
    clients = {}
    for item in partitions:
        profile = profile_by_id[item.client_id]
        if (profile.task, profile.dataset, profile.num_samples) != (
            item.task,
            item.dataset,
            item.num_samples,
        ):
            raise ValueError(f"System metadata/data mismatch for {item.client_id}")
        samples = data_module.get_client_train_data(item.client_id)
        clients[item.client_id] = FederatedClient(item.client_id, item.task, item.dataset, samples)

    model = create_model(config["model"]["adapter"])
    model_config = dict(config["model"])
    model_config["max_text_length"] = config["training"]["max_text_length"]
    model.load(model_config)
    method.configure_model(model, profiles)
    server = FederatedServer(model, method, len(clients))
    federation = config["federation"]
    if method.capabilities.requires_custom_scheduler or method.capabilities.mode == "synchronous":
        plan = build_train_plan(
            profiles,
            int(federation["rounds"]),
            int(federation["base_local_steps"]),
            method.capabilities.mode,
            config["method"].get("params", {}),
        )
    else:
        plan = load_train_plan(federation["train_plan"])
    records = event_records(plan)
    _write_json(output / "train_plan.json", records)
    record_by_event = {int(item["event_id"]): item for item in records}
    group_sizes = defaultdict(int)
    for item in plan:
        if item.group_id is not None:
            group_sizes[item.group_id] += 1
    downloads: dict[int, int] = {}
    download_states: dict[int, dict[str, Any]] = {}
    active_snapshot_references: dict[int, int] = defaultdict(int)
    timeline = []
    for event in plan:
        timeline.extend(
            ((event.start_time, 1, "start", event), (event.arrival_time, 0, "arrival", event))
        )
    timeline.sort(key=lambda item: (item[0], item[1], item[3].client_id, item[3].local_round))
    staleness_values: list[int] = []
    buffer_occupancies: list[int] = []
    buffer_waiting_times: list[float] = []
    client_updates: Counter[str] = Counter()
    task_updates: Counter[str] = Counter()
    last_virtual_time = 0.0
    last_evaluated_version = -1
    eval_interval = int(config["evaluation"].get("eval_every_server_updates", 0))
    periodic_split = str(config["evaluation"].get("periodic_split", "validation"))
    final_split = str(config["evaluation"].get("final_split", "final"))
    training = config["training"]
    for virtual_time, _, kind, event in timeline:
        last_virtual_time = max(last_virtual_time, virtual_time)
        if kind == "start":
            downloads[event.event_id] = server.version
            record_by_event[event.event_id]["base_version"] = server.version
            download_states.setdefault(server.version, server.state)
            active_snapshot_references[server.version] += 1
            continue
        base_version = downloads[event.event_id]
        base_state = download_states[base_version]
        stale = server.version - base_version
        staleness_values.append(stale)
        train_config = TrainConfig(
            local_epochs=int(training["local_epochs"]),
            batch_size=int(training["batch_size"]),
            gradient_accumulation=int(training["gradient_accumulation"]),
            learning_rate=float(training["learning_rate"]),
            max_text_length=int(training["max_text_length"]),
            seed=_seed(seed, event.client_id, event.local_round),
            max_local_steps=event.local_steps,
        )
        fresh_state = (
            server.state
            if method.capabilities.requires_fresh_global_during_local_training
            else None
        )
        update = clients[event.client_id].train(
            model,
            method,
            base_state,
            base_version,
            event.local_round,
            event.arrival_time,
            train_config,
            fresh_state,
            server.version if fresh_state is not None else None,
            event.group_id,
            group_sizes[event.group_id] if event.group_id is not None else None,
        )
        active_snapshot_references[base_version] -= 1
        if active_snapshot_references[base_version] == 0:
            del active_snapshot_references[base_version]
            del download_states[base_version]
        _append(
            output / "updates.jsonl",
            {
                "update_id": update.update_id,
                "client_id": update.client_id,
                "task": update.task,
                "dataset": update.dataset,
                "num_samples": update.num_samples,
                "base_version": update.base_version,
                "arrival_time": update.arrival_time,
                "local_delta_norm": state_norm(update.delta),
                "optimizer_steps": update.optimizer_steps,
                "sample_ids_hash": update.sample_ids_hash,
            },
        )
        receive_version = server.version
        results = server.receive(update)
        client_updates[update.client_id] += 1
        task_updates[update.task] += 1
        for result in results:
            if "buffer_occupancy" in result.metadata:
                buffer_occupancies.append(int(result.metadata["buffer_occupancy"]))
            if "mean_buffer_waiting_time" in result.metadata:
                buffer_waiting_times.append(float(result.metadata["mean_buffer_waiting_time"]))
        _append(
            output / "events.jsonl",
            {
                "event_id": event.event_id,
                "virtual_time": event.arrival_time,
                "client_id": event.client_id,
                "task": event.task,
                "dataset": event.dataset,
                "num_samples": update.num_samples,
                "base_version": base_version,
                "server_version_before": receive_version,
                "staleness": stale,
                "server_version_after": server.version,
                "arrival_time": event.arrival_time,
                "accepted": any(item.applied_weight > 0 for item in results),
                "aggregation_weight": sum(item.applied_weight for item in results),
                "assigned_local_steps": event.local_steps,
                "local_steps": update.optimizer_steps,
                "speed_factor": event.speed_factor,
                "group_id": event.group_id,
                "result_metadata": [item.metadata for item in results],
            },
        )
        if (
            eval_interval > 0
            and server.version > 0
            and server.version % eval_interval == 0
            and server.version != last_evaluated_version
        ):
            validation_metrics = evaluate_method(model, method, server.state, tasks, periodic_split)
            _append(
                output / "task_metrics.jsonl",
                {
                    "kind": "validation",
                    "protocol": method.evaluation_scope,
                    "split": periodic_split,
                    "server_version": server.version,
                    "virtual_time": event.arrival_time,
                    "per_task": validation_metrics,
                },
            )
            last_evaluated_version = server.version
    tail = server.finish()
    for result in tail:
        if "buffer_occupancy" in result.metadata:
            buffer_occupancies.append(int(result.metadata["buffer_occupancy"]))
        if "mean_buffer_waiting_time" in result.metadata:
            buffer_waiting_times.append(float(result.metadata["mean_buffer_waiting_time"]))
    _write_json(output / "train_plan.json", records)
    metrics = evaluate_method(model, method, server.state, tasks, final_split)
    _write_json(
        output / "metrics.json",
        {
            "protocol": method.evaluation_scope,
            "split": final_split,
            "server_version": server.version,
            "virtual_time": last_virtual_time,
            "per_task": metrics,
            "note": "Incompatible task metrics are not raw-averaged.",
        },
    )
    _append(
        output / "task_metrics.jsonl",
        {
            "kind": "final",
            "protocol": method.evaluation_scope,
            "split": final_split,
            "server_version": server.version,
            "virtual_time": last_virtual_time,
            "per_task": metrics,
        },
    )
    stats = {
        "mean_staleness": mean_staleness(staleness_values),
        "median_staleness": statistics.median(staleness_values) if staleness_values else None,
        "max_staleness": max(staleness_values, default=0),
        "update_count": server.received_updates,
        "accepted_update_count": server.accepted_updates,
        "server_aggregation_count": server.version,
        "server_version": server.version,
        "virtual_completion_time": last_virtual_time,
        "client_update_distribution": dict(client_updates),
        "task_update_distribution": dict(task_updates),
    }
    if method.capabilities.requires_buffer:
        stats.update(
            {
                "buffer_aggregations": getattr(
                    method, "buffer_aggregations", len(buffer_occupancies)
                ),
                "mean_buffer_occupancy": getattr(method, "occupancy_sum", 0)
                / max(1, getattr(method, "arrivals", 0)),
                "mean_buffer_waiting_time": sum(buffer_waiting_times) / len(buffer_waiting_times)
                if buffer_waiting_times
                else 0.0,
            }
        )
    if method.capabilities.requires_custom_scheduler:
        stats["local_step_allocation"] = {
            f"{item.client_id}:r{item.local_round}": item.local_steps for item in plan
        }
        stats["group_completion"] = {
            str(group): max(item.arrival_time for item in plan if item.group_id == group)
            for group in group_sizes
        }
        stats["group_completion_spread"] = {
            str(group): max(item.arrival_time for item in plan if item.group_id == group)
            - min(item.arrival_time for item in plan if item.group_id == group)
            for group in group_sizes
        }
    _write_json(output / "system_stats.json", stats)
    if bool(config["output"].get("save_checkpoint", True)):
        import torch

        checkpoint_dir = output / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "federated_state": server.state,
                "server_version": server.version,
                "method": method.name,
                "method_state": method.state_dict(),
                "scheduler_state": {
                    "completed_events": len(plan),
                    "virtual_time": last_virtual_time,
                    "train_plan": records,
                },
                "random_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                },
                "metadata": {
                    "dataset": config["dataset"]["name"],
                    "clients_per_task": config["dataset"]["clients_per_task"],
                    "model_path": config["model"]["model_path"],
                },
            },
            checkpoint_dir / "final_trainable.pt",
        )
    with (output / "train.log").open("a", encoding="utf-8") as handle:
        handle.write(
            f"AFVLM-CM run complete\nserver_version={server.version}\n"
            f"virtual_completion_time={last_virtual_time}\n"
        )
    return {"metrics": metrics, "system": stats, "output": str(output)}
