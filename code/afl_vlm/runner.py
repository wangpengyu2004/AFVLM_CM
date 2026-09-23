"""Single formal AFVLM-CM training entry point."""

from __future__ import annotations

import json
import os
import random
import statistics
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import yaml
from tqdm.auto import tqdm

from afl_vlm.config import resolved_public_config, validate_config
from afl_vlm.data.afvlm_cm import AFVLMDataModule
from afl_vlm.evaluation import merge_evaluation_outputs
from afl_vlm.federation.client import FederatedClient
from afl_vlm.federation.server import FederatedServer
from afl_vlm.federation.types import ClientContext
from afl_vlm.methods.registry import create_method
from afl_vlm.models.base import TrainConfig, clone_state, state_norm
from afl_vlm.models.registry import create_model
from afl_vlm.scheduling.train_plan import (
    build_train_plan,
    event_records,
    load_system_profile,
    load_train_plan,
    mean_staleness,
    optimizer_steps_for_distributed_epochs,
    optimizer_steps_for_epochs,
)


def _seed(seed: int, *parts: Any) -> int:
    import hashlib

    return int.from_bytes(
        hashlib.sha256(":".join(map(str, (seed, *parts))).encode()).digest()[:4], "big"
    )


def _write_json(path: Path, payload: Any) -> None:
    if os.environ.get("AFVLM_PRIMARY_PROCESS", "1") != "1":
        return
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _append(path: Path, payload: Any) -> None:
    if os.environ.get("AFVLM_PRIMARY_PROCESS", "1") != "1":
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _evaluate_method_primary(
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


def _evaluate_distributed_task(
    model: Any,
    task_adapter: Any,
    sample_ids: list[str],
) -> dict[str, float]:
    """Generate one balanced shard per DDP rank and compute exact corpus metrics."""

    import torch

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    indexed_ids = list(enumerate(sample_ids))
    local_items = indexed_ids[rank::world_size]
    if rank == 0:
        tqdm.write(
            f"[AFVLM-CM] DDP evaluation start: task={task_adapter.task_key}, "
            f"samples={len(sample_ids)}, ranks={world_size}, "
            f"rank0_shard={len(local_items)}"
        )
    try:
        predictions, references = model.generate_evaluation_outputs(
            task_adapter,
            [sample_id for _, sample_id in local_items],
            progress_label=(
                f"evaluate {task_adapter.task_key} rank {rank + 1}/{world_size} shard"
            ),
        )
        if len(predictions) != len(local_items) or len(references) != len(local_items):
            raise RuntimeError("Model returned incomplete distributed evaluation outputs")
        local_rows = [
            (index, prediction, reference)
            for (index, _), prediction, reference in zip(
                local_items, predictions, references, strict=True
            )
        ]
        local_payload: dict[str, Any] = {"rank": rank, "rows": local_rows, "error": None}
    except Exception as exc:
        local_payload = {
            "rank": rank,
            "rows": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    gathered: list[dict[str, Any] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered, local_payload)
    complete = [payload for payload in gathered if payload is not None]
    if len(complete) != world_size:
        raise RuntimeError("Distributed evaluation did not receive every rank shard")
    errors = [
        f"rank {payload['rank']}: {payload['error']}"
        for payload in complete
        if payload["error"] is not None
    ]
    if errors:
        raise RuntimeError("Distributed evaluation shard failed: " + "; ".join(errors))
    ordered_predictions, ordered_references = merge_evaluation_outputs(
        [payload["rows"] for payload in complete], len(sample_ids)
    )
    metrics = task_adapter.metric(ordered_predictions, ordered_references)
    if rank == 0:
        tqdm.write(
            f"[AFVLM-CM] DDP evaluation complete: task={task_adapter.task_key}, "
            f"samples={len(sample_ids)}, ranks={world_size}"
        )
    return metrics


def _evaluate_method_distributed(
    model: Any,
    method: Any,
    server_state: dict[str, Any],
    tasks: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    """Evaluate every model state cooperatively on all DDP ranks."""

    sample_ids = {
        task: [sample.id for sample in adapter.load_split(split)]
        for task, adapter in tasks.items()
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
            for task in sorted(tasks):
                model.set_evaluation_context(task, None)
                metrics[task] = _evaluate_distributed_task(
                    model, tasks[task], sample_ids[task]
                )
            return metrics
        by_task: dict[str, list[dict[str, float]]] = defaultdict(list)
        for client_id in sorted(states):
            task = client_id.split("/", 1)[0]
            model.load_trainable(states[client_id])
            model.set_evaluation_context(task, client_id)
            by_task[task].append(
                _evaluate_distributed_task(model, tasks[task], sample_ids[task])
            )
        return {
            task: {metric: sum(row[metric] for row in rows) / len(rows) for metric in rows[0]}
            for task, rows in by_task.items()
        }
    finally:
        model.load_trainable(original)


def evaluate_method(
    model: Any, method: Any, server_state: dict[str, Any], tasks: dict[str, Any], split: str
) -> dict[str, Any]:
    """Use all DDP ranks for evaluation without long idle collectives."""
    import torch

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not distributed:
        return _evaluate_method_primary(model, method, server_state, tasks, split)
    return _evaluate_method_distributed(model, method, server_state, tasks, split)


def _initialize_server(
    model: Any,
    method: Any,
    profiles: list[Any],
    client_count: int,
    *,
    distributed: bool,
    local_rank: int,
) -> FederatedServer:
    """Install extensions, synchronize DDP parameters, then snapshot server state."""

    method.configure_model(model, profiles)
    if distributed:
        model.enable_distributed_data_parallel(local_rank)
    server = FederatedServer(model, method, client_count)
    method.configure_server(server.state, profiles)
    return server


def execute(config: dict[str, Any]) -> dict[str, Any]:
    """Select the configured execution backend without changing method semantics."""
    backend = str(config.get("runtime", {}).get("backend", "serial"))
    if backend == "client_parallel":
        from afl_vlm.parallel_runner import execute_parallel

        return execute_parallel(config)
    return execute_serial(config)


def execute_serial(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    method = create_method(config["method"]["name"], config["method"].get("params", {}))
    method.validate_runtime()
    import numpy as np
    import torch

    distributed = str(config.get("runtime", {}).get("backend")) == "client_ddp"
    rank, local_rank, world_size = 0, 0, 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("client_ddp requires CUDA")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        ddp_timeout = float(config.get("runtime", {}).get("ddp_timeout_seconds", 7200))
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=ddp_timeout),
        )
    primary = rank == 0
    os.environ["AFVLM_PRIMARY_PROCESS"] = "1" if primary else "0"

    run = config["run"]
    seed = int(run["seed"])
    progress_enabled = bool(config["output"].get("progress_bar", True)) and primary
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    output = Path(str(config["output"]["directory"])).resolve()
    if primary and (
        output.exists()
        and any(output.iterdir())
        and not bool(config["output"].get("overwrite", False))
    ):
        raise FileExistsError(f"Output directory is not empty: {output}")
    data_module = AFVLMDataModule(config["dataset"])
    if primary:
        output.mkdir(parents=True, exist_ok=True)
        for name in ("events.jsonl", "updates.jsonl", "task_metrics.jsonl"):
            (output / name).write_text("", encoding="utf-8")
        (output / "train.log").write_text(
            f"AFVLM-CM run start\nmethod={method.name}\nseed={seed}\n"
            f"runtime={'client_ddp' if distributed else 'serial'}\n"
            f"world_size={world_size}\n",
            encoding="utf-8",
        )
        (output / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved_public_config(config), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    if distributed:
        torch.distributed.barrier()

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

    if progress_enabled:
        tqdm.write(
            f"[AFVLM-CM] data ready: {len(clients)} clients, "
            f"{sum(len(client.samples) for client in clients.values())} training samples"
        )
        tqdm.write(f"[AFVLM-CM] loading model adapter: {config['model']['adapter']}")
    model = create_model(config["model"]["adapter"])
    model_config = dict(config["model"])
    model_config["max_text_length"] = config["training"]["max_text_length"]
    model_config["progress_bar"] = progress_enabled
    if distributed:
        model_config["device_map"] = {"": local_rank}
    model.load(model_config)
    if progress_enabled:
        tqdm.write("[AFVLM-CM] model ready; starting federated event replay")
    # DDP's constructor synchronizes rank 0 parameters. Create the authoritative
    # federated snapshot only after that synchronization; otherwise independently
    # initialized LoRA parameters could leave nonzero ranks with stale server state.
    server = _initialize_server(
        model,
        method,
        profiles,
        len(clients),
        distributed=distributed,
        local_rank=local_rank,
    )
    federation = config["federation"]
    training = config["training"]
    if method.capabilities.requires_custom_scheduler or method.capabilities.mode == "synchronous":
        plan = build_train_plan(
            profiles,
            int(federation["rounds"]),
            int(training["local_epochs"]),
            int(training["batch_size"]),
            int(training["gradient_accumulation"]),
            method.capabilities.mode,
            config["method"].get("params", {}),
        )
    else:
        plan = load_train_plan(federation["train_plan"])
        expected_steps = {
            item.id: optimizer_steps_for_epochs(
                item.num_samples,
                int(training["local_epochs"]),
                int(training["batch_size"]),
                int(training["gradient_accumulation"]),
            )
            for item in profiles
        }
        if any(event.local_steps != expected_steps[event.client_id] for event in plan):
            raise ValueError(
                "The persisted TrainPlan does not match training.local_epochs, batch_size, "
                "and gradient_accumulation; regenerate it with tools/generate_system_profiles.py"
            )
        task_costs = {item.id: item.task_compute_factor for item in profiles}
        if any(
            abs(event.task_compute_factor - task_costs[event.client_id]) > 1e-12 for event in plan
        ):
            raise ValueError(
                "The persisted TrainPlan does not match the system profile task compute factors"
            )
    records = event_records(plan)
    _write_json(output / "train_plan.json", records)
    record_by_event = {int(item["event_id"]): item for item in records}
    group_sizes = defaultdict(int)
    for item in plan:
        if item.group_id is not None:
            group_sizes[item.group_id] += 1
    job_contexts: dict[int, ClientContext] = {}
    job_base_states: dict[int, dict[str, Any]] = {}
    job_start_states: dict[int, dict[str, Any]] = {}
    job_method_states: dict[int, dict[str, Any]] = {}
    logical_fresh_states: dict[int, tuple[dict[str, Any], int]] = {}
    timeline = []
    for event in plan:
        timeline.append((event.start_time, 2, "start", event))
        if method.capabilities.requires_fresh_global_during_local_training:
            refresh_fraction = float(
                config["method"].get("params", {}).get("refresh_fraction", 0.5)
            )
            refresh_time = event.start_time + refresh_fraction * event.estimated_train_time
            timeline.append((refresh_time, 1, "refresh", event))
            record_by_event[event.event_id]["fresh_request_time"] = refresh_time
        timeline.append((event.arrival_time, 0, "arrival", event))
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
    federation_progress = tqdm(
        total=len(plan),
        desc=f"{method.name} client updates",
        unit="update",
        dynamic_ncols=True,
        mininterval=0.0,
        miniters=1,
        smoothing=0.0,
        disable=not progress_enabled,
    )
    if progress_enabled:
        tqdm.write(
            "[AFVLM-CM] periodic evaluation schedule: "
            f"every {eval_interval} accepted server updates"
        )
    for virtual_time, _, kind, event in timeline:
        last_virtual_time = max(last_virtual_time, virtual_time)
        if kind == "start":
            base_version = server.version
            context = ClientContext(
                client_id=event.client_id,
                task=event.task,
                dataset=event.dataset,
                num_samples=len(clients[event.client_id].samples),
                local_round=event.local_round,
                base_version=base_version,
                arrival_time=event.arrival_time,
                seed=_seed(seed, event.client_id, event.local_round),
            )
            base_state = server.state
            job_contexts[event.event_id] = context
            job_base_states[event.event_id] = base_state
            job_start_states[event.event_id] = method.prepare_download(
                clone_state(base_state), context
            )
            job_method_states[event.event_id] = method.client_runtime_state(context)
            record_by_event[event.event_id]["base_version"] = server.version
            record_by_event[event.event_id]["download_policy"] = method.name
            continue
        if kind == "refresh":
            logical_fresh_states[event.event_id] = (server.state, server.version)
            record_by_event[event.event_id]["fresh_global_version"] = server.version
            continue
        context = job_contexts.pop(event.event_id)
        base_version = context.base_version
        base_state = job_base_states.pop(event.event_id)
        start_state = job_start_states.pop(event.event_id)
        stale = server.version - base_version
        staleness_values.append(stale)
        physical_optimizer_steps = (
            optimizer_steps_for_distributed_epochs(
                len(clients[event.client_id].samples),
                int(training["local_epochs"]),
                int(training["batch_size"]),
                int(training["gradient_accumulation"]),
                world_size,
            )
            if distributed
            else event.local_steps
        )
        record_by_event[event.event_id]["physical_optimizer_steps"] = physical_optimizer_steps
        record_by_event[event.event_id]["physical_executor"] = (
            "client_ddp" if distributed else "serial"
        )
        train_config = TrainConfig(
            local_epochs=int(training["local_epochs"]),
            batch_size=int(training["batch_size"]),
            gradient_accumulation=int(training["gradient_accumulation"]),
            learning_rate=float(training["learning_rate"]),
            max_text_length=int(training["max_text_length"]),
            seed=context.seed,
            collect_mean_gradient=method.capabilities.requires_mean_gradient,
            planned_optimizer_steps=physical_optimizer_steps,
            max_local_steps=event.local_steps
            if method.capabilities.requires_local_step_control
            else None,
            distributed_rank=rank,
            distributed_world_size=world_size,
        )
        client_method = create_method(method.name, method.params)
        client_method.load_client_runtime_state(job_method_states.pop(event.event_id), context)
        fresh_snapshot = logical_fresh_states.pop(event.event_id, None)
        update = clients[event.client_id].train_prepared(
            model=model,
            method=client_method,
            global_state=base_state,
            start_state=start_state,
            context=context,
            train_config=train_config,
            group_id=event.group_id,
            group_size=group_sizes.get(event.group_id),
            fresh_state_provider=(lambda snapshot=fresh_snapshot: snapshot)
            if fresh_snapshot is not None
            else None,
        )
        update = method.prepare_upload(update, context)
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
                "planned_local_steps": event.local_steps,
                "assigned_local_steps": physical_optimizer_steps
                if distributed or method.capabilities.requires_local_step_control
                else None,
                "optimizer_steps": update.optimizer_steps,
                "speed_factor": event.speed_factor,
                "task_compute_factor": event.task_compute_factor,
                "group_id": event.group_id,
                "result_metadata": [item.metadata for item in results],
            },
        )
        federation_progress.set_postfix(
            server_version=server.version,
            staleness=stale,
            client=event.client_id,
            refresh=False,
        )
        federation_progress.update(1)
        if (
            eval_interval > 0
            and server.version > 0
            and server.version % eval_interval == 0
            and server.version != last_evaluated_version
        ):
            if progress_enabled:
                federation_progress.refresh()
                tqdm.write(
                    "[AFVLM-CM] starting periodic evaluation: "
                    f"server_version={server.version}, interval={eval_interval}, "
                    f"split={periodic_split}"
                )
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
    federation_progress.close()
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
        "runtime_backend": "client_ddp" if distributed else "serial",
        "ddp_world_size": world_size,
        "per_device_batch_size": int(training["batch_size"]),
        "effective_batch_size": (
            int(training["batch_size"]) * int(training["gradient_accumulation"]) * world_size
        ),
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
    if primary and bool(config["output"].get("save_checkpoint", True)):
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
                    "runtime_backend": "client_ddp" if distributed else "serial",
                    "ddp_world_size": world_size,
                },
            },
            checkpoint_dir / "final_trainable.pt",
        )
    if primary:
        with (output / "train.log").open("a", encoding="utf-8") as handle:
            handle.write(
                f"AFVLM-CM run complete\nserver_version={server.version}\n"
                f"virtual_completion_time={last_virtual_time}\n"
            )
    result = {"metrics": metrics, "system": stats, "output": str(output)}
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return result
