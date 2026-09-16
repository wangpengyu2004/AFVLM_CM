"""All-visible-GPU client-parallel execution with deterministic virtual arrivals.

The parent process is the only authoritative server, scheduler, and method
instance.  Spawned workers each own one GPU and one resident LLaVA replica.
TrainPlan controls system conditions; method lifecycle hooks control which
state is downloaded, how local work behaves, and how returned updates are
interpreted.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from tqdm.auto import tqdm

from afl_vlm.config import resolved_public_config, validate_config
from afl_vlm.data.afvlm_cm import AFVLMDataModule
from afl_vlm.federation.server import FederatedServer
from afl_vlm.federation.types import ClientContext
from afl_vlm.federation.worker_pool import (
    ClientWorkerPool,
    EvaluationCompleted,
    EvaluationJob,
    FreshStateRequest,
    TrainCompleted,
    TrainJob,
    WorkerFailed,
    WorkerProgress,
    WorkerReady,
    WorkerStatus,
    update_to_device,
)
from afl_vlm.methods.registry import create_method
from afl_vlm.models.base import (
    TrainConfig,
    clone_state,
    nested_to_device,
    state_norm,
    state_to_device,
)
from afl_vlm.scheduling.train_plan import (
    build_train_plan,
    event_records,
    load_system_profile,
    load_train_plan,
    mean_staleness,
    optimizer_steps_for_epochs,
)


def _seed(seed: int, *parts: Any) -> int:
    return int.from_bytes(
        hashlib.sha256(":".join(map(str, (seed, *parts))).encode()).digest()[:4], "big"
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _append(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _build_plan(config: dict[str, Any], method: Any, profiles: list[Any]) -> list[Any]:
    federation = config["federation"]
    training = config["training"]
    if method.capabilities.requires_custom_scheduler or method.capabilities.mode == "synchronous":
        return build_train_plan(
            profiles,
            int(federation["rounds"]),
            int(training["local_epochs"]),
            int(training["batch_size"]),
            int(training["gradient_accumulation"]),
            method.capabilities.mode,
            config["method"].get("params", {}),
        )
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
        abs(event.task_compute_factor - task_costs[event.client_id]) > 1e-12
        for event in plan
    ):
        raise ValueError(
            "The persisted TrainPlan does not match the system profile task compute factors"
        )
    return plan


def _buffer_stats(method: Any, occupancies: list[int], waits: list[float]) -> dict[str, Any]:
    return {
        "buffer_aggregations": getattr(method, "buffer_aggregations", len(occupancies)),
        "mean_buffer_occupancy": getattr(method, "occupancy_sum", 0)
        / max(1, getattr(method, "arrivals", 0)),
        "mean_buffer_waiting_time": sum(waits) / len(waits) if waits else 0.0,
    }


def _resolve_runtime_devices(runtime: dict[str, Any]) -> tuple[list[int], Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("client_parallel requires at least one visible CUDA GPU")
    visible_count = torch.cuda.device_count()
    configured = runtime.get("devices", "all")
    devices = list(range(visible_count)) if configured == "all" else list(configured)
    if not devices:
        raise RuntimeError("No visible GPU is available to the client worker pool")
    if min(devices) < 0 or max(devices) >= visible_count:
        raise ValueError(
            f"runtime.devices={devices} exceeds the {visible_count} visible CUDA devices"
        )
    configured_aggregation = str(runtime.get("aggregation_device", "auto"))
    aggregation_device = (
        torch.device("cuda", devices[0])
        if configured_aggregation == "auto"
        else torch.device(configured_aggregation)
    )
    if aggregation_device.index not in devices:
        raise ValueError("The aggregation GPU must also belong to runtime.devices")
    return devices, aggregation_device


def execute_parallel(config: dict[str, Any]) -> dict[str, Any]:
    """Run a formal experiment using one process per configured GPU."""
    validate_config(config)
    method = create_method(config["method"]["name"], config["method"].get("params", {}))
    method.validate_runtime()
    import numpy as np

    runtime = config["runtime"]
    devices, aggregation_device = _resolve_runtime_devices(runtime)
    seed = int(config["run"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    progress_enabled = bool(config["output"].get("progress_bar", True))
    output = Path(str(config["output"]["directory"])).resolve()
    if (
        output.exists()
        and any(output.iterdir())
        and not config["output"].get("overwrite", False)
    ):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("events.jsonl", "updates.jsonl", "task_metrics.jsonl"):
        (output / name).write_text("", encoding="utf-8")
    started_at = time.time()
    (output / "train.log").write_text(
        f"AFVLM-CM run start\nmethod={method.name}\nseed={seed}\n"
        "runtime=client_parallel\n",
        encoding="utf-8",
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_public_config(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    data_module = AFVLMDataModule(config["dataset"])
    partitions = list(data_module.clients.values())
    profiles = load_system_profile(config["federation"]["system_profile"])
    profile_path = Path(str(config["federation"]["system_profile"])).resolve()
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
    for item in partitions:
        profile = profile_by_id[item.client_id]
        if (profile.task, profile.dataset, profile.num_samples) != (
            item.task,
            item.dataset,
            item.num_samples,
        ):
            raise ValueError(f"System metadata/data mismatch for {item.client_id}")

    plan = _build_plan(config, method, profiles)
    records = event_records(plan)
    record_by_event = {int(item["event_id"]): item for item in records}
    _write_json(output / "train_plan.json", records)
    group_sizes: dict[int, int] = defaultdict(int)
    for event in plan:
        if event.group_id is not None:
            group_sizes[event.group_id] += 1

    model_config = dict(config["model"])
    model_config["max_text_length"] = config["training"]["max_text_length"]
    model_config["progress_bar"] = False
    if progress_enabled:
        tqdm.write(
            f"[AFVLM-CM] metadata ready: {len(partitions)} clients, "
            f"{sum(item.num_samples for item in partitions)} training samples"
        )
        tqdm.write(
            f"[AFVLM-CM] starting {len(devices)} persistent LLaVA workers on GPUs {devices}"
        )
        tqdm.write(f"[AFVLM-CM] server aggregation device: {aggregation_device}")
    pool = ClientWorkerPool(
        devices=devices,
        model_config=model_config,
        dataset_config=config["dataset"],
        method_name=method.name,
        method_params=config["method"].get("params", {}),
        profiles=profiles,
        seed=seed,
        start_method=str(runtime.get("start_method", "spawn")),
    )
    job_progress: dict[str, Any] = {}
    completed_updates: dict[int, Any] = {}
    completed_evaluations: dict[str, dict[str, Any]] = {}
    logical_fresh_states: dict[int, tuple[dict[str, Any], int]] = {}
    waiting_fresh_requests: dict[int, list[FreshStateRequest]] = defaultdict(list)

    def status_hook(message: Any) -> None:
        if progress_enabled:
            if isinstance(message, WorkerReady):
                tqdm.write(
                    f"[AFVLM-CM] worker {message.worker_id} ready on GPU {message.device_id}"
                )
            else:
                tqdm.write(f"[AFVLM-CM] worker {message.worker_id}: {message.message}")

    try:
        initial_state = pool.wait_until_ready(
            status_hook,
            timeout_seconds=float(runtime.get("startup_timeout_seconds", 1800)),
        )
        initial_state = state_to_device(initial_state, aggregation_device)
        server = FederatedServer(None, method, len(partitions), initial_state=initial_state)
        method.configure_server(server.state, profiles)

        def handle_message(message: Any) -> None:
            if isinstance(message, WorkerStatus):
                status_hook(message)
                return
            if isinstance(message, WorkerProgress):
                if progress_enabled:
                    bar = job_progress.get(message.job_id)
                    if bar is None:
                        bar = tqdm(
                            total=message.total,
                            desc=f"GPU{message.worker_id} {message.client_id}",
                            unit="step",
                            dynamic_ncols=True,
                            leave=False,
                            position=message.worker_id + 1,
                        )
                        job_progress[message.job_id] = bar
                    bar.update(max(0, message.current - bar.n))
                    bar.set_postfix(loss=f"{message.loss:.4f}", refresh=False)
                return
            if isinstance(message, FreshStateRequest):
                snapshot = logical_fresh_states.get(message.event_id)
                if snapshot is None:
                    waiting_fresh_requests[message.event_id].append(message)
                else:
                    pool.respond_fresh(message, snapshot[0], snapshot[1])
                return
            if isinstance(message, TrainCompleted):
                completed_updates[message.event_id] = message.update
                job_id = f"train:{message.event_id}"
                bar = job_progress.pop(job_id, None)
                if bar is not None:
                    bar.close()
                return
            if isinstance(message, EvaluationCompleted):
                completed_evaluations[message.job_id] = message.metrics
                return
            if isinstance(message, WorkerFailed):
                raise RuntimeError(
                    f"GPU worker {message.worker_id} failed during {message.job_id}:\n"
                    f"{message.traceback_text}"
                )
            raise RuntimeError(f"Unexpected worker message: {type(message).__name__}")

        def wait_for_update(event_id: int) -> Any:
            while event_id not in completed_updates:
                handle_message(pool.receive())
            return completed_updates.pop(event_id)

        evaluation_counter = 0

        def run_evaluation(split: str) -> dict[str, Any]:
            nonlocal evaluation_counter
            evaluation_counter += 1
            job_id = f"evaluation:{evaluation_counter}:v{server.version}:{split}"
            states = (
                dict(method.evaluation_states(server.state))
                if method.evaluation_scope == "client_local_mean"
                else {"global": server.state}
            )
            worker_states = {
                key: state_to_device(state, "cpu") for key, state in states.items()
            }
            pool.submit(EvaluationJob(job_id, worker_states, split), priority=True)
            while job_id not in completed_evaluations:
                handle_message(pool.receive())
            return completed_evaluations.pop(job_id)

        contexts: dict[int, ClientContext] = {}
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
        timeline: list[tuple[float, int, str, Any]] = []
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
        timeline.sort(key=lambda row: (row[0], row[1], row[3].client_id, row[3].local_round))
        federation_progress = tqdm(
            total=len(plan),
            desc=f"{method.name} virtual arrivals",
            unit="update",
            dynamic_ncols=True,
            disable=not progress_enabled,
            position=0,
        )

        for virtual_time, _, kind, event in timeline:
            last_virtual_time = max(last_virtual_time, virtual_time)
            if kind == "start":
                base_version = server.version
                base_state = server.state
                context = ClientContext(
                    client_id=event.client_id,
                    task=event.task,
                    dataset=event.dataset,
                    num_samples=profile_by_id[event.client_id].num_samples,
                    local_round=event.local_round,
                    base_version=base_version,
                    arrival_time=event.arrival_time,
                    seed=_seed(seed, event.client_id, event.local_round),
                )
                start_state = method.prepare_download(clone_state(base_state), context)
                method_runtime_state = method.client_runtime_state(context)
                train_config = TrainConfig(
                    local_epochs=int(training["local_epochs"]),
                    batch_size=int(training["batch_size"]),
                    gradient_accumulation=int(training["gradient_accumulation"]),
                    learning_rate=float(training["learning_rate"]),
                    max_text_length=int(training["max_text_length"]),
                    seed=context.seed,
                    collect_mean_gradient=method.capabilities.requires_mean_gradient,
                    planned_optimizer_steps=event.local_steps,
                    max_local_steps=event.local_steps
                    if method.capabilities.requires_local_step_control
                    else None,
                )
                contexts[event.event_id] = context
                record_by_event[event.event_id]["base_version"] = base_version
                record_by_event[event.event_id]["download_policy"] = method.name
                pool.submit(
                    TrainJob(
                        event=event,
                        context=context,
                        base_state=state_to_device(base_state, "cpu"),
                        start_state=state_to_device(start_state, "cpu"),
                        train_config=train_config,
                        method_runtime_state=nested_to_device(method_runtime_state, "cpu"),
                        group_size=group_sizes.get(event.group_id),
                        dispatch_wall_time=time.time(),
                    )
                )
                continue
            if kind == "refresh":
                snapshot = (server.state, server.version)
                logical_fresh_states[event.event_id] = snapshot
                record_by_event[event.event_id]["fresh_global_version"] = server.version
                for request in waiting_fresh_requests.pop(event.event_id, []):
                    pool.respond_fresh(request, snapshot[0], snapshot[1])
                continue

            raw_update = wait_for_update(event.event_id)
            context = contexts.pop(event.event_id)
            raw_update = update_to_device(raw_update, aggregation_device)
            update = method.prepare_upload(raw_update, context)
            stale = server.version - update.base_version
            staleness_values.append(stale)
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
                    "worker_id": update.metadata.get("worker_id"),
                    "visible_device_id": update.metadata.get("visible_device_id"),
                    "worker_start_wall_time": update.metadata.get("worker_start_wall_time"),
                    "worker_finish_wall_time": update.metadata.get("worker_finish_wall_time"),
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
                    "base_version": update.base_version,
                    "server_version_before": receive_version,
                    "staleness": stale,
                    "server_version_after": server.version,
                    "arrival_time": event.arrival_time,
                    "accepted": any(item.applied_weight > 0 for item in results),
                    "aggregation_weight": sum(item.applied_weight for item in results),
                    "planned_local_steps": event.local_steps,
                    "assigned_local_steps": event.local_steps
                    if method.capabilities.requires_local_step_control
                    else None,
                    "optimizer_steps": update.optimizer_steps,
                    "speed_factor": event.speed_factor,
                    "task_compute_factor": event.task_compute_factor,
                    "group_id": event.group_id,
                    "worker_id": update.metadata.get("worker_id"),
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
            logical_fresh_states.pop(event.event_id, None)
            if (
                eval_interval > 0
                and server.version > 0
                and server.version % eval_interval == 0
                and server.version != last_evaluated_version
            ):
                validation_metrics = run_evaluation(periodic_split)
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
        metrics = run_evaluation(final_split)
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
        stats: dict[str, Any] = {
            "runtime_backend": "client_parallel",
            "gpu_workers": len(devices),
            "visible_device_ids": devices,
            "aggregation_device": str(aggregation_device),
            "wall_clock_seconds": time.time() - started_at,
            "mean_staleness": mean_staleness(staleness_values),
            "median_staleness": statistics.median(staleness_values)
            if staleness_values
            else None,
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
            stats.update(_buffer_stats(method, buffer_occupancies, buffer_waiting_times))
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
                    "federated_state": state_to_device(server.state, "cpu"),
                    "server_version": server.version,
                    "method": method.name,
                    "method_state": nested_to_device(method.state_dict(), "cpu"),
                    "scheduler_state": {
                        "completed_events": len(plan),
                        "virtual_time": last_virtual_time,
                        "train_plan": records,
                    },
                    "random_state": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": [],
                    },
                    "runtime": {
                        "backend": "client_parallel",
                        "devices": devices,
                        "worker_model_replicas": len(devices),
                    },
                    "metadata": {
                        "dataset": config["dataset"]["name"],
                        "clients_per_task": config["dataset"]["clients_per_task"],
                        "model_path": config["model"]["model_path"],
                    },
                },
                checkpoint_dir / "final_trainable.pt",
            )
        pool.shutdown()
        with (output / "train.log").open("a", encoding="utf-8") as handle:
            handle.write(
                f"AFVLM-CM run complete\nserver_version={server.version}\n"
                f"virtual_completion_time={last_virtual_time}\n"
                f"wall_clock_seconds={stats['wall_clock_seconds']}\n"
            )
        return {"metrics": metrics, "system": stats, "output": str(output)}
    except BaseException:
        for bar in job_progress.values():
            bar.close()
        pool.abort()
        raise
