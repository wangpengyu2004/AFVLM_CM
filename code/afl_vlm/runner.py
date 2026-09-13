"""Top-level experiment construction and execution."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from afl_vlm.config import RunSpec, expand_runs, validate_config
from afl_vlm.data.partition import partition_disjoint, select_ids
from afl_vlm.data.registry import create_task
from afl_vlm.experiments import biased_start, end_to_end, stale_twin
from afl_vlm.experiments.context import ArtifactWriter, ExperimentContext
from afl_vlm.federation.client import FederatedClient
from afl_vlm.federation.types import ClientSpec
from afl_vlm.methods.registry import create_method
from afl_vlm.models.base import TrainConfig
from afl_vlm.models.registry import create_model
from afl_vlm.scheduling.delay_models import CombinedDelay, create_delay_model
from afl_vlm.scheduling.virtual_event import (
    build_fedcompass_schedule,
    build_schedule,
    build_sync_schedule,
    schedule_records,
    shuffled_task_delays,
)


def _derived_seed(seed: int, stream: str, part: str) -> int:
    import hashlib

    material = f"{seed}:{stream}:{part}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _client_specs(config: dict[str, Any]) -> list[ClientSpec]:
    return [
        ClientSpec(
            id=str(item["id"]),
            task=str(item["task"]),
            virtual_train_time=float(item["virtual_train_time"]),
        )
        for item in config["clients"]["assignments"]
    ]


def dry_run(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    plans: list[dict[str, Any]] = []
    specs = _client_specs(config)
    for run_spec in expand_runs(config):
        plan: dict[str, Any] = {
            "experiment": run_spec.experiment_id,
            "type": run_spec.experiment_type,
            "method": run_spec.method_id,
            "seed": run_spec.seed,
            "scenario": run_spec.scenario,
        }
        if run_spec.experiment_type == "end_to_end":
            scenario_specs = (
                shuffled_task_delays(specs) if run_spec.scenario == "task_shuffled" else specs
            )
            defaults = {item.id: item.virtual_train_time for item in scenario_specs}
            train_delay = create_delay_model(config["timing"]["train_delay"], defaults)
            network_delay = create_delay_model(
                config["timing"]["network_delay"], {key: 0.0 for key in defaults}
            )
            method = create_method(run_spec.method_implementation, run_spec.method_params)
            delay_model = CombinedDelay(train_delay, network_delay)
            if method.schedule_mode == "synchronous":
                schedule = build_sync_schedule(
                    scenario_specs,
                    int(config["clients"]["upload_quota"]),
                    delay_model,
                    run_spec.seed,
                )
            elif method.schedule_mode == "fedcompass":
                schedule = build_fedcompass_schedule(
                    scenario_specs,
                    int(config["clients"]["upload_quota"]),
                    delay_model,
                    run_spec.seed,
                    int(config["training"]["local_steps"]),
                    method.min_local_steps,
                    method.max_local_steps,
                )
            else:
                schedule = build_schedule(
                    scenario_specs,
                    int(config["clients"]["upload_quota"]),
                    delay_model,
                    run_spec.seed,
                )
            plan["schedule"] = schedule_records(schedule)
        else:
            plan["controlled_paths"] = run_spec.experiment_params
        plans.append(plan)
    return {"valid": True, "model_will_load": False, "runs": plans}


def _prepare_context(
    config: dict[str, Any], run_spec: RunSpec, writer: ArtifactWriter
) -> ExperimentContext:
    tasks = {
        task_key: create_task(task_key, task_config)
        for task_key, task_config in config["tasks"].items()
    }
    specs = _client_specs(config)
    assignments = {str(item["id"]): item for item in config["clients"]["assignments"]}
    prepartitioned = config["clients"]["data_partition"] == "prepartitioned"
    clients: dict[str, FederatedClient] = {}
    heldout: dict[str, list[Any]] = {}
    probe_ids: dict[str, list[str]] = {}
    final_ids: dict[str, list[str]] = {}
    required_batch = int(config["training"]["local_steps"]) * int(
        config["training"]["grad_accumulation"]
    )
    for task_key, task in tasks.items():
        task_specs = [item for item in specs if item.task == task_key]
        if prepartitioned:
            seen_ids: set[str] = set()
            for item in task_specs:
                samples = task.load_file(str(assignments[item.id]["train_file"]), "train")
                sample_ids = {sample.id for sample in samples}
                overlap = seen_ids & sample_ids
                if overlap:
                    raise ValueError(
                        f"Prepartitioned clients overlap in task {task_key}: {sorted(overlap)[:3]}"
                    )
                seen_ids.update(sample_ids)
                clients[item.id] = FederatedClient(item.id, item.task, samples)
            heldout[task_key] = []
        else:
            train_samples = task.load_split("train")
            per_client = int(config["tasks"][task_key]["train_per_client"])
            partitions = partition_disjoint(
                train_samples,
                [item.id for item in task_specs],
                per_client,
                _derived_seed(run_spec.seed, "partition", task_key),
            )
            used_ids = {sample.id for samples in partitions.values() for sample in samples}
            heldout[task_key] = [sample for sample in train_samples if sample.id not in used_ids][
                :required_batch
            ]
            if len(heldout[task_key]) < required_batch:
                raise ValueError(
                    f"Task {task_key} needs {required_batch} held-out training samples "
                    "after client partitioning"
                )
            for item in task_specs:
                clients[item.id] = FederatedClient(item.id, item.task, partitions[item.id])
        probe_ids[task_key] = select_ids(
            task.load_split("probe"),
            int(config["evaluation"]["probe_samples_per_task"]),
            _derived_seed(run_spec.seed, "probe", task_key),
        )
        final_ids[task_key] = select_ids(
            task.load_split("final"),
            int(config["evaluation"]["final_samples_per_task"]),
            _derived_seed(run_spec.seed, "final", task_key),
        )
    # Data and image paths are checked before loading a potentially large VLM.
    model = create_model(str(config["model"]["adapter"]))
    model_config = dict(config["model"])
    model_config["max_text_length"] = int(config["training"]["max_text_length"])
    model.load(model_config)
    method = create_method(run_spec.method_implementation, run_spec.method_params)
    training = config["training"]
    train_config = TrainConfig(
        local_steps=int(training["local_steps"]),
        batch_size=int(training["batch_size"]),
        grad_accumulation=int(training["grad_accumulation"]),
        client_lr=float(training["client_lr"]),
        max_text_length=int(training["max_text_length"]),
        seed=run_spec.seed,
    )
    return ExperimentContext(
        config=config,
        model=model,
        method=method,
        tasks=tasks,
        clients=clients,
        client_specs=specs,
        probe_ids=probe_ids,
        final_ids=final_ids,
        heldout_samples=heldout,
        train_config=train_config,
        seed=run_spec.seed,
        writer=writer,
    )


def _serializable_state(state: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in state.items():
        if hasattr(value, "detach"):
            result[key] = value.detach().float().cpu().tolist()
        else:
            result[key] = value
    return result


def _run_directory(config: dict[str, Any], spec: RunSpec) -> Path:
    return (
        Path(config["run"]["output_root"])
        / spec.experiment_id
        / spec.scenario
        / spec.method_id
        / f"seed{spec.seed}"
    ).resolve()


def execute(config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_config(config)
    all_rows: list[dict[str, Any]] = []
    for spec in expand_runs(config):
        run_dir = _run_directory(config, spec)
        writer = ArtifactWriter(run_dir, bool(config["output"]["overwrite"]))
        resolved = {key: value for key, value in config.items() if not key.startswith("_")}
        resolved = json.loads(json.dumps(resolved))
        resolved["resolved_run"] = {
            "experiment": spec.experiment_id,
            "type": spec.experiment_type,
            "method": spec.method_id,
            "implementation": spec.method_implementation,
            "seed": spec.seed,
            "scenario": spec.scenario,
        }
        (run_dir / "config.resolved.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        context = _prepare_context(config, spec, writer)
        if spec.experiment_type == "stale_twin":
            rows = stale_twin.run(context, spec.experiment_params)
        elif spec.experiment_type == "biased_start":
            rows = biased_start.run(context, spec.experiment_params)
        else:
            rows = end_to_end.run(context, spec.experiment_params, spec.scenario)
        for row in rows:
            row.update({"method": spec.method_id, "seed": spec.seed, "scenario": spec.scenario})
        summary = {
            "experiment": spec.experiment_id,
            "type": spec.experiment_type,
            "method": spec.method_id,
            "seed": spec.seed,
            "scenario": spec.scenario,
            "rows": rows,
        }
        writer.write_json("summary.json", summary)
        writer.write_csv("summary.csv", rows)
        if not (run_dir / "schedule.json").exists():
            writer.write_json("schedule.json", [])
        if config["output"]["save_adapter"]:
            writer.write_json(
                "adapter_state.json", _serializable_state(context.model.snapshot_trainable())
            )
        if config["output"]["save_method_state"]:
            writer.write_json("method_state.json", context.method.state_dict())
        all_rows.extend(rows)
    root = Path(config["run"]["output_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in all_rows for key in row})
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer_csv = csv.DictWriter(handle, fieldnames=fields)
            writer_csv.writeheader()
            writer_csv.writerows(all_rows)
    (root / "summary.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return all_rows
