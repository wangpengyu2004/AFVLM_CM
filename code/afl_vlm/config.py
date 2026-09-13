"""Single-file configuration loading, strict validation, and run expansion."""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from afl_vlm.data.fcit_preset import resolve_fcit_preset, variant_names
from afl_vlm.data.registry import backend_names
from afl_vlm.methods.registry import create_method, method_names
from afl_vlm.models.registry import model_names

TOP_LEVEL_KEYS = {
    "run",
    "model",
    "tasks",
    "clients",
    "training",
    "timing",
    "methods",
    "experiments",
    "evaluation",
    "output",
    "dataset",
}
REQUIRED_TOP_LEVEL_KEYS = TOP_LEVEL_KEYS - {"dataset"}


@dataclass(frozen=True, slots=True)
class RunSpec:
    experiment_id: str
    experiment_type: str
    experiment_params: dict[str, Any]
    method_id: str
    method_implementation: str
    method_params: dict[str, Any]
    seed: int
    scenario: str


def _check_keys(section: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration field(s) in {path}: {sorted(unknown)}")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a mapping")
    payload = resolve_fcit_preset(payload, config_path)
    payload["_config_path"] = str(config_path)
    return payload


def validate_config(config: Mapping[str, Any]) -> None:
    visible = {key for key in config if not key.startswith("_")}
    unknown = visible - TOP_LEVEL_KEYS
    missing = REQUIRED_TOP_LEVEL_KEYS - visible
    if unknown or missing:
        raise ValueError(
            f"Top-level config fields: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if "dataset" in config:
        dataset = config["dataset"]
        _check_keys(
            dataset,
            {"name", "variant", "resolved_variant", "root", "image_root", "require_images"},
            "dataset",
        )
        if dataset["name"] != "fcit_fixed":
            raise ValueError("dataset.name must be 'fcit_fixed'")
        if dataset["variant"] not in variant_names():
            raise ValueError(f"dataset.variant must be one of: {sorted(variant_names())}")
        if not isinstance(dataset.get("require_images", True), bool):
            raise ValueError("dataset.require_images must be boolean")
    run = config["run"]
    _check_keys(run, {"seeds", "output_root"}, "run")
    seeds = run["seeds"]
    if not isinstance(seeds, list) or not seeds or any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("run.seeds must be a non-empty list of integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("run.seeds must be unique")

    model = config["model"]
    _check_keys(
        model,
        {
            "adapter",
            "hf_id",
            "dtype",
            "quantization",
            "gradient_checkpointing",
            "device_map",
            "lora",
            "mock_dimension",
            "mock_step_size",
            "mm_projector_lr",
        },
        "model",
    )
    if model["adapter"] not in model_names():
        raise ValueError(f"Unknown model adapter: {model['adapter']}")
    if model["dtype"] not in {"bf16", "fp16"}:
        raise ValueError("model.dtype must be 'bf16' or 'fp16'")
    if model["quantization"] not in {"nf4", "none"}:
        raise ValueError("model.quantization must be 'nf4' or 'none'")
    lora = model["lora"]
    _check_keys(
        lora,
        {
            "r",
            "alpha",
            "dropout",
            "include_visual",
            "target_modules",
            "train_mm_projector",
        },
        "model.lora",
    )
    if int(lora["r"]) <= 0 or int(lora["alpha"]) <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0.0 <= float(lora["dropout"]) < 1.0:
        raise ValueError("model.lora.dropout must be in [0, 1)")

    tasks = config["tasks"]
    if not isinstance(tasks, dict) or len(tasks) < 2:
        raise ValueError("tasks must define at least two tasks")
    for task_key, task in tasks.items():
        _check_keys(
            task,
            {
                "backend",
                "name",
                "train_per_client",
                "data_files",
                "allow_synthetic",
                "image_root",
                "require_images",
            },
            f"tasks.{task_key}",
        )
        if task["backend"] not in backend_names():
            raise ValueError(f"Unknown backend for tasks.{task_key}: {task['backend']}")
        if int(task["train_per_client"]) <= 0:
            raise ValueError(f"tasks.{task_key}.train_per_client must be positive")
        if not isinstance(task.get("require_images", False), bool):
            raise ValueError(f"tasks.{task_key}.require_images must be boolean")
        data_files = task.get("data_files") or {}
        unknown_splits = set(data_files) - {"train", "probe", "final"}
        if unknown_splits:
            raise ValueError(
                f"Unknown data split(s) for tasks.{task_key}: {sorted(unknown_splits)}"
            )
    clients = config["clients"]
    _check_keys(
        clients,
        {"count", "upload_quota", "data_partition", "assignments"},
        "clients",
    )
    assignments = clients["assignments"]
    prepartitioned = clients["data_partition"] == "prepartitioned"

    if model["adapter"] != "tiny_mock":
        for task_key, task in tasks.items():
            if task.get("allow_synthetic", False):
                raise ValueError(f"Real model cannot use synthetic data: tasks.{task_key}")
            required = {"probe", "final"} if prepartitioned else {"train", "probe", "final"}
            if required - set(task.get("data_files") or {}):
                required_text = "/".join(sorted(required))
                raise ValueError(f"tasks.{task_key}.data_files must define {required_text}")
    if int(clients["count"]) != len(assignments):
        raise ValueError("clients.count does not match clients.assignments length")
    identifiers = [str(item["id"]) for item in assignments]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Client IDs must be unique")
    if int(clients["upload_quota"]) <= 0:
        raise ValueError("clients.upload_quota must be positive")
    if clients["data_partition"] not in {"disjoint", "prepartitioned"}:
        raise ValueError("clients.data_partition must be 'disjoint' or 'prepartitioned'")
    for item in assignments:
        _check_keys(
            item,
            {"id", "task", "virtual_train_time", "train_file"},
            f"clients.assignments[{item.get('id')}]",
        )
        if item["task"] not in tasks:
            raise ValueError(f"Client {item['id']} references unknown task {item['task']}")
        if float(item["virtual_train_time"]) <= 0:
            raise ValueError("Client virtual_train_time must be positive")
        if prepartitioned and not item.get("train_file"):
            raise ValueError("Every prepartitioned client assignment needs train_file")
        if not prepartitioned and item.get("train_file"):
            raise ValueError("train_file is only valid for prepartitioned client data")

    training = config["training"]
    _check_keys(
        training,
        {"local_steps", "batch_size", "grad_accumulation", "client_lr", "max_text_length"},
        "training",
    )
    for key in ("local_steps", "batch_size", "grad_accumulation", "max_text_length"):
        if int(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if float(training["client_lr"]) <= 0:
        raise ValueError("training.client_lr must be positive")

    timing = config["timing"]
    _check_keys(timing, {"clock", "train_delay", "network_delay"}, "timing")
    if timing["clock"] != "virtual":
        raise ValueError("Only the deterministic virtual clock is supported")
    for key in ("train_delay", "network_delay"):
        _check_keys(timing[key], {"type", "value", "values"}, f"timing.{key}")
        if timing[key]["type"] not in {"fixed", "per_task", "per_client", "scripted"}:
            raise ValueError(f"Unknown timing.{key}.type")
        values = timing[key].get("values") or {}
        if timing[key]["type"] == "per_task" and set(values) != set(tasks):
            raise ValueError(f"timing.{key}.values must define every task exactly once")
        if timing[key]["type"] in {"per_client", "scripted"} and values:
            unknown_clients = set(values) - set(identifiers)
            if unknown_clients:
                raise ValueError(
                    f"timing.{key}.values has unknown clients: {sorted(unknown_clients)}"
                )

    method_configs = config["methods"]
    method_ids = [str(item["id"]) for item in method_configs]
    if len(set(method_ids)) != len(method_ids):
        raise ValueError("Method IDs must be unique")
    for item in method_configs:
        _check_keys(
            item, {"id", "implementation", "enabled", "params"}, f"methods.{item.get('id')}"
        )
        if item["implementation"] not in method_names():
            raise ValueError(f"Unknown method implementation: {item['implementation']}")
        if item["enabled"]:
            create_method(item["implementation"], item.get("params") or {})

    experiment_ids: set[str] = set()
    for experiment in config["experiments"]:
        _check_keys(experiment, {"id", "type", "enabled", "methods", "params"}, "experiments[]")
        if experiment["id"] in experiment_ids:
            raise ValueError(f"Duplicate experiment ID: {experiment['id']}")
        experiment_ids.add(experiment["id"])
        if experiment["type"] not in {"stale_twin", "biased_start", "end_to_end"}:
            raise ValueError(f"Unknown experiment type: {experiment['type']}")
        allowed_params = {
            "stale_twin": {"tau", "paths"},
            "biased_start": {"biased_path", "balanced_path"},
            "end_to_end": {"delay_scenarios"},
        }[experiment["type"]]
        _check_keys(
            experiment.get("params") or {}, allowed_params, f"experiments.{experiment['id']}.params"
        )
        selected = method_ids if experiment["methods"] == "all_enabled" else experiment["methods"]
        unknown_methods = set(selected) - set(method_ids)
        if unknown_methods:
            raise ValueError(f"Experiment references unknown methods: {sorted(unknown_methods)}")
        if experiment["enabled"] and experiment["type"] in {"stale_twin", "biased_start"}:
            if {"fast", "slow"} - set(tasks):
                raise ValueError(
                    f"Experiment {experiment['id']} requires tasks named 'fast' and 'slow'"
                )
            if prepartitioned:
                raise ValueError(
                    f"Experiment {experiment['id']} does not support prepartitioned client data"
                )
            by_id = {item["id"]: item for item in method_configs}
            for method_id in selected:
                method_config = by_id[method_id]
                if method_config["enabled"]:
                    method = create_method(
                        method_config["implementation"], method_config.get("params") or {}
                    )
                    if not method.branch_compatible:
                        raise ValueError(
                            f"Experiment {experiment['id']} is incompatible with method {method_id}"
                        )

    evaluation = config["evaluation"]
    _check_keys(
        evaluation,
        {
            "probe_samples_per_task",
            "final_samples_per_task",
            "window_size",
            "save_update_norm",
            "probe_on_selected_events",
            "generation_on_selected_checkpoints",
        },
        "evaluation",
    )
    for key in ("probe_samples_per_task", "final_samples_per_task", "window_size"):
        if int(evaluation[key]) <= 0:
            raise ValueError(f"evaluation.{key} must be positive")
    _check_keys(config["output"], {"save_adapter", "save_method_state", "overwrite"}, "output")
    task_counts = Counter(item["task"] for item in assignments)
    missing_client_tasks = [task_key for task_key in tasks if task_counts[task_key] == 0]
    if missing_client_tasks:
        raise ValueError(f"Every task needs at least one client: {missing_client_tasks}")


def expand_runs(config: Mapping[str, Any]) -> list[RunSpec]:
    enabled_methods = {item["id"]: item for item in config["methods"] if item["enabled"]}
    runs: list[RunSpec] = []
    for experiment in config["experiments"]:
        if not experiment["enabled"]:
            continue
        selected = (
            list(enabled_methods)
            if experiment["methods"] == "all_enabled"
            else [item for item in experiment["methods"] if item in enabled_methods]
        )
        scenarios = (
            list(experiment.get("params", {}).get("delay_scenarios", ["task_correlated"]))
            if experiment["type"] == "end_to_end"
            else ["controlled"]
        )
        for method_id in selected:
            method = enabled_methods[method_id]
            for seed in config["run"]["seeds"]:
                for scenario in scenarios:
                    runs.append(
                        RunSpec(
                            experiment_id=str(experiment["id"]),
                            experiment_type=str(experiment["type"]),
                            experiment_params=copy.deepcopy(dict(experiment.get("params") or {})),
                            method_id=str(method_id),
                            method_implementation=str(method["implementation"]),
                            method_params=copy.deepcopy(dict(method.get("params") or {})),
                            seed=int(seed),
                            scenario=str(scenario),
                        )
                    )
    if not runs:
        raise ValueError("Configuration expands to zero runs")
    return runs
