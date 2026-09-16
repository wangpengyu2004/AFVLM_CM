"""Hierarchical YAML configuration loading and static validation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader variant that fails on duplicate mapping keys."""


def _construct_mapping(loader: UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Configuration inheritance cycle: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    parents = payload.pop("inherits", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list):
        raise ValueError(f"inherits must be a string or list: {path}")
    merged: dict[str, Any] = {}
    for parent in parents:
        merged = _merge(merged, _read((path.parent / str(parent)).resolve(), (*stack, path)))
    return _merge(merged, payload)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = _read(config_path, ())
    config["_config_path"] = str(config_path)
    return config


def validate_config(config: Mapping[str, Any], *, check_paths: bool = True) -> None:
    """Validate one resolved formal experiment without loading LLaVA."""
    from afl_vlm.methods.registry import create_method

    required = {
        "run",
        "model",
        "dataset",
        "training",
        "federation",
        "method",
        "evaluation",
        "output",
    }
    visible = {key for key in config if not key.startswith("_")}
    missing = required - visible
    if missing:
        raise ValueError(f"Missing top-level configuration sections: {sorted(missing)}")
    model = config["model"]
    if model.get("adapter") != "llava15":
        raise ValueError("Formal AFVLM-CM experiments support only model.adapter=llava15")
    if not bool(model.get("local_files_only", False)):
        raise ValueError("Training must set model.local_files_only=true")
    lora = model.get("lora", {})
    if not bool(lora.get("enabled", False)):
        raise ValueError("Formal AFVLM-CM experiments require LoRA enabled")
    if int(lora.get("r", 0)) <= 0 or int(lora.get("alpha", 0)) <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if lora.get("bias") != "none":
        raise ValueError("The default federated LoRA scope requires bias=none")
    dataset = config["dataset"]
    if dataset.get("name") != "afvlm_cm":
        raise ValueError("dataset.name must be afvlm_cm")
    if int(dataset.get("clients_per_task", 0)) not in {2, 5, 10}:
        raise ValueError("dataset.clients_per_task must be one of 2, 5, 10")
    tasks = tuple(dataset.get("tasks", ()))
    expected = ("cls", "caption", "vqa", "chart_vqa", "visual_reasoning", "grounding")
    if tasks != expected:
        raise ValueError(f"dataset.tasks must preserve the canonical order {expected}")
    if check_paths:
        partition_root = Path(str(dataset["partition_root"]))
        if not partition_root.is_absolute():
            partition_root = Path.cwd() / partition_root
        if not partition_root.is_dir():
            raise FileNotFoundError(f"AFVLM-CM partition not found: {partition_root.resolve()}")
        profile = Path(str(config["federation"]["system_profile"]))
        if not profile.is_absolute():
            profile = Path.cwd() / profile
        if not profile.is_file():
            raise FileNotFoundError(f"Shared system profile not found: {profile.resolve()}")
        plan = Path(str(config["federation"]["train_plan"]))
        if not plan.is_absolute():
            plan = Path.cwd() / plan
        if not plan.is_file():
            raise FileNotFoundError(f"Shared asynchronous TrainPlan not found: {plan.resolve()}")
    training = config["training"]
    for key in ("local_epochs", "batch_size", "gradient_accumulation", "max_text_length"):
        if int(training.get(key, 0)) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if float(training.get("learning_rate", 0.0)) <= 0:
        raise ValueError("training.learning_rate must be positive")
    if int(config["federation"].get("rounds", 0)) <= 0:
        raise ValueError("federation.rounds must be positive")
    task_costs = config["federation"].get("task_compute_factors", {})
    if task_costs and set(task_costs) != set(expected):
        raise ValueError("federation.task_compute_factors must contain all six task IDs")
    if any(float(value) <= 0 for value in task_costs.values()):
        raise ValueError("federation.task_compute_factors values must be positive")
    evaluation = config["evaluation"]
    if evaluation.get("protocol") != "server_global":
        raise ValueError("evaluation.protocol must be server_global")
    if evaluation.get("interval_unit") != "server_updates":
        raise ValueError("evaluation.interval_unit must be server_updates")
    if int(evaluation.get("eval_every_server_updates", 0)) <= 0:
        raise ValueError("evaluation.eval_every_server_updates must be positive")
    if evaluation.get("periodic_split") not in {"validation", "val"}:
        raise ValueError("evaluation.periodic_split must be validation or val")
    if evaluation.get("final_split") not in {"final", "test"}:
        raise ValueError("evaluation.final_split must be final or test")
    output = config["output"]
    if "progress_bar" in output and not isinstance(output["progress_bar"], bool):
        raise ValueError("output.progress_bar must be true or false")
    runtime = config.get("runtime", {"backend": "serial"})
    backend = str(runtime.get("backend", "serial"))
    if backend not in {"serial", "client_parallel"}:
        raise ValueError("runtime.backend must be serial or client_parallel")
    if backend == "client_parallel":
        devices = runtime.get("devices")
        if devices != "all":
            if not isinstance(devices, list) or not devices:
                raise ValueError("runtime.devices must be 'all' or a non-empty GPU index list")
            if any(not isinstance(item, int) or item < 0 for item in devices):
                raise ValueError("runtime.devices must contain non-negative integers")
            if len(set(devices)) != len(devices):
                raise ValueError("runtime.devices must not contain duplicates")
        aggregation_device = str(runtime.get("aggregation_device", "auto"))
        if aggregation_device != "auto" and not aggregation_device.startswith("cuda:"):
            raise ValueError("runtime.aggregation_device must be auto or cuda:<index>")
        if runtime.get("start_method", "spawn") != "spawn":
            raise ValueError("client_parallel requires runtime.start_method=spawn for CUDA safety")
        if float(runtime.get("startup_timeout_seconds", 1800)) <= 0:
            raise ValueError("runtime.startup_timeout_seconds must be positive")
        if runtime.get("arrival_policy", "planned") != "planned":
            raise ValueError("client_parallel currently requires runtime.arrival_policy=planned")
        if runtime.get("worker_queue", "fifo") != "fifo":
            raise ValueError("client_parallel currently requires runtime.worker_queue=fifo")
        if str(model.get("dtype", "fp16")) not in {"fp16", "bf16"}:
            raise ValueError("client_parallel model.dtype must be fp16 or bf16")
    method_cfg = config["method"]
    method = create_method(str(method_cfg["name"]), method_cfg.get("params", {}))
    configured_mode = str(config["federation"]["mode"])
    if method.capabilities.mode != configured_mode:
        raise ValueError(
            f"Method {method.name} requires federation.mode={method.capabilities.mode}, "
            f"got {configured_mode}"
        )


def resolved_public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in config.items() if not key.startswith("_")}
