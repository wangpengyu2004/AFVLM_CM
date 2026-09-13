"""Resolve one fixed-task VLM benchmark variant into runtime configuration."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

VARIANT_ALIASES = {
    "small_balanced": "small_16_clients/balanced",
    "small_quantity_skew": "small_16_clients/quantity_skew",
    "medium_balanced": "medium_40_clients/balanced",
    "medium_quantity_skew": "medium_40_clients/quantity_skew",
    "large_balanced": "large_80_clients/balanced",
    "large_quantity_skew": "large_80_clients/quantity_skew",
}


def variant_names() -> set[str]:
    """Return stable, user-facing variant aliases."""
    return set(VARIANT_ALIASES)


def _root_path(raw: str, config_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    repository_candidate = (config_path.parent.parent / path).resolve()
    if repository_candidate.exists():
        return repository_candidate
    return path.resolve()


def resolve_fixed_task_preset(payload: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Hydrate tasks and client ownership from a generated benchmark variant.

    The generated ``framework_config.yaml`` remains the benchmark's machine-readable
    manifest. Only dataset-owned fields are imported; training, methods, evaluation,
    and output policy continue to come from the user's config.
    """
    dataset = dict(payload.get("dataset") or {})
    if dataset.get("name") != "fixed_task_vlm":
        return payload
    variant = str(dataset.get("variant", ""))
    try:
        canonical = VARIANT_ALIASES[variant]
    except KeyError as exc:
        raise ValueError(
            f"Unknown fixed-task dataset.variant '{variant}'. Available: {sorted(VARIANT_ALIASES)}"
        ) from exc
    root = _root_path(str(dataset.get("root", "data/fcit/fixed_task_benchmark")), config_path)
    variant_dir = root / Path(canonical)
    generated_path = variant_dir / "framework_config.yaml"
    if not generated_path.is_file():
        raise FileNotFoundError(
            f"Fixed-task variant is not generated: {generated_path}. "
            "Run python -m scripts.prepare_fixed_task_benchmark first."
        )
    generated = yaml.safe_load(generated_path.read_text(encoding="utf-8"))
    if not isinstance(generated, dict):
        raise ValueError(f"Invalid generated fixed-task config: {generated_path}")

    resolved = copy.deepcopy(payload)
    image_root = str(dataset.get("image_root", "data/fcit/dataset"))
    require_images = bool(dataset.get("require_images", True))
    tasks = copy.deepcopy(dict(generated["tasks"]))
    for task in tasks.values():
        task["image_root"] = image_root
        task["require_images"] = require_images
    resolved["tasks"] = tasks

    clients = copy.deepcopy(dict(resolved.get("clients") or {}))
    generated_clients = dict(generated["clients"])
    clients["count"] = int(generated_clients["count"])
    clients["data_partition"] = "prepartitioned"
    clients["assignments"] = copy.deepcopy(generated_clients["assignments"])
    resolved["clients"] = clients

    timing = copy.deepcopy(dict(resolved.get("timing") or {}))
    timing["train_delay"] = copy.deepcopy(generated["timing"]["train_delay"])
    resolved["timing"] = timing
    if str(resolved.get("run", {}).get("output_root")) == "auto":
        resolved["run"]["output_root"] = f"runs/afvlm_cm/{canonical}"
    resolved["dataset"]["resolved_variant"] = canonical
    return resolved
