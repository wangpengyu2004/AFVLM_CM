"""Metadata-derived descriptors; client indices are deliberately excluded."""

from __future__ import annotations

from collections.abc import Mapping

TASKS = ("cls", "caption", "vqa", "chart_vqa", "visual_reasoning", "grounding")


def task_descriptor(
    task: str,
    metadata: Mapping[str, str] | None = None,
    configured: Mapping[str, list[float]] | None = None,
) -> list[float]:
    if task not in TASKS:
        raise ValueError(f"Unknown task descriptor: {task}")
    meta = dict(metadata or {})
    modality = meta.get("modality", "vision_language")
    architecture = meta.get("architecture", "llava15_7b")
    task_vector = (
        list(configured[task])
        if configured is not None
        else [1.0 if item == task else 0.0 for item in TASKS]
    )
    return [
        *task_vector,
        1.0 if modality == "vision_language" else 0.0,
        1.0 if architecture == "llava15_7b" else 0.0,
    ]


def parameter_descriptor(name: str) -> list[float]:
    lower = name.lower()
    module = [
        float(token in lower) for token in ("q_proj", "k_proj", "v_proj", "o_proj", "projector")
    ]
    digits = [int(item) for item in name.replace(".", " ").split() if item.isdigit()]
    depth = (digits[0] / 32.0) if digits else 0.0
    return [*module, depth]
