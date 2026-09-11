"""Explicit task backend registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from afl_vlm.data.base import TaskAdapter

_REGISTRY: dict[str, Callable[[str, Mapping[str, Any]], TaskAdapter]] = {}


def register_task_backend(
    name: str,
) -> Callable[[type[TaskAdapter]], type[TaskAdapter]]:
    def decorator(cls: type[TaskAdapter]) -> type[TaskAdapter]:
        if name in _REGISTRY:
            raise ValueError(f"Task backend already registered: {name}")
        _REGISTRY[name] = cls
        return cls

    return decorator


def create_task(task_key: str, config: Mapping[str, Any]) -> TaskAdapter:
    from afl_vlm.data import fedmllm  # noqa: F401

    backend = str(config["backend"])
    try:
        return _REGISTRY[backend](task_key, config)
    except KeyError as exc:
        raise ValueError(
            f"Unknown task backend '{backend}'. Available: {sorted(_REGISTRY)}"
        ) from exc


def backend_names() -> set[str]:
    from afl_vlm.data import fedmllm  # noqa: F401

    return set(_REGISTRY)
