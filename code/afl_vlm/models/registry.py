"""Explicit model-adapter registry."""

from __future__ import annotations

from collections.abc import Callable

from afl_vlm.models.base import ModelAdapter

_REGISTRY: dict[str, Callable[[], ModelAdapter]] = {}


def register_model(name: str) -> Callable[[type[ModelAdapter]], type[ModelAdapter]]:
    def decorator(cls: type[ModelAdapter]) -> type[ModelAdapter]:
        if name in _REGISTRY:
            raise ValueError(f"Model adapter already registered: {name}")
        _REGISTRY[name] = cls
        return cls

    return decorator


def create_model(name: str) -> ModelAdapter:
    _load_builtins()
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown model adapter '{name}'. Available: {sorted(_REGISTRY)}") from exc


def model_names() -> set[str]:
    _load_builtins()
    return set(_REGISTRY)


def _load_builtins() -> None:
    from afl_vlm.models import llava15_fcit, qwen25_vl, tiny_mock  # noqa: F401
