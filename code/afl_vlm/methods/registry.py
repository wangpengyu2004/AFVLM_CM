"""Explicit method registry; adding a method does not touch experiment loops."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from afl_vlm.methods.base import Method

_REGISTRY: dict[str, Callable[[Mapping[str, Any]], Method]] = {}


def register_method(name: str) -> Callable[[type[Method]], type[Method]]:
    def decorator(cls: type[Method]) -> type[Method]:
        if name in _REGISTRY:
            raise ValueError(f"Method already registered: {name}")
        _REGISTRY[name] = cls
        return cls

    return decorator


def _load_builtins() -> None:
    from afl_vlm.methods.baselines import (  # noqa: F401
        async_additive,
        fedasync,
        fedavg_sync,
        fedbuff,
        fedcompass_sim,
        fedopt_sync,
        staleness_decay,
    )
    from afl_vlm.methods.custom import my_method  # noqa: F401


def create_method(name: str, params: Mapping[str, Any]) -> Method:
    _load_builtins()
    try:
        return _REGISTRY[name](params)
    except KeyError as exc:
        raise ValueError(f"Unknown method '{name}'. Available: {sorted(_REGISTRY)}") from exc


def method_names() -> set[str]:
    _load_builtins()
    return set(_REGISTRY)
