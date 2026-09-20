"""Unified method registry/factory."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from afl_vlm.methods.base import Method

METHOD_REGISTRY: dict[str, Callable[[Mapping[str, Any]], Method]] = {}


def register_method(name: str) -> Callable[[type[Method]], type[Method]]:
    def decorator(cls: type[Method]) -> type[Method]:
        if name in METHOD_REGISTRY:
            raise ValueError(f"Method already registered: {name}")
        METHOD_REGISTRY[name] = cls
        return cls

    return decorator


def _load_builtins() -> None:
    from afl_vlm.methods import fedasmu, fedcompass, masfl, ours, standard  # noqa: F401
    from afl_vlm.methods.pilot import method as pilot  # noqa: F401
    from afl_vlm.methods.unifed_lora import method as unifed_lora  # noqa: F401


def create_method(name: str, params: Mapping[str, Any] | None = None) -> Method:
    _load_builtins()
    try:
        return METHOD_REGISTRY[name](params or {})
    except KeyError as exc:
        raise ValueError(f"Unknown method '{name}'. Available: {sorted(METHOD_REGISTRY)}") from exc


def method_names() -> tuple[str, ...]:
    _load_builtins()
    return tuple(sorted(METHOD_REGISTRY))


def method_capabilities() -> dict[str, Any]:
    _load_builtins()
    return {name: cls({}).capabilities for name, cls in METHOD_REGISTRY.items()}
