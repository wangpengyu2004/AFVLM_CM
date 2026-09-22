"""Capability-driven physical executor selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve_runtime_backend(config: Mapping[str, Any], method: Any) -> str:
    """Resolve physical execution without method-name branches."""

    runtime = config.get("runtime", {})
    configured = str(runtime.get("backend", "serial"))
    policy = str(runtime.get("executor_policy", "fixed"))
    if policy not in {"fixed", "capability"}:
        raise ValueError("runtime.executor_policy must be fixed or capability")
    if policy == "fixed" or configured != "client_parallel":
        return configured
    return "client_ddp" if bool(method.capabilities.supports_client_ddp) else "client_parallel"
