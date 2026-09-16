"""Reserved extension point for the proposed AFVLM method."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import MethodCapabilities
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method


@register_method("ours")
class Ours(Method):
    name = "ours"
    capabilities = MethodCapabilities(
        mode="asynchronous",
        requires_task_id=True,
        requires_staleness=True,
        requires_local_state=True,
    )
    allowed_params = {"server_memory", "client_memory"}

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.server_memory: dict[str, Any] = {}
        self.client_memory: dict[str, Any] = {}

    def validate_runtime(self) -> None:
        raise NotImplementedError("The proposed AFVLM method has not been implemented yet.")

    def on_arrival(self, update: Any, server_context: Any) -> list[Any]:
        _ = {
            "client_id": update.client_id,
            "task": update.task,
            "dataset": update.dataset,
            "local_delta": update.delta,
            "base_version": update.base_version,
            "server_version": server_context.version,
            "staleness": server_context.version - update.base_version,
            "arrival_time": update.arrival_time,
            "num_samples": update.num_samples,
            "server_memory": self.server_memory,
            "client_memory": self.client_memory,
        }
        raise NotImplementedError("The proposed AFVLM method has not been implemented yet.")

    def state_dict(self) -> dict[str, Any]:
        """Checkpoint future method-owned memory without defining its algorithm."""
        return {
            "params": copy.deepcopy(self.params),
            "server_memory": copy.deepcopy(self.server_memory),
            "client_memory": copy.deepcopy(self.client_memory),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self.server_memory = copy.deepcopy(dict(state.get("server_memory", {})))
        self.client_memory = copy.deepcopy(dict(state.get("client_memory", {})))
