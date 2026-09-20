"""Common method lifecycle and capability contract."""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import (
    ClientContext,
    MethodCapabilities,
    ServerContext,
    ServerMutation,
    Update,
)
from afl_vlm.models.base import LoRAState, clone_state


def staleness_weight(staleness: int, config: Mapping[str, Any]) -> float:
    kind = str(config.get("type", "constant"))
    if staleness < 0:
        raise ValueError("Staleness cannot be negative")
    if kind == "constant":
        return 1.0
    if kind == "polynomial":
        return (staleness + 1.0) ** (-float(config.get("a", 0.5)))
    if kind == "hinge":
        threshold = int(config.get("threshold", 1))
        slope = float(config.get("a", 0.5))
        return 1.0 if staleness <= threshold else 1.0 / (1.0 + slope * (staleness - threshold))
    raise ValueError(f"Unknown staleness function: {kind}")


class Method(ABC):
    name = "base"
    evaluation_scope = "server_global"
    capabilities = MethodCapabilities(mode="asynchronous")
    allowed_params: set[str] = set()

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        self.params = dict(params or {})
        unknown = set(self.params) - self.allowed_params
        if unknown:
            raise ValueError(f"Unknown parameter(s) for {self.name}: {sorted(unknown)}")

    def configure_model(self, model: Any, clients: list[Any]) -> None:
        """Install worker/model-side extensions after a model replica loads."""
        return None

    def configure_server(self, initial_state: LoRAState, clients: list[Any]) -> None:
        """Initialize server-only algorithm state without requiring a GPU model."""
        return None

    def validate_runtime(self) -> None:
        """Fail early when a registered method is intentionally unavailable."""
        return None

    def prepare_download(self, global_state: LoRAState, context: ClientContext) -> LoRAState:
        return clone_state(global_state)

    def local_loss(self, base_loss: Any, model: Any, batch: Any, context: Mapping[str, Any]) -> Any:
        return base_loss

    def transform_gradients(self, model: Any, context: Mapping[str, Any]) -> None:
        """Optional post-backward gradient transformation (MasFL/AdaMasFL)."""
        return None

    def local_step(
        self, model: Any, step: int, total_steps: int, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Optional post-step hook. Only FedASMU declares fresh-global access."""
        return {}

    def prepare_upload(self, update: Update, context: ClientContext) -> Update:
        return update

    def client_runtime_state(self, context: ClientContext) -> dict[str, Any]:
        """Export only state required by local hooks for one dispatched job.

        Server aggregation buffers and optimizer state remain authoritative in the
        parent process.  GPU workers receive this minimal snapshot and never call
        :meth:`on_arrival`.
        """
        return {}

    def load_client_runtime_state(self, state: Mapping[str, Any], context: ClientContext) -> None:
        """Install a dispatch-time local-hook snapshot inside one worker."""
        if state:
            raise ValueError(f"Method {self.name} does not accept client runtime state")

    @abstractmethod
    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        raise NotImplementedError

    def on_finish(self, server_context: ServerContext) -> list[ServerMutation]:
        return []

    def evaluation_states(self, global_state: LoRAState) -> Mapping[str, LoRAState]:
        """Return non-global states only for methods whose primary scope requires them."""
        return {"global": clone_state(global_state)}

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy({"params": self.params})

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.params = copy.deepcopy(dict(state["params"]))

    @staticmethod
    def tensor_sqrt(value: Any) -> Any:
        return value.sqrt() if hasattr(value, "sqrt") else math.sqrt(value)
