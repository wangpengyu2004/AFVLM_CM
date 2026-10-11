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
    ScheduledEvent,
    ServerContext,
    ServerMutation,
    Update,
)
from afl_vlm.models.base import LoRAState, clone_state


def staleness_weight(staleness: int, config: Mapping[str, Any]) -> float:
    kind = str(config.get("type", "constant"))
    if staleness < 0:
        raise ValueError("Staleness cannot be negative")
    if kind in {"constant", "none"}:
        return 1.0
    slope = float(config.get("a", 0.5))
    if not math.isfinite(slope) or slope < 0:
        raise ValueError("Staleness a must be finite and non-negative")
    if kind in {"polynomial", "power"}:
        return (staleness + 1.0) ** (-slope)
    if kind == "linear":
        return 1.0 / (1.0 + slope * staleness)
    if kind in {"exponential", "exp"}:
        return math.exp(-slope * staleness)
    if kind == "sigmoid":
        # Author FedAsync code: floor 1/b, not a logistic function.
        floor_denominator = float(config.get("b", 1.0))
        if not math.isfinite(floor_denominator) or floor_denominator < 1:
            raise ValueError("Sigmoid staleness b must be finite and >= 1")
        value = math.exp(-slope * staleness)
        return (1.0 - value) / floor_denominator + value
    if kind == "hinge":
        threshold = int(config.get("b", config.get("threshold", 1)))
        if threshold < 0:
            raise ValueError("Hinge staleness threshold must be non-negative")
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

    def configure_schedule(self, events: list[ScheduledEvent]) -> None:
        """Observe the final virtual Plan without changing scheduling or local work."""
        return None

    def validate_runtime(self) -> None:
        """Fail early when a registered method is intentionally unavailable."""
        return None

    def configure_local_training(self, train_config: Any) -> Any:
        """Algorithm-specific optimizer requirements; default preserves all settings."""
        return train_config

    def select_refresh_step(self, context: ClientContext, total_steps: int) -> int | None:
        """Select a causal within-job refresh slot; ordinary methods have none."""
        if not self.capabilities.requires_fresh_global_during_local_training or total_steps < 2:
            return None
        return max(
            1,
            min(
                total_steps - 1,
                round(total_steps * float(self.params.get("refresh_fraction", 0.5))),
            ),
        )

    def arrival_feedback_request(
        self, update: Update, context: ServerContext
    ) -> dict[str, Any] | None:
        """Optional client-training-objective oracle; no default computation."""
        return None

    def prepare_local_model(self, model: Any, context: ClientContext) -> None:
        """Default no-op; method-private adapters may select their local stage."""
        return None

    def export_local_metadata(self, model: Any, context: ClientContext) -> dict[str, Any]:
        return {}

    def initialize_client_gradient(self, update: Update) -> None:
        raise NotImplementedError(f"{self.name} has no initial-gradient protocol")

    def finalize_initial_gradients(self) -> None:
        raise NotImplementedError(f"{self.name} has no initial-gradient protocol")

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

    def secondary_evaluation_states(self, global_state: LoRAState) -> Mapping[str, LoRAState]:
        """Optional native personalized results, separate from global main metrics."""
        return {}

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy({"params": self.params})

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.params = copy.deepcopy(dict(state["params"]))

    @staticmethod
    def tensor_sqrt(value: Any) -> Any:
        return value.sqrt() if hasattr(value, "sqrt") else math.sqrt(value)
