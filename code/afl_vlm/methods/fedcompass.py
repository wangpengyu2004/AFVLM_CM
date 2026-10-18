"""FedCompass adaptation for the shared AFVLM-CM virtual clock.

Paper: "FedCompass: Efficient Cross-Silo Federated Learning on
Heterogeneous Client Devices Using a Computing Power-Aware Scheduler",
ICLR 2024.

The custom scheduler (not a staleness coefficient) estimates completion
time from the persisted client speed profile, assigns Q_i in [Q_min,Q_max],
and places near-simultaneous arrivals in the same aggregation group.  The
model state is the LLaVA LoRA/federated-trainable state rather than a full
model.  See scheduling/train_plan.py for the scheduler component.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import MethodCapabilities, ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method, staleness_weight
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import add_scaled, clone_state, subtract


@register_method("fedcompass")
class FedCompass(Method):
    name = "fedcompass"
    capabilities = MethodCapabilities(
        mode="semi_asynchronous",
        supports_client_ddp=True,
        requires_client_speed=True,
        requires_custom_scheduler=True,
        requires_local_step_control=True,
    )
    allowed_params = {
        "min_local_steps",
        "max_local_steps",
        "group_window",
        "speed_momentum",
        "latest_time_factor",
        "alpha",
        "gradient_based",
        "staleness",
    }

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.groups: dict[int, list[Update]] = defaultdict(list)
        if not 0 < float(self.params.get("alpha", 0.9)) <= 1:
            raise ValueError("FedCompass alpha must be in (0, 1]")

    def _staleness(self, age: int) -> float:
        config = self.params.get("staleness", {"type": "constant"})
        if age < 0:
            raise ValueError("Negative FedCompass staleness")
        if config.get("type") == "polynomial":
            # Author exponent convention, with a corrected decaying default.
            exponent = float(config.get("a", -0.5))
            if not math.isfinite(exponent):
                raise ValueError("FedCompass staleness exponent must be finite")
            return (age + 1) ** exponent
        return staleness_weight(age, config)

    def validate_runtime(self) -> None:
        if "group_window" in self.params:
            raise ValueError(
                "Legacy FedCompass group_window described static round grouping; "
                "regenerate a profile for the dynamic computing-power-aware scheduler"
            )
        minimum = int(self.params.get("min_local_steps", 20))
        maximum = int(self.params.get("max_local_steps", 100))
        if not 0 < minimum <= maximum:
            raise ValueError("FedCompass requires 0 < min_local_steps <= max_local_steps")
        momentum = float(self.params.get("speed_momentum", 0.9))
        deadline = float(self.params.get("latest_time_factor", 1.5))
        if not 0 <= momentum <= 1 or not math.isfinite(deadline) or deadline < 1:
            raise ValueError("Invalid FedCompass speed/deadline parameters")
        self._staleness(1)

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        if update.group_id is None:
            raise ValueError("FedCompass requires scheduler-assigned group_id")
        group = self.groups[update.group_id]
        group.append(update)
        expected = int(update.metadata["group_size"])
        if len(group) < expected:
            return []
        del self.groups[update.group_id]
        state = clone_state(server_context.global_state)
        alpha = float(self.params.get("alpha", 0.9))
        gradient_based = bool(self.params.get("gradient_based", True))
        weights = {}
        for item in group:
            weight = (
                alpha
                * self._staleness(server_context.version - item.base_version)
                / server_context.expected_clients
            )
            direction = (
                item.delta
                if gradient_based
                else subtract(item.local_state, server_context.global_state)
            )
            state = add_scaled(state, direction, weight)
            weights[item.update_id] = weight
        return [
            ServerMutation(
                state,
                sum(weights.values()),
                [item.update_id for item in group],
                metadata={
                    "group_id": update.group_id,
                    "group_size": len(group),
                    "local_step_allocation": {
                        item.client_id: item.optimizer_steps for item in group
                    },
                    "group_completion_time": max(item.arrival_time for item in group),
                    "client_coefficients": weights,
                    "global_population_denominator": server_context.expected_clients,
                    "gradient_based": gradient_based,
                    "group_completion_spread": max(
                        item.metadata.get("communication_arrival_time", item.arrival_time)
                        for item in group
                    )
                    - min(
                        item.metadata.get("communication_arrival_time", item.arrival_time)
                        for item in group
                    ),
                },
            )
        ]

    def on_finish(self, server_context: ServerContext) -> list[ServerMutation]:
        if self.groups:
            raise RuntimeError(f"Incomplete FedCompass scheduling groups: {sorted(self.groups)}")
        return []

    def state_dict(self) -> dict[str, Any]:
        return {"params": self.params, "groups": dict(self.groups)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self.groups = defaultdict(list, state.get("groups", {}))
