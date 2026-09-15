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

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import MethodCapabilities, ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import weighted_mean_states


@register_method("fedcompass")
class FedCompass(Method):
    name = "fedcompass"
    capabilities = MethodCapabilities(
        mode="semi_asynchronous",
        requires_client_speed=True,
        requires_custom_scheduler=True,
        requires_local_step_control=True,
    )
    allowed_params = {"min_local_steps", "max_local_steps", "group_window"}

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.groups: dict[int, list[Update]] = defaultdict(list)

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        if update.group_id is None:
            raise ValueError("FedCompass requires scheduler-assigned group_id")
        group = self.groups[update.group_id]
        group.append(update)
        expected = int(update.metadata["group_size"])
        if len(group) < expected:
            return []
        del self.groups[update.group_id]
        state = weighted_mean_states(
            [item.local_state for item in group], [item.num_samples for item in group]
        )
        return [
            ServerMutation(
                state,
                1.0,
                [item.update_id for item in group],
                metadata={
                    "group_id": update.group_id,
                    "group_size": len(group),
                    "local_step_allocation": {
                        item.client_id: item.optimizer_steps for item in group
                    },
                    "group_completion_time": max(item.arrival_time for item in group),
                    "group_completion_spread": max(item.arrival_time for item in group)
                    - min(item.arrival_time for item in group),
                },
            )
        ]

    def on_finish(self, server_context: ServerContext) -> list[ServerMutation]:
        if self.groups:
            raise RuntimeError(f"Incomplete FedCompass scheduling groups: {sorted(self.groups)}")
        return []
