"""Pilot adaptation for AFVLM-CM.

Paper: "Pilot: Building the Federated Multimodal Instruction Tuning
Framework", AAAI 2025.  This implementation preserves (1) task/client visual
adapters and their difference loss, (2) CT-MoA cross-task routing, and (3)
task-wise visual plus nearest-client text-LoRA aggregation.  It uses the same
LLaVA-1.5 backbone and fixed AFVLM-CM partitions; it does not redefine data.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import MethodCapabilities, ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.pilot.aggregation import pilot_aggregate
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import clone_state


@register_method("pilot")
class Pilot(Method):
    name = "pilot"
    capabilities = MethodCapabilities(
        mode="synchronous",
        requires_task_id=True,
        requires_custom_adapter=True,
        requires_local_state=True,
    )
    allowed_params = {
        "adapter_bottleneck",
        "difference_weight",
        "balance_weight",
        "router_z_weight",
        "text_neighbors",
    }

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.rounds: dict[int, list[Update]] = defaultdict(list)
        self.personalized_states: dict[str, dict[str, Any]] = {}

    def configure_model(self, model: Any, clients: list[Any]) -> None:
        model.install_pilot_extension(
            sorted({item.task for item in clients}),
            [item.id for item in clients],
            int(self.params.get("adapter_bottleneck", 64)),
        )

    def local_loss(self, base_loss: Any, model: Any, batch: Any, context: Mapping[str, Any]) -> Any:
        losses = model.pilot_auxiliary_losses()
        return (
            base_loss
            + float(self.params.get("difference_weight", 0.1)) * losses.get("difference", 0.0)
            + float(self.params.get("balance_weight", 0.01)) * losses.get("load_balance", 0.0)
            + float(self.params.get("router_z_weight", 0.001)) * losses.get("router_z", 0.0)
        )

    def prepare_download(self, global_state: dict[str, Any], context: Any) -> dict[str, Any]:
        return clone_state(self.personalized_states.get(context.client_id, global_state))

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        bucket = self.rounds[update.local_round]
        bucket.append(update)
        if len(bucket) < server_context.expected_clients:
            return []
        del self.rounds[update.local_round]
        state, personalized = pilot_aggregate(
            bucket, server_context.global_state, int(self.params.get("text_neighbors", 3))
        )
        self.personalized_states.update(personalized)
        return [
            ServerMutation(
                state,
                1.0,
                [item.update_id for item in bucket],
                metadata={"pilot_tasks": sorted({item.task for item in bucket})},
            )
        ]

    def evaluation_states(self, global_state: dict[str, Any]) -> Mapping[str, dict[str, Any]]:
        return self.personalized_states or {"global": clone_state(global_state)}

    def state_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "personalized_states": self.personalized_states,
            "pending_rounds": dict(self.rounds),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self.personalized_states = dict(state["personalized_states"])
        self.rounds = defaultdict(list, state.get("pending_rounds", {}))
