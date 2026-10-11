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
from afl_vlm.models.base import clone_state, nested_to_device


@register_method("pilot")
class Pilot(Method):
    name = "pilot"
    capabilities = MethodCapabilities(
        mode="synchronous",
        supports_client_ddp=False,
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
        "stage1_rounds",
        "normalize_difference",
    }

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.rounds: dict[int, list[Update]] = defaultdict(list)
        self.personalized_states: dict[str, dict[str, Any]] = {}
        self.private_states: dict[str, dict[str, Any]] = {}
        self.local_private: dict[str, Any] | None = None

    def validate_runtime(self) -> None:
        if "adapter_bottleneck" in self.params:
            raise ValueError("Legacy Pilot residual-bottleneck profiles must be regenerated")
        if int(self.params.get("stage1_rounds", 1)) < 1:
            raise ValueError("Pilot stage1_rounds must be positive")
        if int(self.params.get("text_neighbors", 6)) < 0:
            raise ValueError("Pilot text_neighbors must be non-negative")

    def configure_model(self, model: Any, clients: list[Any]) -> None:
        model.install_pilot_extension(
            sorted({item.task for item in clients}),
            [item.id for item in clients],
            bool(self.params.get("normalize_difference", True)),
        )

    def prepare_local_model(self, model: Any, context: Any) -> None:
        stage = 1 if context.local_round < int(self.params.get("stage1_rounds", 1)) else 2
        model._pilot_connector.configure_local(
            context.task, context.client_id, stage, self.local_private
        )

    def export_local_metadata(self, model: Any, context: Any) -> dict[str, Any]:
        return {"pilot_private_state": model._pilot_connector.export_private()}

    def client_runtime_state(self, context: Any) -> dict[str, Any]:
        return {"private": self.private_states.get(context.client_id)}

    def load_client_runtime_state(self, state: Mapping[str, Any], context: Any) -> None:
        self.local_private = state.get("private")

    def local_loss(self, base_loss: Any, model: Any, batch: Any, context: Mapping[str, Any]) -> Any:
        losses = model.pilot_auxiliary_losses()
        return (
            base_loss
            + float(self.params.get("difference_weight", 0.1)) * losses.get("difference", 0.0)
            + float(self.params.get("balance_weight", 0.1)) * losses.get("load_balance", 0.0)
            + float(self.params.get("router_z_weight", 0.01)) * losses.get("router_z", 0.0)
        )

    def prepare_download(self, global_state: dict[str, Any], context: Any) -> dict[str, Any]:
        return clone_state(self.personalized_states.get(context.client_id, global_state))

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        if "pilot_private_state" not in update.metadata:
            raise ValueError("Pilot upload is missing client-private visual state")
        self.private_states[update.client_id] = nested_to_device(
            update.metadata["pilot_private_state"], "cpu"
        )
        bucket = self.rounds[update.local_round]
        if any(item.client_id == update.client_id for item in bucket):
            raise ValueError("Duplicate Pilot client in a synchronous round")
        bucket.append(update)
        if len(bucket) > server_context.expected_clients:
            raise ValueError("Pilot round exceeds the client population")
        if len(bucket) < server_context.expected_clients:
            return []
        del self.rounds[update.local_round]
        state, personalized = pilot_aggregate(
            bucket, server_context.global_state, int(self.params.get("text_neighbors", 6))
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
        return {"global": clone_state(global_state)}

    def secondary_evaluation_states(self, global_state: Any) -> Mapping[str, Any]:
        return {
            client: {
                "__federated_state__": state,
                "__private_state__": self.private_states[client],
            }
            for client, state in self.personalized_states.items()
        }

    def on_finish(self, server_context: ServerContext) -> list[ServerMutation]:
        if self.rounds:
            raise ValueError("Pilot ended with an incomplete synchronous round")
        return []

    def state_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "personalized_states": self.personalized_states,
            "pending_rounds": dict(self.rounds),
            "private_states": self.private_states,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self.personalized_states = dict(state["personalized_states"])
        self.rounds = defaultdict(list, state.get("pending_rounds", {}))
        self.private_states = dict(state.get("private_states", {}))
