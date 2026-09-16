"""UniFed-LoRA specialization for a homogeneous LLaVA backbone.

Paper: "UniFed-LoRA: Exploiting Semantic Task Correlation for Heterogeneous
Multimodal Federated Fine-Tuning", CVPR Workshops 2026.

The original method conditions LoRA across task, modality, layer, module, and
backbone descriptors.  Here backbone heterogeneity is explicitly disabled:
all clients use LLaVA-1.5-7B, while task/modality/layer/module descriptors and
server hypernetwork conditioning remain active.  Additional architecture
descriptor dimensions can be introduced without client-index assumptions.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import MethodCapabilities, ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method
from afl_vlm.methods.unifed_lora.descriptors import parameter_descriptor, task_descriptor
from afl_vlm.methods.unifed_lora.hypernetwork import LoRAHypernetwork
from afl_vlm.models.base import add_scaled, clone_state, state_norm, weighted_mean_states


@register_method("unifed_lora")
class UniFedLoRA(Method):
    name = "unifed_lora"
    capabilities = MethodCapabilities(
        mode="synchronous", requires_task_id=True, requires_custom_adapter=True
    )
    allowed_params = {
        "hidden_size",
        "hyper_lr",
        "conditioning_scale",
        "seed",
        "task_descriptors",
    }

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.hypernetwork: LoRAHypernetwork | None = None
        self.rounds: dict[int, list[Update]] = defaultdict(list)
        self.server_device: Any = "cpu"

    def configure_server(self, initial_state: dict[str, Any], clients: list[Any]) -> None:
        first = next(iter(initial_state.values()))
        self.server_device = getattr(first, "device", "cpu")

    def _network(self) -> LoRAHypernetwork:
        if self.hypernetwork is None:
            configured = self.params.get("task_descriptors")
            descriptor_size = len(next(iter(configured.values()))) + 8 if configured else 14
            self.hypernetwork = LoRAHypernetwork(
                descriptor_size,
                int(self.params.get("hidden_size", 32)),
                float(self.params.get("hyper_lr", 1e-3)),
                int(self.params.get("seed", 2026)),
                self.server_device,
            )
        return self.hypernetwork

    def prepare_download(self, global_state: dict[str, Any], context: Any) -> dict[str, Any]:
        network = self._network()
        result = clone_state(global_state)
        task = task_descriptor(context.task, configured=self.params.get("task_descriptors"))
        scale = float(self.params.get("conditioning_scale", 0.05))
        for key in result:
            gate = network.gate([*task, *parameter_descriptor(key)])
            result[key] = result[key] * (1.0 + scale * gate)
        return result

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        bucket = self.rounds[update.local_round]
        bucket.append(update)
        if len(bucket) < server_context.expected_clients:
            return []
        del self.rounds[update.local_round]
        network = self._network()
        losses = []
        for item in bucket:
            for key in item.delta:
                if hasattr(item.delta[key], "detach"):
                    delta_norm = item.delta[key].detach().float().norm()
                    base_norm = item.base_state[key].detach().float().norm().clamp_min(1e-12)
                    target = (delta_norm / base_norm).clamp(max=1.0)
                else:
                    target = min(
                        1.0,
                        state_norm({key: item.delta[key]})
                        / max(1e-12, state_norm({key: item.base_state[key]})),
                    )
                descriptor = [
                    *task_descriptor(item.task, configured=self.params.get("task_descriptors")),
                    *parameter_descriptor(key),
                ]
                losses.append(network.fit(descriptor, target))
        delta = weighted_mean_states(
            [item.delta for item in bucket], [item.num_samples for item in bucket]
        )
        return [
            ServerMutation(
                add_scaled(server_context.global_state, delta, 1.0),
                1.0,
                [item.update_id for item in bucket],
                metadata={
                    "hypernetwork_loss": float((sum(losses) / len(losses)).item())
                    if hasattr(losses[0], "item")
                    else sum(losses) / len(losses),
                    "backbone_heterogeneity": False,
                },
            )
        ]

    def state_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "hypernetwork": self.hypernetwork.state_dict() if self.hypernetwork else None,
            "pending_rounds": dict(self.rounds),
            "backbone_heterogeneity": False,
        }
