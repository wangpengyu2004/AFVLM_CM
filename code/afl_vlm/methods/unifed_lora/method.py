"""UniFed-LoRA-L homogeneous-backbone specialization, CVPR Workshops 2026.
Core: learned task/modality/layer/module descriptors -> shared hypernetwork ->
client-specific real A/B factors -> local tuning -> displacement-based VJP.

Adaptations: a full-participation synchronous meta-batch replaces one random
client per iteration; initialization is centered at common PEFT factors;
architecture heterogeneity is disabled. Global main evaluation uses the mean
generated factor state, while personalized generated states are secondary.
These are explicit protocol adaptations, not exact original benchmark results.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import MethodCapabilities, ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method
from afl_vlm.methods.unifed_lora.descriptors import TASKS
from afl_vlm.methods.unifed_lora.hypernetwork import LoRAHypernetwork
from afl_vlm.models.base import clone_state, mean_states, state_to_device


@register_method("unifed_lora")
class UniFedLoRA(Method):
    name = "unifed_lora"
    capabilities = MethodCapabilities(
        mode="synchronous",
        supports_client_ddp=False,
        requires_task_id=True,
        requires_custom_adapter=True,
        requires_local_state=True,
    )
    allowed_params = {
        "hidden_size",
        "embedding_dim",
        "hyper_lr",
        "seed",
        "task_descriptors",
        "hyper_device",
        "conditioning_scale",
    }

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        rate = float(self.params.get("hyper_lr", 2e-5))
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("UniFed-LoRA hyper_lr must be finite and positive")
        self.hypernetwork: LoRAHypernetwork | None = None
        self.rounds: dict[int, list[Update]] = defaultdict(list)
        self.downloads: dict[tuple[str, int], dict[str, Any]] = {}
        self.clients: dict[str, str] = {}
        self.initial_state: dict[str, Any] | None = None
        self.server_device: Any = "cpu"

    def validate_runtime(self) -> None:
        if "conditioning_scale" in self.params:
            raise ValueError(
                "This legacy UniFed-LoRA profile describes the former scalar-gate "
                "surrogate. Create a new profile with the real factor hypernetwork."
            )

    def configure_model(self, model: Any, clients: list[Any]) -> None:
        # Generated FP32 A/B and their local displacement must share an exact
        # download anchor, not include FP16 factor rounding as a spurious update.
        for _, parameter in model.named_federated_parameters():
            parameter.data = parameter.data.float()

    def configure_server(self, initial_state: dict[str, Any], clients: list[Any]) -> None:
        self.validate_runtime()
        self.server_device = next(iter(initial_state.values())).device
        self.initial_state = clone_state(initial_state)
        self.clients = {item.id: item.task for item in clients}
        if any(task not in TASKS for task in self.clients.values()):
            raise ValueError("Unknown UniFed-LoRA client task descriptor")
        descriptors = self.params.get(
            "task_descriptors", {task: [float(task == other) for other in TASKS] for task in TASKS}
        )
        if not set(self.clients.values()) <= set(descriptors):
            raise ValueError("Task descriptor configuration does not cover all clients")
        device = self.params.get("hyper_device", "auto")
        device = self.server_device if device == "auto" else device
        self.hypernetwork = LoRAHypernetwork(
            self.initial_state,
            self.clients,
            descriptors,
            int(self.params.get("hidden_size", 256)),
            int(self.params.get("embedding_dim", 16)),
            float(self.params.get("hyper_lr", 2e-5)),
            int(self.params.get("seed", 42)),
            device,
        )

    def prepare_download(self, global_state: dict[str, Any], context: Any) -> dict[str, Any]:
        if self.hypernetwork is None:
            raise RuntimeError("UniFed-LoRA hypernetwork is not initialized")
        with self.hypernetwork.torch.no_grad():
            generated = self.hypernetwork.generate(context.client_id)
            state = state_to_device(generated, self.server_device)
        self.downloads[context.client_id, context.local_round] = clone_state(state)
        return state

    def prepare_upload(self, update: Update, context: Any) -> Update:
        # Common runner's base_state is the shared reference, not this method's
        # generated client model. Use the exact immutable generated download.
        from afl_vlm.models.base import subtract

        start = self.downloads[update.client_id, update.local_round]
        update.base_state = clone_state(start)
        update.delta = subtract(update.local_state, start)
        update.metadata["download_policy"] = "generated_personalized_lora"
        return update

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        bucket = self.rounds[update.local_round]
        if any(item.client_id == update.client_id for item in bucket):
            raise ValueError("Duplicate UniFed-LoRA client in meta-batch")
        bucket.append(update)
        if len(bucket) > server_context.expected_clients:
            raise ValueError("UniFed-LoRA meta-batch exceeds the client population")
        if len(bucket) < server_context.expected_clients:
            return []
        if self.hypernetwork is None:
            raise RuntimeError("UniFed-LoRA hypernetwork is not initialized")
        self.hypernetwork.begin_meta_step()
        for item in bucket:
            self.hypernetwork.accumulate_meta_gradient(
                item.client_id,
                item.local_state,
                self.downloads[item.client_id, item.local_round],
                1 / len(bucket),
            )
        self.hypernetwork.finish_meta_step()
        for item in bucket:
            del self.downloads[item.client_id, item.local_round]
        del self.rounds[update.local_round]
        states = self.secondary_evaluation_states(server_context.global_state)
        global_view = mean_states(states.values())
        return [
            ServerMutation(
                global_view,
                1,
                [item.update_id for item in bucket],
                metadata={
                    "hyper_update": "local_displacement_vjp",
                    "meta_batch_size": len(bucket),
                    "backbone_heterogeneity": False,
                    "primary_global_view": "mean_generated_lora_factors",
                },
            )
        ]

    def secondary_evaluation_states(
        self, global_state: dict[str, Any]
    ) -> Mapping[str, dict[str, Any]]:
        if self.hypernetwork is None:
            return {}
        with self.hypernetwork.torch.no_grad():
            return {
                cid: state_to_device(self.hypernetwork.generate(cid), self.server_device)
                for cid in sorted(self.clients)
            }

    def on_finish(self, server_context: ServerContext) -> list[ServerMutation]:
        if self.rounds or self.downloads:
            raise RuntimeError("Incomplete UniFed-LoRA meta-batches/downloads")
        return []

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "params": self.params,
                "initial_state": self.initial_state,
                "clients": self.clients,
                "hypernetwork": self.hypernetwork.state_dict() if self.hypernetwork else None,
                "pending_rounds": dict(self.rounds),
                "downloads": self.downloads,
            }
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        from types import SimpleNamespace

        self.__init__(state["params"])
        if state["initial_state"] is not None:
            self.configure_server(
                state["initial_state"],
                [SimpleNamespace(id=cid, task=task) for cid, task in state["clients"].items()],
            )
            self.hypernetwork.module.load_state_dict(state["hypernetwork"]["module"])
            self.hypernetwork.optimizer.load_state_dict(state["hypernetwork"]["optimizer"])
        self.rounds = defaultdict(list, copy.deepcopy(state["pending_rounds"]))
        self.downloads = copy.deepcopy(state["downloads"])
