"""MasFL and AdaMasFL (ICML 2025) adapted to federated LoRA states.

Paper: "Momentum-Driven Adaptivity: Towards Tuning-Free Asynchronous
Federated Learning", ICML 2025.

MasFL implements the paper's two-level momentum/control-variate recursion:
clients optimize with beta*(grad-c_i+c)+(1-beta)*g and the server updates the
client controls, global control, and historical descent momentum.  AdaMasFL
normalizes this local direction and aggregates the actual normalized local
displacement, as specified by the adaptive variant.  All vectors below are
LLaVA LoRA/federated-trainable states, not frozen backbone parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import MethodCapabilities, ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import (
    add_scaled,
    clone_state,
    scale_state,
    subtract,
    weighted_mean_states,
    zeros_like,
)


class _MasFL(Method):
    capabilities = MethodCapabilities(
        mode="asynchronous",
        supports_client_ddp=False,
        requires_staleness=True,
        requires_local_state=True,
        requires_mean_gradient=True,
    )
    allowed_params = {
        "beta",
        "server_lr",
        "client_lr",
        "buffer_size",
        "total_clients",
        "normalize_epsilon",
    }
    adaptive = False

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.client_controls: dict[str, dict[str, Any]] = {}
        self.global_control = None
        self.momentum = None
        self.buffer: list[Update] = []

    def prepare_download(self, global_state: dict[str, Any], context: Any) -> dict[str, Any]:
        if self.global_control is None:
            self.global_control = zeros_like(global_state)
            self.momentum = zeros_like(global_state)
        self.client_controls.setdefault(context.client_id, zeros_like(global_state))
        return clone_state(global_state)

    def transform_gradients(self, model: Any, context: Mapping[str, Any]) -> None:
        client = context["client"]
        beta = float(self.params.get("beta", 0.9))
        local = self.client_controls[client.client_id]
        eps = float(self.params.get("normalize_epsilon", 1e-12))
        device_controls = context.setdefault("masfl_device_controls", {})
        direction = {}
        for name, parameter in model.named_federated_parameters():
            controls = device_controls.get(name)
            if controls is None:
                controls = (
                    local[name].to(parameter.device, dtype=parameter.dtype),
                    self.global_control[name].to(parameter.device, dtype=parameter.dtype),
                    self.momentum[name].to(parameter.device, dtype=parameter.dtype),
                )
                device_controls[name] = controls
            old_c, global_c, old_g = controls
            direction[name] = beta * (parameter.grad - old_c + global_c) + (1.0 - beta) * old_g
        if self.adaptive:
            norm_squared = None
            for value in direction.values():
                term = value.detach().float().square().sum()
                norm_squared = term if norm_squared is None else norm_squared + term
            norm = norm_squared.sqrt().clamp_min(eps)
            direction = scale_state(direction, norm.reciprocal())
        for name, parameter in model.named_federated_parameters():
            parameter.grad.copy_(direction[name])

    def client_runtime_state(self, context: Any) -> dict[str, Any]:
        return {
            "client_control": clone_state(self.client_controls[context.client_id]),
            "global_control": clone_state(self.global_control),
            "momentum": clone_state(self.momentum),
        }

    def load_client_runtime_state(self, state: Mapping[str, Any], context: Any) -> None:
        self.client_controls = {
            context.client_id: clone_state(state["client_control"]),
        }
        self.global_control = clone_state(state["global_control"])
        self.momentum = clone_state(state["momentum"])

    def _flush(self, context: ServerContext) -> ServerMutation:
        updates, self.buffer = self.buffer, []
        beta = float(self.params.get("beta", 0.9))
        total_clients = int(self.params.get("total_clients", context.expected_clients))
        changes = []
        for update in updates:
            old = self.client_controls[update.client_id]
            new = update.mean_gradient or scale_state(
                update.delta,
                -1.0 / max(1, update.optimizer_steps * float(self.params.get("client_lr", 2e-5))),
            )
            changes.append(subtract(new, old))
            self.client_controls[update.client_id] = clone_state(new)
        mean_change = weighted_mean_states(changes, [u.num_samples for u in updates])
        momentum_input = add_scaled(self.global_control, mean_change, 1.0)
        self.momentum = add_scaled(scale_state(momentum_input, beta), self.momentum, 1.0 - beta)
        for change in changes:
            self.global_control = add_scaled(self.global_control, change, 1.0 / total_clients)
        if self.adaptive:
            direction = weighted_mean_states(
                [
                    scale_state(
                        update.delta,
                        -1.0
                        / max(
                            1, update.optimizer_steps * float(self.params.get("client_lr", 2e-5))
                        ),
                    )
                    for update in updates
                ],
                [u.num_samples for u in updates],
            )
            new_state = add_scaled(
                context.global_state, direction, -float(self.params.get("server_lr", 1.0))
            )
        else:
            new_state = add_scaled(
                context.global_state, self.momentum, -float(self.params.get("server_lr", 0.01))
            )
        return ServerMutation(
            new_state,
            float(self.params.get("server_lr", 0.01)),
            [u.update_id for u in updates],
            metadata={"variant": self.name, "control_updates": len(updates)},
        )

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        self.buffer.append(update)
        if len(self.buffer) >= int(self.params.get("buffer_size", 1)):
            return [self._flush(server_context)]
        return []

    def on_finish(self, server_context: ServerContext) -> list[ServerMutation]:
        return [self._flush(server_context)] if self.buffer else []

    def state_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "client_controls": self.client_controls,
            "global_control": self.global_control,
            "momentum": self.momentum,
            "buffer": self.buffer,
            "adaptive": self.adaptive,
        }


@register_method("masfl")
class MasFL(_MasFL):
    name = "masfl"


@register_method("adamasfl")
class AdaMasFL(_MasFL):
    name = "adamasfl"
    adaptive = True
