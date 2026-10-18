"""MasFL/AdaMasFL, ICML 2025, Algorithms 1--4, LoRA specialization.
Paper: https://proceedings.mlr.press/v267/yan25f.html .
No verified author repository. Real fixed-model gradient initialization,
uniform arrival averaging, separate local/server controls, SGD/normalized SGD.
Initial-gradient work is recorded separately and never counted as an upload.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from afl_vlm.federation.types import MethodCapabilities, ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import add_scaled, clone_state, mean_states, scale_state, subtract


class _MasFL(Method):
    capabilities = MethodCapabilities(
        mode="asynchronous",
        supports_client_ddp=True,
        requires_staleness=True,
        requires_local_state=True,
        requires_mean_gradient=True,
        requires_initial_gradient=True,
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
        beta = float(self.params.get("beta", 0.9))
        if not math.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError("MasFL beta must be in [0, 1]")
        for key, default in (
            ("server_lr", 1.0 if self.adaptive else 0.01),
            ("client_lr", 2e-5),
            ("normalize_epsilon", 1e-12),
        ):
            value = float(self.params.get(key, default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"MasFL {key} must be finite and positive")
        if int(self.params.get("buffer_size", 1)) <= 0:
            raise ValueError("MasFL buffer_size must be positive")
        self.client_controls: dict[str, dict[str, Any]] = {}
        self.local_controls: dict[str, dict[str, Any]] = {}
        self.global_control = None
        self.momentum = None
        self.buffer: list[Update] = []
        self.initial_gradient_only = False
        self.expected_clients = 0

    def configure_server(self, initial_state: dict[str, Any], clients: list[Any]) -> None:
        self.expected_clients = len(clients)
        if int(self.params.get("total_clients", self.expected_clients)) != self.expected_clients:
            raise ValueError("MasFL total_clients must match the actual client population")

    def configure_model(self, model: Any, clients: list[Any]) -> None:
        # Normalized SGD can make per-coordinate steps far smaller than FP16
        # resolution. Keep only federated trainables in FP32, not the 7B backbone.
        for _, parameter in model.named_federated_parameters():
            parameter.data = parameter.data.float()

    def initialize_client_gradient(self, update: Update) -> None:
        if update.mean_gradient is None or update.optimizer_steps <= 0:
            raise ValueError("MasFL initialization requires real local mean gradients")
        if update.client_id in self.client_controls:
            raise ValueError("Duplicate MasFL initial client gradient")
        self.client_controls[update.client_id] = clone_state(update.mean_gradient)
        self.local_controls[update.client_id] = clone_state(update.mean_gradient)

    def finalize_initial_gradients(self) -> None:
        if len(self.client_controls) != self.expected_clients:
            raise ValueError("MasFL requires initial gradients from every client")
        self.global_control = mean_states(self.client_controls.values())
        self.momentum = clone_state(self.global_control)

    def prepare_download(self, global_state: dict[str, Any], context: Any) -> dict[str, Any]:
        if self.global_control is None or context.client_id not in self.local_controls:
            raise RuntimeError("MasFL initial-gradient protocol has not completed")
        return clone_state(global_state)

    def configure_local_training(self, train_config: Any) -> Any:
        # AdamW destroys the paper's SGD / normalized-SGD recursion.
        configured = replace(
            train_config,
            optimizer="sgd",
            learning_rate=float(self.params.get("client_lr", train_config.learning_rate)),
            gradient_estimation_only=self.initial_gradient_only,
        )
        if self.initial_gradient_only:
            import torch

            # Shared initialization packages are executor-neutral. Bind their
            # real rank here, privately, so probes cover distinct DDP shards.
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                from afl_vlm.batching import planned_optimizer_steps

                world = torch.distributed.get_world_size()
                client = configured.context["client"]
                configured = replace(
                    configured,
                    distributed_rank=torch.distributed.get_rank(),
                    distributed_world_size=world,
                    planned_optimizer_steps=planned_optimizer_steps(
                        client.num_samples,
                        configured.local_epochs,
                        {
                            "batch_size": configured.batch_size,
                            "gradient_accumulation": configured.gradient_accumulation,
                            "batch_size_scope": configured.batch_size_scope,
                            "reference_world_size": configured.reference_world_size,
                        },
                        world,
                    ),
                )
        return configured

    def transform_gradients(self, model: Any, context: Mapping[str, Any]) -> None:
        if self.initial_gradient_only:
            return
        import torch

        beta = float(self.params.get("beta", 0.9))
        local = self.local_controls[context["client"].client_id]
        controls = context.setdefault("masfl_device_controls", {})
        direction = {}
        with torch.no_grad():
            for name, parameter in model.named_federated_parameters():
                if parameter.grad is None:
                    raise RuntimeError(f"MasFL requires every federated gradient: {name}")
                if name not in controls:
                    controls[name] = tuple(
                        state[name].detach().to(parameter.device, dtype=torch.float32)
                        for state in (local, self.global_control, self.momentum)
                    )
                ci, c, g = controls[name]
                direction[name] = beta * (parameter.grad.float() - ci + c) + (1 - beta) * g
            if self.adaptive:
                norm = sum(value.square().sum() for value in direction.values()).sqrt()
                direction = scale_state(
                    direction,
                    norm.clamp_min(float(self.params.get("normalize_epsilon", 1e-12))).reciprocal(),
                )
            for name, parameter in model.named_federated_parameters():
                parameter.grad.copy_(direction[name])

    def client_runtime_state(self, context: Any) -> dict[str, Any]:
        return {
            "client_control": clone_state(self.local_controls[context.client_id]),
            "global_control": clone_state(self.global_control),
            "momentum": clone_state(self.momentum),
        }

    def load_client_runtime_state(self, state: Mapping[str, Any], context: Any) -> None:
        self.initial_gradient_only = bool(state.get("initial_gradient_only", False))
        if self.initial_gradient_only:
            return
        self.local_controls = {context.client_id: clone_state(state["client_control"])}
        self.global_control = clone_state(state["global_control"])
        self.momentum = clone_state(state["momentum"])

    def prepare_upload(self, update: Update, context: Any) -> Update:
        if update.mean_gradient is None:
            raise ValueError("MasFL raw mean-gradient control cannot be replaced by displacement")
        self.local_controls[update.client_id] = clone_state(update.mean_gradient)
        return update

    def _flush(self, context: ServerContext) -> ServerMutation:
        updates, self.buffer = self.buffer, []
        beta = float(self.params.get("beta", 0.9))
        changes = []
        for update in updates:
            if update.mean_gradient is None:
                raise ValueError("MasFL arrival is missing its raw mean gradient")
            changes.append(subtract(update.mean_gradient, self.client_controls[update.client_id]))
            self.client_controls[update.client_id] = clone_state(update.mean_gradient)
        if self.adaptive:
            directions = []
            for update in updates:
                eta = float(update.metadata["local_learning_rate"])
                if eta <= 0 or update.optimizer_steps <= 0:
                    raise ValueError("AdaMasFL requires positive actual eta and K")
                directions.append(scale_state(update.delta, -1 / (eta * update.optimizer_steps)))
            descent = mean_states(directions)  # Algorithm 3 step 5
        else:
            descent = self.momentum
        self.momentum = add_scaled(
            scale_state(add_scaled(self.global_control, mean_states(changes), 1), beta),
            descent,
            1 - beta,
        )
        for change in changes:
            self.global_control = add_scaled(self.global_control, change, 1 / self.expected_clients)
        server_lr = float(self.params.get("server_lr", 1.0 if self.adaptive else 0.01))
        state = add_scaled(
            context.global_state, descent if self.adaptive else self.momentum, -server_lr
        )
        return ServerMutation(
            state,
            server_lr,
            [u.update_id for u in updates],
            metadata={
                "variant": self.name,
                "control_updates": len(updates),
                "buffer_weighting": "uniform",
                "local_optimizer": "sgd",
            },
        )

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        self.buffer.append(update)
        return (
            [self._flush(server_context)]
            if len(self.buffer) >= int(self.params.get("buffer_size", 1))
            else []
        )

    def on_finish(self, server_context: ServerContext) -> list[ServerMutation]:
        return [self._flush(server_context)] if self.buffer else []

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                key: getattr(self, key)
                for key in (
                    "params",
                    "client_controls",
                    "local_controls",
                    "global_control",
                    "momentum",
                    "buffer",
                    "expected_clients",
                )
            }
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.__init__(state["params"])
        for key in (
            "client_controls",
            "local_controls",
            "global_control",
            "momentum",
            "buffer",
            "expected_clients",
        ):
            setattr(self, key, copy.deepcopy(state[key]))


@register_method("masfl")
class MasFL(_MasFL):
    name = "masfl"


@register_method("adamasfl")
class AdaMasFL(_MasFL):
    name = "adamasfl"
    adaptive = True
