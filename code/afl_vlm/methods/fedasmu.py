"""FedASMU adaptation for LoRA-state asynchronous VLM tuning.

Paper: "FedASMU: Efficient Asynchronous Federated Learning with Dynamic
Staleness-Aware Model Update", AAAI 2024.

Implemented components:
* paper Eq. (2): dynamically parameterized staleness factor xi(o, t);
* paper Eq. (1): bounded alpha = mu_alpha*xi/(1+mu_alpha*xi) server mixing;
* paper Eq. (7--8): one mid-local interpolation with a fresher global state.

The paper's loss-gradient meta-parameter updates and device request policy are
represented by explicit, configurable online coefficient updates and a
deterministic refresh fraction.  This is an AFVLM-CM/LoRA adaptation, not a
claim that the original full-model reinforcement-learning controller is
reproduced byte-for-byte.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import MethodCapabilities, ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import add_scaled, clone_state, subtract


@register_method("fedasmu")
class FedASMU(Method):
    name = "fedasmu"
    capabilities = MethodCapabilities(
        mode="asynchronous",
        supports_client_ddp=False,
        requires_staleness=True,
        requires_fresh_global_during_local_training=True,
    )
    allowed_params = {
        "mu_alpha",
        "lambda",
        "sigma",
        "iota",
        "tau",
        "server_meta_lr",
        "mu_beta",
        "gamma",
        "upsilon",
        "local_meta_lr",
        "refresh_fraction",
    }

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.server_coefficients: dict[str, dict[str, float]] = {}
        self.local_coefficients: dict[str, dict[str, float]] = {}
        self.last_loss: dict[str, float] = {}
        self.last_local_loss: dict[str, float] = {}

    def _server_coeffs(self, client: str) -> dict[str, float]:
        return self.server_coefficients.setdefault(
            client,
            {
                "lambda": float(self.params.get("lambda", 1.0)),
                "sigma": float(self.params.get("sigma", 0.5)),
                "iota": float(self.params.get("iota", 0.0)),
            },
        )

    def local_step(
        self, model: Any, step: int, total_steps: int, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        client = context["client"]
        target = max(1, round(total_steps * float(self.params.get("refresh_fraction", 0.5))))
        if step != target:
            return {}
        fresh_state = client.fresh_global_state
        fresh_version = client.fresh_global_version
        provider = context.get("fresh_state_provider")
        if provider is not None:
            fresh_state, fresh_version = provider()
        if fresh_state is None:
            return {}
        global_round = max(1, int(fresh_version or client.base_version) + 1)
        age = max(1, global_round - client.base_version + 1)
        coeffs = self.local_coefficients.setdefault(
            client.client_id,
            {
                "gamma": float(self.params.get("gamma", 1.0)),
                "upsilon": float(self.params.get("upsilon", 0.5)),
            },
        )
        phi = coeffs["gamma"] / math.sqrt(global_round) * (1.0 - coeffs["upsilon"] / math.sqrt(age))
        phi = max(0.0, phi)
        mu = float(self.params.get("mu_beta", 1.0))
        beta = mu * phi / (1.0 + mu * phi)
        current = model.snapshot_trainable()
        adjusted = add_scaled(current, subtract(fresh_state, current), beta)
        model.load_trainable(adjusted)
        return {
            "fresh_adjustment_step": step,
            "fresh_global_version": fresh_version,
            "beta": beta,
        }

    def client_runtime_state(self, context: Any) -> dict[str, Any]:
        coefficients = self.local_coefficients.get(context.client_id)
        return {
            "local_coefficients": {context.client_id: dict(coefficients)}
            if coefficients is not None
            else {},
        }

    def load_client_runtime_state(self, state: Mapping[str, Any], context: Any) -> None:
        self.local_coefficients = {
            key: dict(value) for key, value in state.get("local_coefficients", {}).items()
        }

    def prepare_upload(self, update: Update, context: Any) -> Update:
        if update.losses:
            loss = update.losses[-1]
            previous = self.last_local_loss.get(update.client_id, loss)
            signal = max(-1.0, min(1.0, previous - loss))
            coefficients = self.local_coefficients.setdefault(
                update.client_id,
                {
                    "gamma": float(self.params.get("gamma", 1.0)),
                    "upsilon": float(self.params.get("upsilon", 0.5)),
                },
            )
            learning_rate = float(self.params.get("local_meta_lr", 0.01))
            coefficients["gamma"] = max(1e-6, coefficients["gamma"] + learning_rate * signal)
            coefficients["upsilon"] = max(0.0, coefficients["upsilon"] - learning_rate * signal)
            self.last_local_loss[update.client_id] = loss
            update.metadata["local_coefficients"] = dict(coefficients)
        return update

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        delay = server_context.version - update.base_version + 1
        if delay > int(self.params.get("tau", 1000000)):
            return [
                ServerMutation(
                    clone_state(server_context.global_state),
                    0.0,
                    [update.update_id],
                    increment_version=False,
                    metadata={"rejected": "staleness_threshold", "staleness": delay - 1},
                )
            ]
        coeffs = self._server_coeffs(update.client_id)
        t = max(1, server_context.version + 1)
        xi = coeffs["lambda"] * math.sqrt(t) / (t * (delay ** coeffs["sigma"])) + coeffs["iota"]
        xi = max(0.0, xi)
        mu = float(self.params.get("mu_alpha", 1.0))
        alpha = mu * xi / (1.0 + mu * xi)
        state = add_scaled(
            server_context.global_state,
            subtract(update.local_state, server_context.global_state),
            alpha,
        )
        if update.losses:
            loss = update.losses[-1]
            previous = self.last_loss.get(update.client_id, loss)
            signal = max(-1.0, min(1.0, previous - loss))
            lr = float(self.params.get("server_meta_lr", 0.01))
            coeffs["lambda"] = max(1e-6, coeffs["lambda"] + lr * signal)
            coeffs["sigma"] = max(0.0, coeffs["sigma"] - lr * signal)
            coeffs["iota"] = max(0.0, coeffs["iota"] + lr * signal / max(1, delay))
            self.last_loss[update.client_id] = loss
        return [
            ServerMutation(
                state,
                alpha,
                [update.update_id],
                metadata={
                    "staleness": delay - 1,
                    "xi": xi,
                    "alpha": alpha,
                    "server_coefficients": dict(coeffs),
                },
            )
        ]

    def state_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "server_coefficients": self.server_coefficients,
            "local_coefficients": self.local_coefficients,
            "last_loss": self.last_loss,
            "last_local_loss": self.last_local_loss,
        }
