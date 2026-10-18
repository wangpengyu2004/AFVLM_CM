"""FedASMU (AAAI 2024), no-server-data AFVLM specialization.

https://arxiv.org/html/2312.05770v1: Eqs. 1--3 (dynamic server mixing),
5--6 (LSTM/Q request controller), 7--9 (fresh-model local mixing).
The server derivative is measured on the arriving client's private training
mini-batch at the actual current-server/local interpolation. No raw examples
reach the server; validation/test are never used. This single-client objective
is a biased substitute for global F, NOT an exact reproduction. Additional
objective forwards/backwards are algorithm work, not optimizer steps.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import (
    ClientContext,
    MethodCapabilities,
    ServerContext,
    ServerMutation,
    Update,
)
from afl_vlm.methods.base import Method
from afl_vlm.methods.fedasmu_controller import RequestController
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import LoRAState, add_scaled, subtract


@register_method("fedasmu")
class FedASMU(Method):
    name = "fedasmu"
    capabilities = MethodCapabilities(
        mode="asynchronous",
        supports_client_ddp=True,
        requires_staleness=True,
        requires_fresh_global_during_local_training=True,
        requires_local_state=True,
        requires_arrival_objective_feedback=True,
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
        "lambda_lr",
        "sigma_lr",
        "iota_lr",
        "gamma_lr",
        "upsilon_lr",
        "request_bins",
        "rl_hidden_size",
        "rl_lr",
        "rl_epsilon",
        "reward_baseline_rate",
        "q_lr",
        "q_discount",
        "controller_seed",
        "feedback_samples",
        "control_max",
        "feedback_objective",
    }

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        if (
            self.params.get("feedback_objective", "arriving_client_train")
            != "arriving_client_train"
        ):
            raise ValueError("FedASMU only supports the disclosed arriving_client_train oracle")
        for key, default in {
            "mu_alpha": 1,
            "mu_beta": 1,
            "tau": 99,
            "request_bins": 8,
            "rl_hidden_size": 16,
            "feedback_samples": 4,
            "control_max": 100,
        }.items():
            value = float(self.params.get(key, default))
            if (
                not math.isfinite(value)
                or value <= 0
                or (key not in {"mu_alpha", "mu_beta", "control_max"} and int(value) != value)
            ):
                raise ValueError(f"FedASMU {key} must be positive (counts must be integers)")
        for key in (
            "lambda",
            "sigma",
            "iota",
            "gamma",
            "server_meta_lr",
            "local_meta_lr",
            "lambda_lr",
            "sigma_lr",
            "iota_lr",
            "gamma_lr",
            "upsilon_lr",
            "rl_lr",
        ):
            value = float(
                self.params.get(key, 1 if key in {"lambda", "sigma", "iota", "gamma"} else 0.0001)
            )
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"FedASMU {key} must be finite and non-negative")
        for key, default in (
            ("upsilon", 0.5),
            ("rl_epsilon", 0.1),
            ("reward_baseline_rate", 0.1),
            ("q_lr", 0.1),
            ("q_discount", 0.9),
        ):
            if not 0 <= float(self.params.get(key, default)) <= 1:
                raise ValueError(f"FedASMU {key} must lie in [0, 1]")
        self.server_controls: dict[str, dict[str, float]] = {}
        self.local_controls: dict[str, dict[str, float]] = {}
        self.decisions: dict[tuple[str, int], dict[str, Any]] = {}
        self.controller: RequestController | None = None
        self.runtime: dict[str, Any] = {}
        self.cached_batch: Any = None

    def configure_model(self, model: Any, clients: list[Any]) -> None:
        # FP32 LoRA masters for meta-gradients; frozen 7B stays common dtype.
        for _, parameter in model.named_federated_parameters():
            parameter.data = parameter.data.float()
        from afl_vlm.methods.fedasmu_ddp import install_feedback

        install_feedback(model)

    def configure_server(self, initial_state: LoRAState, clients: list[Any]) -> None:
        self.controller = RequestController(self.params, next(iter(initial_state.values())).device)

    def _server(self, client: str) -> dict[str, float]:
        return self.server_controls.setdefault(
            client, {k: float(self.params.get(k, 1.0)) for k in ("lambda", "sigma", "iota")}
        )

    def _local(self, client: str) -> dict[str, float]:
        return self.local_controls.setdefault(
            client,
            {
                "gamma": float(self.params.get("gamma", 1)),
                "upsilon": float(self.params.get("upsilon", 0.5)),
            },
        )

    @staticmethod
    def server_weight(
        controls: Mapping[str, float], version: int, age: int, mu: float
    ) -> tuple[float, dict[str, float]]:
        """Eq. 2 plus exact coefficient derivatives; max(1,t) handles startup."""
        stale = age + 1
        factor = math.exp(-controls["sigma"] * math.log(stale)) / math.sqrt(max(1, version))
        xi = controls["lambda"] * factor + controls["iota"]
        derivative = mu / (1 + mu * xi) ** 2
        return mu * xi / (1 + mu * xi), {
            "lambda": derivative * factor,
            "sigma": -derivative * controls["lambda"] * factor * math.log(stale),
            "iota": derivative,
        }

    @staticmethod
    def local_weight(
        controls: Mapping[str, float], fresh: int, base: int, mu: float
    ) -> tuple[float, dict[str, float]]:
        """Eq. 8: gamma/sqrt(g) * (1-upsilon/sqrt(g-o+1))."""
        if fresh <= base:
            return 0.0, {"gamma": 0.0, "upsilon": 0.0}
        time, stale = math.sqrt(fresh), math.sqrt(fresh - base + 1)
        phi = controls["gamma"] / time * (1 - controls["upsilon"] / stale)
        derivative = mu / (1 + mu * phi) ** 2
        return mu * phi / (1 + mu * phi), {
            "gamma": derivative / time * (1 - controls["upsilon"] / stale),
            "upsilon": -derivative * controls["gamma"] / (time * stale),
        }

    def _learn(
        self,
        controls: dict[str, float],
        derivatives: dict[str, float],
        objective_derivative: float,
        prefix: str,
    ) -> None:
        if not math.isfinite(objective_derivative):
            raise ValueError("FedASMU objective derivative is non-finite")
        for key, derivative in derivatives.items():
            rate = float(self.params.get(f"{key}_lr", self.params.get(f"{prefix}_meta_lr", 0.0001)))
            upper = 1.0 if key == "upsilon" else float(self.params.get("control_max", 100))
            controls[key] = min(
                upper, max(0.0, controls[key] - rate * objective_derivative * derivative)
            )

    def select_refresh_step(self, context: ClientContext, total_steps: int) -> int | None:
        if self.controller is None:
            raise RuntimeError("Configure FedASMU request controller before dispatch")
        from afl_vlm.methods.fedasmu_ddp import distributed_active, select_decision

        if distributed_active():
            decision = select_decision(self.controller, context, total_steps)
        else:
            decision = self.controller.select(
                context.client_id,
                context.local_round,
                context.seed,
                total_steps,
                context.base_version,
            )
        self.decisions[(context.client_id, context.local_round)] = decision
        return decision["step"]

    def client_runtime_state(self, context: ClientContext) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "decision": self.decisions[(context.client_id, context.local_round)],
                "controls": self._local(context.client_id),
            }
        )

    def load_client_runtime_state(self, state: Mapping[str, Any], context: ClientContext) -> None:
        self.runtime = copy.deepcopy(dict(state))
        self.cached_batch = None

    def local_loss(self, base_loss: Any, model: Any, batch: Any, context: Mapping[str, Any]) -> Any:
        self.cached_batch = batch
        return base_loss

    def local_step(
        self, model: Any, step: int, total_steps: int, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not self.runtime or step != self.runtime["decision"]["step"]:
            return {}
        provider, client = context.get("fresh_state_provider"), context["client"]
        if provider is None:
            raise RuntimeError("FedASMU requires its causally scheduled fresh-state provider")
        fresh_state, fresh_version = provider()
        if fresh_version <= client.base_version:
            return {
                "fedasmu_local": {
                    "reward": 0.0,
                    "fresh_version": fresh_version,
                    "mixed": False,
                    "controls": self.runtime["controls"],
                    "request_step": step,
                    "decision": self.runtime["decision"],
                }
            }
        local_state = model.snapshot_trainable()
        beta, derivatives = self.local_weight(
            self.runtime["controls"],
            fresh_version,
            client.base_version,
            float(self.params.get("mu_beta", 1)),
        )
        feedback = model.mixing_objective_feedback(
            local_state, fresh_state, beta, encoded=self.cached_batch
        )
        self._learn(
            self.runtime["controls"], derivatives, feedback["directional_derivative"], "local"
        )
        model.load_trainable(add_scaled(local_state, subtract(fresh_state, local_state), beta))
        return {
            "fedasmu_local": {
                **feedback,
                "reward": feedback["loss_before"] - feedback["loss_mixed"],
                "beta": beta,
                "mixed": True,
                "controls": self.runtime["controls"],
                "fresh_version": fresh_version,
                "request_step": step,
                "decision": self.runtime["decision"],
            }
        }

    def prepare_upload(self, update: Update, context: ClientContext) -> Update:
        decision = self.decisions.pop((context.client_id, context.local_round))
        local = update.metadata.get("fedasmu_local", {})
        if local:
            self.local_controls[context.client_id] = copy.deepcopy(local["controls"])
        if self.controller is None:
            raise RuntimeError("FedASMU request controller is missing")
        from afl_vlm.methods.fedasmu_ddp import controller_owner

        # Only rank zero owns the request policy; every rank receives the same
        # dispatch decision and globally reduced local feedback/reward.
        if controller_owner():
            self.controller.observe(
                context.client_id, context.local_round, decision, float(local.get("reward", 0))
            )
        update.metadata["fedasmu_request_decision"] = decision
        return update

    def arrival_feedback_request(
        self, update: Update, context: ServerContext
    ) -> dict[str, Any] | None:
        if context.version - update.base_version + 1 > int(self.params.get("tau", 99)):
            return None
        alpha, _ = self.server_weight(
            self._server(update.client_id),
            context.version,
            context.version - update.base_version,
            float(self.params.get("mu_alpha", 1)),
        )
        return {
            "left": context.global_state,
            "right": update.local_state,
            "weight": alpha,
            "client_id": update.client_id,
            "task": update.task,
            "seed": update.seed,
            "num_samples": int(self.params.get("feedback_samples", 4)),
            "server_version": context.version,
        }

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        age = server_context.version - update.base_version
        if age < 0:
            raise ValueError("FedASMU received a future base version")
        if age + 1 > int(self.params.get("tau", 99)):
            update.metadata["rejection_reason"] = "fedasmu_staleness_bound"
            return []
        feedback = update.metadata.get("arrival_objective_feedback")
        if not feedback or feedback["server_version"] != server_context.version:
            raise RuntimeError(
                "FedASMU requires real client-training-objective feedback "
                "at the current server version"
            )
        controls = self._server(update.client_id)
        before = copy.deepcopy(controls)
        _, derivatives = self.server_weight(
            controls, server_context.version, age, float(self.params.get("mu_alpha", 1))
        )
        self._learn(controls, derivatives, float(feedback["directional_derivative"]), "server")
        alpha, _ = self.server_weight(
            controls, server_context.version, age, float(self.params.get("mu_alpha", 1))
        )
        new_state = add_scaled(
            server_context.global_state,
            subtract(update.local_state, server_context.global_state),
            alpha,
        )
        return [
            ServerMutation(
                new_state,
                alpha,
                [update.update_id],
                metadata={
                    "fedasmu": True,
                    "paper_staleness": age + 1,
                    "server_controls_before": before,
                    "server_controls_after": copy.deepcopy(controls),
                    "arrival_objective_feedback": feedback,
                    "local_adjustment": update.metadata.get("fedasmu_local"),
                    "request_decision": update.metadata.get("fedasmu_request_decision"),
                    "adaptation": "client_train_objective_no_server_data",
                    "extra_work_not_in_train_plan": True,
                },
            )
        ]

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                **super().state_dict(),
                "server_controls": self.server_controls,
                "local_controls": self.local_controls,
                "decisions": self.decisions,
                "controller": self.controller.state_dict() if self.controller else None,
            }
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self.server_controls = copy.deepcopy(state["server_controls"])
        self.local_controls = copy.deepcopy(state["local_controls"])
        self.decisions = copy.deepcopy(state["decisions"])
        if state["controller"] is not None:
            if self.controller is None:
                raise RuntimeError("Configure FedASMU server before restoring its GPU controller")
            self.controller.load_state_dict(state["controller"])
