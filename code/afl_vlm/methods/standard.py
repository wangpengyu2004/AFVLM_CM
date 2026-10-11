"""Reference, classical, and basic asynchronous federated optimizers."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import MethodCapabilities, ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method, staleness_weight
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import (
    add_scaled,
    clone_state,
    map_state,
    subtract,
    weighted_mean_states,
    zeros_like,
    zip_states,
)


def _proximal_loss(base_loss: Any, model: Any, context: Mapping[str, Any], mu: float) -> Any:
    """Download-anchored proximal objective, over federated trainables only."""
    if mu == 0:
        return base_loss
    device_anchor = context.setdefault("proximal_device_anchor", {})
    penalty = None
    for name, parameter in model.named_federated_parameters():
        reference = device_anchor.get(name)
        if reference is None:
            reference = context["base_state"][name].detach().to(parameter.device).float()
            device_anchor[name] = reference
        value = (parameter.float() - reference).square().sum()
        penalty = value if penalty is None else penalty + value
    return base_loss if penalty is None else base_loss + 0.5 * mu * penalty


class _RoundAggregator(Method):
    capabilities = MethodCapabilities(mode="synchronous")

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self._rounds: dict[int, list[Update]] = defaultdict(list)

    def aggregate(self, updates: list[Update], server_context: ServerContext) -> ServerMutation:
        states = [update.local_state for update in updates]
        state = weighted_mean_states(states, [update.num_samples for update in updates])
        return ServerMutation(state, 1.0, [update.update_id for update in updates])

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        bucket = self._rounds[update.local_round]
        if any(item.client_id == update.client_id for item in bucket):
            raise ValueError("Duplicate client upload in a synchronous round")
        bucket.append(update)
        if len(bucket) < server_context.expected_clients:
            return []
        if len(bucket) > server_context.expected_clients:
            raise RuntimeError(f"Too many updates in synchronous round {update.local_round}")
        del self._rounds[update.local_round]
        return [self.aggregate(bucket, server_context)]

    def on_finish(self, server_context: ServerContext) -> list[ServerMutation]:
        if self._rounds:
            raise RuntimeError(f"Incomplete synchronous rounds: {sorted(self._rounds)}")
        return []

    def state_dict(self) -> dict[str, Any]:
        return {"params": self.params, "pending_rounds": dict(self._rounds)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self._rounds = defaultdict(list, state.get("pending_rounds", {}))


@register_method("local")
class LocalTraining(Method):
    name = "local"
    evaluation_scope = "client_local_mean"
    capabilities = MethodCapabilities(mode="local", requires_local_state=True)
    allowed_params = set()

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.client_states: dict[str, Any] = {}

    def prepare_download(self, global_state: dict[str, Any], context: Any) -> dict[str, Any]:
        return clone_state(self.client_states.get(context.client_id, global_state))

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        self.client_states[update.client_id] = clone_state(update.local_state)
        return [
            ServerMutation(
                clone_state(server_context.global_state),
                0.0,
                [update.update_id],
                increment_version=False,
                metadata={"local_only": True},
            )
        ]

    def evaluation_states(self, global_state: dict[str, Any]) -> Mapping[str, dict[str, Any]]:
        return {key: clone_state(value) for key, value in self.client_states.items()}

    def state_dict(self) -> dict[str, Any]:
        return {"params": self.params, "client_states": self.client_states}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self.client_states = {
            key: clone_state(value) for key, value in state["client_states"].items()
        }


@register_method("fedavg")
class FedAvg(_RoundAggregator):
    """Sample-size weighted synchronous FedAvg over federated trainables."""

    name = "fedavg"
    allowed_params = set()


@register_method("fedprox")
class FedProx(_RoundAggregator):
    name = "fedprox"
    allowed_params = {"mu"}

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        mu = float(self.params.get("mu", 0.01))
        if not math.isfinite(mu) or mu < 0:
            raise ValueError("FedProx mu must be finite and non-negative")

    def local_loss(self, base_loss: Any, model: Any, batch: Any, context: Mapping[str, Any]) -> Any:
        return _proximal_loss(base_loss, model, context, float(self.params.get("mu", 0.01)))


@register_method("fedadam")
class FedAdam(_RoundAggregator):
    name = "fedadam"
    allowed_params = {"server_lr", "beta1", "beta2", "epsilon", "variant"}

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.m = None
        self.v = None
        self.step = 0
        if self.params.get("variant", "legacy") not in {"paper", "keras", "legacy"}:
            raise ValueError("FedAdam variant must be paper, keras, or legacy")
        for key, default in (("beta1", 0.9), ("beta2", 0.999)):
            value = float(self.params.get(key, default))
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"FedAdam {key} must be finite and in [0,1)")
        for key, default in (("server_lr", 0.01), ("epsilon", 1e-8)):
            value = float(self.params.get(key, default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"FedAdam {key} must be finite and positive")

    def aggregate(self, updates: list[Update], server_context: ServerContext) -> ServerMutation:
        delta = weighted_mean_states([u.delta for u in updates], [u.num_samples for u in updates])
        variant = str(self.params.get("variant", "legacy"))
        eps = float(self.params.get("epsilon", 1e-8))
        if self.m is None:
            self.m, self.v = zeros_like(delta), zeros_like(delta)
            if variant == "paper":
                # FedOpt Algorithm 2 requires v_-1 >= tau^2. Choose the
                # smallest permitted accumulator; Keras/legacy start at zero.
                self.v = map_state(self.v, lambda value: value + eps**2)
        beta1, beta2 = float(self.params.get("beta1", 0.9)), float(self.params.get("beta2", 0.999))
        self.m = zip_states(self.m, delta, lambda m, g: beta1 * m + (1 - beta1) * g)
        self.v = zip_states(self.v, delta, lambda v, g: beta2 * v + (1 - beta2) * g * g)
        self.step += 1
        # FedOpt Algorithm 2 uses uncorrected moments. Author experiments use
        # Keras Adam: bias corrections multiply the learning rate, with epsilon
        # added to sqrt(v), NOT sqrt(v_hat). Keep old profiles explicitly legacy.
        m_hat, v_hat = self.m, self.v
        learning_rate = float(self.params.get("server_lr", 0.01))
        if variant == "legacy":
            m_hat = map_state(self.m, lambda x: x / (1 - beta1**self.step))
            v_hat = map_state(self.v, lambda x: x / (1 - beta2**self.step))
        elif variant == "keras":
            learning_rate *= math.sqrt(1 - beta2**self.step) / (1 - beta1**self.step)
        direction = zip_states(
            m_hat,
            v_hat,
            lambda m, v: m / (v.sqrt() + eps) if hasattr(v, "sqrt") else m / (v**0.5 + eps),
        )
        state = add_scaled(server_context.global_state, direction, learning_rate)
        return ServerMutation(
            state,
            learning_rate,
            [u.update_id for u in updates],
            metadata={"fedadam_variant": variant, "server_optimizer_step": self.step},
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "m": self.m,
            "v": self.v,
            "step": self.step,
            "pending_rounds": dict(self._rounds),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self.m, self.v, self.step = state["m"], state["v"], int(state["step"])
        self._rounds = defaultdict(list, state.get("pending_rounds", {}))


@register_method("fedasync")
class FedAsync(Method):
    name = "fedasync"
    capabilities = MethodCapabilities(mode="asynchronous", requires_staleness=True)
    allowed_params = {
        "alpha",
        "staleness",
        "proximal_mode",
        "rho",
        "proximal_mu",
        "alpha_decay",
        "alpha_decay_milestones",
    }

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        alpha = float(self.params.get("alpha", 0.9))
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("FedAsync alpha must be finite and in (0,1]")
        mode = self.params.get("proximal_mode", "paper_loss")
        if mode not in {"official_contraction", "paper_loss", "none"}:
            raise ValueError("Unknown FedAsync proximal_mode")
        rho = float(self.params.get("rho", 1e-5))
        mu = float(self.params.get("proximal_mu", 0.1))
        decay = float(self.params.get("alpha_decay", 0.5))
        if not math.isfinite(rho) or not 0 <= rho <= 1:
            raise ValueError("Author FedAsync rho=learning_rate*proximal_mu must be in [0,1]")
        if not math.isfinite(mu) or mu < 0:
            raise ValueError("FedAsync proximal_mu must be finite and non-negative")
        if not math.isfinite(decay) or not 0 < decay <= 1:
            raise ValueError("FedAsync alpha_decay must be in (0,1]")
        milestones = self.params.get("alpha_decay_milestones", [])
        if any(not isinstance(t, int) or t < 0 for t in milestones) or len(set(milestones)) != len(
            milestones
        ):
            raise ValueError("FedAsync alpha decay milestones must be unique non-negative integers")
        staleness_weight(0, self.params.get("staleness", {}))

    def local_loss(self, base_loss: Any, model: Any, batch: Any, context: Mapping[str, Any]) -> Any:
        if self.params.get("proximal_mode", "paper_loss") != "paper_loss":
            return base_loss
        return _proximal_loss(base_loss, model, context, float(self.params.get("proximal_mu", 0.1)))

    def transform_gradients(self, model: Any, context: Mapping[str, Any]) -> None:
        """Author async_fl: contract weights AFTER backward, BEFORE optimizer.step.

        rho here is their CLI rho (=gamma*paper_rho), NOT a loss coefficient.
        Using AdamW instead of their SGD is an explicit VLM optimizer adaptation.
        """
        if self.params.get("proximal_mode", "paper_loss") != "official_contraction":
            return
        import torch

        rho = float(self.params.get("rho", 1e-5))
        anchor = context.setdefault("fedasync_contraction_anchor", {})
        with torch.no_grad():
            for name, parameter in model.named_federated_parameters():
                if name not in anchor:
                    anchor[name] = (
                        context["base_state"][name]
                        .detach()
                        .to(parameter.device, dtype=parameter.dtype)
                    )
                parameter.mul_(1.0 - rho).add_(anchor[name], alpha=rho)

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        stale = server_context.version - update.base_version
        # Author epoch is zero-based and increments per applied update.
        decays = sum(
            server_context.version >= t for t in self.params.get("alpha_decay_milestones", [])
        )
        alpha = (
            float(self.params.get("alpha", 0.9))
            * float(self.params.get("alpha_decay", 0.5)) ** decays
            * staleness_weight(stale, self.params.get("staleness", {}))
        )
        state = add_scaled(
            server_context.global_state,
            subtract(update.local_state, server_context.global_state),
            alpha,
        )
        return [
            ServerMutation(
                state,
                alpha,
                [update.update_id],
                metadata={
                    "staleness": stale,
                    "alpha_t": alpha,
                    "proximal_mode": self.params.get("proximal_mode", "paper_loss"),
                    "author_source_commit": "912ecf33e0fb811ac8551c4f3cdab344cdedfb53",
                },
            )
        ]


@register_method("fedbuff")
class FedBuff(Method):
    name = "fedbuff"
    capabilities = MethodCapabilities(
        mode="asynchronous", requires_staleness=True, requires_buffer=True
    )
    allowed_params = {
        "buffer_size",
        "staleness",
        "staleness_weighting",
        "flush_last_buffer",
        "server_lr",
        "weighting",
    }

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.buffer: list[Update] = []
        self.buffer_aggregations = 0
        self.occupancy_sum = 0
        self.arrivals = 0
        if int(self.params.get("buffer_size", 5)) <= 0:
            raise ValueError("FedBuff buffer_size must be positive")
        lr = float(self.params.get("server_lr", 1.0))
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("FedBuff server_lr must be finite and positive")
        if self.params.get("weighting", "sample_size") not in {"uniform", "sample_size"}:
            raise ValueError("FedBuff weighting must be uniform or sample_size")
        staleness_weight(0, self.params.get("staleness", {}))

    def _flush(self, context: ServerContext, flush_time: float) -> ServerMutation:
        updates, self.buffer = self.buffer, []
        weights = []
        for update in updates:
            weights.append(
                float(update.num_samples)
                if self.params.get("weighting", "sample_size") == "sample_size"
                else 1.0
            )
        # Decay the displacement, not normalized weights: with K=1 normalization
        # would cancel all staleness handling. Original Algorithm 1 has no decay.
        deltas = [
            map_state(
                u.delta,
                lambda value, u=u: (
                    value
                    * (
                        staleness_weight(
                            context.version - u.base_version, self.params.get("staleness", {})
                        )
                        if self.params.get("staleness_weighting", False)
                        else 1.0
                    )
                ),
            )
            for u in updates
        ]
        delta = weighted_mean_states(deltas, weights)
        lr = float(self.params.get("server_lr", 1.0))
        self.buffer_aggregations += 1
        return ServerMutation(
            add_scaled(context.global_state, delta, lr),
            lr,
            [u.update_id for u in updates],
            metadata={
                "buffer_occupancy": len(updates),
                "weighting": self.params.get("weighting", "sample_size"),
                "server_lr": lr,
                "buffer_aggregation": self.buffer_aggregations,
                "mean_buffer_waiting_time": sum(
                    flush_time - update.arrival_time for update in updates
                )
                / len(updates),
            },
        )

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        self.buffer.append(update)
        self.arrivals += 1
        self.occupancy_sum += len(self.buffer)
        if len(self.buffer) >= int(self.params.get("buffer_size", 5)):
            return [self._flush(server_context, update.arrival_time)]
        return []

    def on_finish(self, server_context: ServerContext) -> list[ServerMutation]:
        if self.buffer and bool(self.params.get("flush_last_buffer", True)):
            return [self._flush(server_context, max(update.arrival_time for update in self.buffer))]
        if self.buffer:
            raise RuntimeError("FedBuff residual updates remain while flush_last_buffer=false")
        return []

    def state_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "buffer": self.buffer,
            "buffer_aggregations": self.buffer_aggregations,
            "occupancy_sum": self.occupancy_sum,
            "arrivals": self.arrivals,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self.buffer = list(state["buffer"])
        self.buffer_aggregations = int(state["buffer_aggregations"])
        self.occupancy_sum = int(state["occupancy_sum"])
        self.arrivals = int(state["arrivals"])
