"""Reference, classical, and basic asynchronous federated optimizers."""

from __future__ import annotations

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


@register_method("local")
class LocalTraining(Method):
    name = "local"
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

    def local_loss(self, base_loss: Any, model: Any, batch: Any, context: Mapping[str, Any]) -> Any:
        mu = float(self.params.get("mu", 0.01))
        anchor = context["base_state"]
        penalty = None
        for name, parameter in model.named_federated_parameters():
            reference = anchor[name].to(parameter.device, dtype=parameter.dtype)
            value = (parameter - reference).pow(2).sum()
            penalty = value if penalty is None else penalty + value
        return base_loss + 0.5 * mu * penalty


@register_method("fedadam")
class FedAdam(_RoundAggregator):
    name = "fedadam"
    allowed_params = {"server_lr", "beta1", "beta2", "epsilon"}

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.m = None
        self.v = None
        self.step = 0

    def aggregate(self, updates: list[Update], server_context: ServerContext) -> ServerMutation:
        delta = weighted_mean_states([u.delta for u in updates], [u.num_samples for u in updates])
        if self.m is None:
            self.m, self.v = zeros_like(delta), zeros_like(delta)
        beta1, beta2 = float(self.params.get("beta1", 0.9)), float(self.params.get("beta2", 0.999))
        self.m = zip_states(self.m, delta, lambda m, g: beta1 * m + (1 - beta1) * g)
        self.v = zip_states(self.v, delta, lambda v, g: beta2 * v + (1 - beta2) * g * g)
        self.step += 1
        m_hat = map_state(self.m, lambda x: x / (1 - beta1**self.step))
        v_hat = map_state(self.v, lambda x: x / (1 - beta2**self.step))
        eps = float(self.params.get("epsilon", 1e-8))
        direction = zip_states(
            m_hat,
            v_hat,
            lambda m, v: m / (v.sqrt() + eps) if hasattr(v, "sqrt") else m / (v**0.5 + eps),
        )
        state = add_scaled(
            server_context.global_state, direction, float(self.params.get("server_lr", 0.01))
        )
        return ServerMutation(
            state, float(self.params.get("server_lr", 0.01)), [u.update_id for u in updates]
        )

    def state_dict(self) -> dict[str, Any]:
        return {"params": self.params, "m": self.m, "v": self.v, "step": self.step}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self.m, self.v, self.step = state["m"], state["v"], int(state["step"])


@register_method("fedasync")
class FedAsync(Method):
    name = "fedasync"
    capabilities = MethodCapabilities(mode="asynchronous", requires_staleness=True)
    allowed_params = {"alpha", "staleness"}

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        stale = server_context.version - update.base_version
        alpha = float(self.params.get("alpha", 0.5)) * staleness_weight(
            stale, self.params.get("staleness", {})
        )
        state = add_scaled(
            server_context.global_state,
            subtract(update.local_state, server_context.global_state),
            alpha,
        )
        return [ServerMutation(state, alpha, [update.update_id], metadata={"staleness": stale})]


@register_method("fedbuff")
class FedBuff(Method):
    name = "fedbuff"
    capabilities = MethodCapabilities(
        mode="asynchronous", requires_staleness=True, requires_buffer=True
    )
    allowed_params = {"buffer_size", "staleness", "staleness_weighting", "flush_last_buffer"}

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.buffer: list[Update] = []
        self.buffer_aggregations = 0
        self.occupancy_sum = 0
        self.arrivals = 0

    def _flush(self, context: ServerContext, flush_time: float) -> ServerMutation:
        updates, self.buffer = self.buffer, []
        weights = []
        for update in updates:
            weight = float(update.num_samples)
            if self.params.get("staleness_weighting", False):
                weight *= staleness_weight(
                    context.version - update.base_version, self.params.get("staleness", {})
                )
            weights.append(weight)
        delta = weighted_mean_states([u.delta for u in updates], weights)
        self.buffer_aggregations += 1
        return ServerMutation(
            add_scaled(context.global_state, delta, 1.0),
            1.0,
            [u.update_id for u in updates],
            metadata={
                "buffer_occupancy": len(updates),
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
