"""Additive update with polynomial staleness decay."""

from __future__ import annotations

from afl_vlm.aggregation.immediate import ImmediateAggregator
from afl_vlm.federation.types import ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method


@register_method("staleness_decay")
class StalenessDecayMethod(Method):
    name = "staleness_decay"
    allowed_params = {"server_lr", "decay_exponent"}

    def __init__(self, params=None) -> None:
        super().__init__(params)
        self.server_lr = float(self.params.get("server_lr", 0.5))
        self.exponent = float(self.params.get("decay_exponent", 1.0))
        self.aggregator = ImmediateAggregator()

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        staleness = max(0, server_context.version - update.download_version)
        weight = self.server_lr * (staleness + 1) ** (-self.exponent)
        mutation = self.aggregator.apply(server_context.global_state, update, {"weight": weight})
        mutation.metadata.update({"staleness": staleness, "decay_exponent": self.exponent})
        return [mutation]
