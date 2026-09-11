"""Fixed-weight immediate additive baseline used by the mechanism experiments."""

from __future__ import annotations

from afl_vlm.aggregation.immediate import ImmediateAggregator
from afl_vlm.federation.types import ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method


@register_method("async_additive")
class AsyncAdditiveMethod(Method):
    name = "async_additive"
    branch_compatible = True
    allowed_params = {"server_lr"}

    def __init__(self, params=None) -> None:
        super().__init__(params)
        self.server_lr = float(self.params.get("server_lr", 0.5))
        self.aggregator = ImmediateAggregator()

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        return [
            self.aggregator.apply(server_context.global_state, update, {"weight": self.server_lr})
        ]
