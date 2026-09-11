"""FedAsync baseline using paper-defined model interpolation."""

from __future__ import annotations

from afl_vlm.aggregation.fedasync import FedAsyncAggregator
from afl_vlm.federation.types import ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method


@register_method("fedasync")
class FedAsyncMethod(Method):
    name = "fedasync"
    allowed_params = {"alpha", "decay_exponent"}

    def __init__(self, params=None) -> None:
        super().__init__(params)
        self.alpha = float(self.params.get("alpha", 0.5))
        self.exponent = float(self.params.get("decay_exponent", 0.5))
        self.aggregator = FedAsyncAggregator()

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        staleness = max(0, server_context.version - update.download_version)
        alpha_t = self.alpha * (staleness + 1) ** (-self.exponent)
        mutation = self.aggregator.apply(server_context.global_state, update, {"alpha": alpha_t})
        mutation.metadata.update({"staleness": staleness, "alpha_t": alpha_t})
        return [mutation]
