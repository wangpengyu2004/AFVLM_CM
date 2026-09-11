"""FedAsync model-interpolation primitive."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from afl_vlm.aggregation.base import Aggregator
from afl_vlm.federation.types import ServerMutation, Update
from afl_vlm.models.base import add_scaled, subtract


class FedAsyncAggregator(Aggregator):
    def apply(
        self, global_state: Mapping[str, Any], update: Update, context: Mapping[str, Any]
    ) -> ServerMutation:
        alpha = float(context["alpha"])
        local_model = add_scaled(update.base_state, update.delta, 1.0)
        interpolation_delta = subtract(local_model, global_state)
        return ServerMutation(
            new_state=add_scaled(global_state, interpolation_delta, alpha),
            applied_weight=alpha,
            contributing_update_ids=[update.update_id],
            metadata={"aggregation_semantics": "model_interpolation"},
        )
