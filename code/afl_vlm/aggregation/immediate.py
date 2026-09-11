"""Immediate additive aggregation primitive."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from afl_vlm.aggregation.base import Aggregator
from afl_vlm.federation.types import ServerMutation, Update
from afl_vlm.models.base import add_scaled


class ImmediateAggregator(Aggregator):
    def apply(
        self, global_state: Mapping[str, Any], update: Update, context: Mapping[str, Any]
    ) -> ServerMutation:
        weight = float(context.get("weight", 1.0))
        return ServerMutation(
            new_state=add_scaled(global_state, update.delta, weight),
            applied_weight=weight,
            contributing_update_ids=[update.update_id],
        )
