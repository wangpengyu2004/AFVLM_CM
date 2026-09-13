"""FedBuff mean-update aggregation primitive.

The paper defines client pseudo-gradients as ``start - trained`` and subtracts
them at the server. This project stores updates as ``trained - start`` and adds
their mean, which is algebraically equivalent.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import ServerMutation, Update
from afl_vlm.models.base import add_scaled, mean_states, weighted_mean_states


def apply_buffer(
    global_state: Mapping[str, Any],
    updates: list[Update],
    server_lr: float,
    sample_weighted: bool = False,
) -> ServerMutation:
    if not updates:
        raise ValueError("FedBuff cannot apply an empty buffer")
    if sample_weighted:
        weights = [float(update.metadata.get("num_examples", 1.0)) for update in updates]
        mean_delta = weighted_mean_states((update.delta for update in updates), weights)
    else:
        weights = [1.0] * len(updates)
        mean_delta = mean_states(update.delta for update in updates)
    return ServerMutation(
        new_state=add_scaled(global_state, mean_delta, server_lr),
        applied_weight=server_lr,
        contributing_update_ids=[update.update_id for update in updates],
        metadata={
            "buffer_size_applied": len(updates),
            "aggregation_semantics": "sample_weighted_mean_delta"
            if sample_weighted
            else "mean_delta",
            "sample_weights": weights,
        },
    )
