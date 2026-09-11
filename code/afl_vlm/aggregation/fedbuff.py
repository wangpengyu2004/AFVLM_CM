"""FedBuff mean-update aggregation primitive.

The paper defines client pseudo-gradients as ``start - trained`` and subtracts
them at the server. This project stores updates as ``trained - start`` and adds
their mean, which is algebraically equivalent.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import ServerMutation, Update
from afl_vlm.models.base import add_scaled, mean_states


def apply_buffer(
    global_state: Mapping[str, Any], updates: list[Update], server_lr: float
) -> ServerMutation:
    if not updates:
        raise ValueError("FedBuff cannot apply an empty buffer")
    mean_delta = mean_states(update.delta for update in updates)
    return ServerMutation(
        new_state=add_scaled(global_state, mean_delta, server_lr),
        applied_weight=server_lr / len(updates),
        contributing_update_ids=[update.update_id for update in updates],
        metadata={"buffer_size_applied": len(updates), "aggregation_semantics": "mean_delta"},
    )
