"""Synchronous FedAvg reference grouped by local round."""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from afl_vlm.aggregation.fedbuff import apply_buffer
from afl_vlm.federation.types import ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method


@register_method("fedavg_sync")
class FedAvgSyncMethod(Method):
    name = "fedavg_sync"
    allowed_params = {"server_lr"}

    def __init__(self, params=None) -> None:
        super().__init__(params)
        self.server_lr = float(self.params.get("server_lr", 1.0))
        self.rounds: dict[int, list[Update]] = defaultdict(list)

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        self.rounds[update.local_round].append(copy.deepcopy(update))
        pending = self.rounds[update.local_round]
        if len(pending) < server_context.expected_clients:
            return []
        del self.rounds[update.local_round]
        mutation = apply_buffer(server_context.global_state, pending, self.server_lr)
        mutation.metadata["synchronous_round"] = update.local_round
        return [mutation]

    def on_finish(self, server_context: ServerContext) -> list[ServerMutation]:
        if self.rounds:
            incomplete = {round_id: len(items) for round_id, items in self.rounds.items()}
            raise RuntimeError(f"Incomplete synchronous rounds at finish: {incomplete}")
        return []

    def state_dict(self) -> dict[str, Any]:
        return {"params": copy.deepcopy(self.params), "rounds": copy.deepcopy(dict(self.rounds))}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self.rounds = defaultdict(list, copy.deepcopy(dict(state.get("rounds", {}))))
