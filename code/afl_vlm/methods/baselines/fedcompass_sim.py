"""Controlled-simulator realization of FedCompass's grouped aggregation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from afl_vlm.aggregation.fedbuff import apply_buffer
from afl_vlm.federation.types import ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method


@register_method("fedcompass_sim")
class FedCompassSimMethod(Method):
    """Computing-power-aware workload assignment plus grouped aggregation.

    This implements the two protocol mechanisms needed by the virtual simulator;
    it is intentionally named ``_sim`` because it does not reproduce APPFL's
    online speed estimator or deployment layer.
    """

    name = "fedcompass_sim"
    schedule_mode = "fedcompass"
    allowed_params = {
        "server_lr",
        "buffer_size",
        "min_local_steps",
        "max_local_steps",
        "flush_tail",
        "sample_weighted",
    }

    def __init__(self, params=None) -> None:
        super().__init__(params)
        self.server_lr = float(self.params.get("server_lr", 1.0))
        self.buffer_size = int(self.params.get("buffer_size", 4))
        self.min_local_steps = int(self.params.get("min_local_steps", 1))
        self.max_local_steps = int(self.params.get("max_local_steps", 5))
        self.flush_tail = bool(self.params.get("flush_tail", True))
        self.sample_weighted = bool(self.params.get("sample_weighted", True))
        if self.buffer_size <= 0:
            raise ValueError("FedCompass buffer_size must be positive")
        if self.min_local_steps <= 0 or self.max_local_steps < self.min_local_steps:
            raise ValueError("FedCompass needs 0 < min_local_steps <= max_local_steps")
        self.buffer: list[Update] = []

    def _apply(self, state: Mapping[str, Any], pending: list[Update]) -> ServerMutation:
        mutation = apply_buffer(
            state,
            pending,
            self.server_lr,
            sample_weighted=self.sample_weighted,
        )
        mutation.metadata["protocol"] = "fedcompass_sim"
        return mutation

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        self.buffer.append(copy.deepcopy(update))
        if len(self.buffer) < self.buffer_size:
            return []
        pending, self.buffer = self.buffer[: self.buffer_size], self.buffer[self.buffer_size :]
        return [self._apply(server_context.global_state, pending)]

    def on_finish(self, server_context: ServerContext) -> list[ServerMutation]:
        if not self.buffer or not self.flush_tail:
            self.buffer.clear()
            return []
        pending, self.buffer = self.buffer, []
        mutation = self._apply(server_context.global_state, pending)
        mutation.metadata["tail_flush"] = True
        return [mutation]

    def state_dict(self) -> dict[str, Any]:
        return {"params": copy.deepcopy(self.params), "buffer": copy.deepcopy(self.buffer)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self.buffer = copy.deepcopy(list(state.get("buffer", [])))
