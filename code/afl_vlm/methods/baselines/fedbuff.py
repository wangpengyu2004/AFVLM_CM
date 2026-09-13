"""FedBuff baseline with explicit arrival buffer and tail policy."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from afl_vlm.aggregation.fedbuff import apply_buffer
from afl_vlm.federation.types import ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method


@register_method("fedbuff")
class FedBuffMethod(Method):
    name = "fedbuff"
    allowed_params = {"server_lr", "buffer_size", "flush_tail", "sample_weighted"}

    def __init__(self, params=None) -> None:
        super().__init__(params)
        self.server_lr = float(self.params.get("server_lr", 0.5))
        self.buffer_size = int(self.params.get("buffer_size", 2))
        self.flush_tail = bool(self.params.get("flush_tail", True))
        self.sample_weighted = bool(self.params.get("sample_weighted", False))
        if self.buffer_size <= 0:
            raise ValueError("FedBuff buffer_size must be positive")
        self.buffer: list[Update] = []

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        self.buffer.append(copy.deepcopy(update))
        if len(self.buffer) < self.buffer_size:
            return []
        pending, self.buffer = self.buffer[: self.buffer_size], self.buffer[self.buffer_size :]
        return [
            apply_buffer(
                server_context.global_state,
                pending,
                self.server_lr,
                sample_weighted=self.sample_weighted,
            )
        ]

    def on_finish(self, server_context: ServerContext) -> list[ServerMutation]:
        if not self.buffer or not self.flush_tail:
            self.buffer.clear()
            return []
        pending, self.buffer = self.buffer, []
        mutation = apply_buffer(
            server_context.global_state,
            pending,
            self.server_lr,
            sample_weighted=self.sample_weighted,
        )
        mutation.metadata["tail_flush"] = True
        return [mutation]

    def state_dict(self) -> dict[str, Any]:
        return {"params": copy.deepcopy(self.params), "buffer": copy.deepcopy(self.buffer)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self.buffer = copy.deepcopy(list(state.get("buffer", [])))
