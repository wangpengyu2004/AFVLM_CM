"""AFVLM-CM: task-aware download memory and stale-conflict upload correction."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import ClientContext, ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import (
    LoRAState,
    add_scaled,
    clone_state,
    scale_state,
    state_cosine,
    state_dot,
    state_norm,
    subtract,
)


@register_method("afvlm_cm")
class AFVLMCMMethod(Method):
    """Correct task bias at download and stale conflicts at upload."""

    name = "afvlm_cm"
    allowed_params = {
        "server_lr",
        "staleness_exponent",
        "download_correction",
        "download_strength",
        "upload_correction",
        "memory_momentum",
    }

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.server_lr = float(self.params.get("server_lr", 0.5))
        self.staleness_exponent = float(self.params.get("staleness_exponent", 1.0))
        self.download_correction = bool(self.params.get("download_correction", True))
        self.download_strength = float(self.params.get("download_strength", 0.1))
        self.upload_correction = bool(self.params.get("upload_correction", True))
        self.memory_momentum = float(self.params.get("memory_momentum", 0.9))
        if self.server_lr <= 0:
            raise ValueError("afvlm_cm.server_lr must be positive")
        if self.staleness_exponent < 0:
            raise ValueError("afvlm_cm.staleness_exponent must be non-negative")
        if self.download_strength < 0:
            raise ValueError("afvlm_cm.download_strength must be non-negative")
        if not 0 <= self.memory_momentum < 1:
            raise ValueError("afvlm_cm.memory_momentum must be in [0, 1)")
        self.task_memory: dict[str, LoRAState] = {}

    def prepare_download(self, global_state: LoRAState, client_context: ClientContext) -> LoRAState:
        memory = self.task_memory.get(client_context.task)
        if not self.download_correction or memory is None:
            return clone_state(global_state)
        return add_scaled(global_state, memory, self.download_strength)

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        raw_delta = clone_state(update.delta)
        corrected_delta = clone_state(raw_delta)
        drift = subtract(server_context.global_state, update.base_state)
        drift_norm = state_norm(drift)
        conflict_cosine = state_cosine(raw_delta, drift)
        projection_removed = 0.0

        if self.upload_correction and drift_norm > 0 and conflict_cosine < 0:
            projection = state_dot(raw_delta, drift) / (drift_norm * drift_norm)
            corrected_delta = add_scaled(raw_delta, drift, -projection)
            projection_removed = abs(projection) * drift_norm

        raw_norm = state_norm(raw_delta)
        corrected_norm = state_norm(corrected_delta)
        if raw_norm > 0 and corrected_norm > raw_norm:
            corrected_delta = scale_state(corrected_delta, raw_norm / corrected_norm)
            corrected_norm = raw_norm

        previous = self.task_memory.get(update.task)
        if previous is None:
            self.task_memory[update.task] = clone_state(corrected_delta)
        else:
            memory = scale_state(previous, self.memory_momentum)
            self.task_memory[update.task] = add_scaled(
                memory, corrected_delta, 1.0 - self.memory_momentum
            )

        staleness = max(0, server_context.version - update.download_version)
        weight = self.server_lr * (staleness + 1) ** (-self.staleness_exponent)
        return [
            ServerMutation(
                new_state=add_scaled(server_context.global_state, corrected_delta, weight),
                applied_weight=weight,
                contributing_update_ids=[update.update_id],
                metadata={
                    "staleness": staleness,
                    "effective_weight": weight,
                    "raw_update_norm": raw_norm,
                    "corrected_update_norm": corrected_norm,
                    "conflict_cosine": conflict_cosine,
                    "projection_removed": projection_removed,
                },
            )
        ]

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["task_memory"] = copy.deepcopy(self.task_memory)
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self.task_memory = copy.deepcopy(dict(state.get("task_memory", {})))
