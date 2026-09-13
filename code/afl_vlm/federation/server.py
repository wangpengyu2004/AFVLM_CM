"""Versioned server applying only method-produced trainable-state mutations."""

from __future__ import annotations

from typing import Any

from afl_vlm.federation.state_store import StateStore
from afl_vlm.federation.types import ApplyResult, ServerContext, Update
from afl_vlm.models.base import clone_state


class FederatedServer:
    def __init__(self, model: Any, method: Any, expected_clients: int) -> None:
        self.model, self.method, self.expected_clients = model, method, expected_clients
        self.version = 0
        self.state_store = StateStore(model.snapshot_trainable())
        self.received_updates = 0
        self.accepted_updates = 0

    @property
    def state(self) -> dict[str, Any]:
        return self.state_store.get(self.version)

    def context(self) -> ServerContext:
        return ServerContext(self.state, self.version, self.expected_clients)

    def receive(self, update: Update) -> list[ApplyResult]:
        receive_version = self.version
        self.received_updates += 1
        mutations = self.method.on_arrival(update, self.context())
        if not mutations:
            return [
                ApplyResult(
                    False,
                    0.0,
                    receive_version,
                    update.base_version,
                    self.version,
                    [],
                    {"buffered": True},
                )
            ]
        return [self._apply(item, receive_version, update.base_version) for item in mutations]

    def finish(self) -> list[ApplyResult]:
        results = []
        while True:
            mutations = self.method.on_finish(self.context())
            if not mutations:
                return results
            results.extend(self._apply(item, self.version, self.version) for item in mutations)

    def _apply(self, mutation: Any, receive_version: int, base_version: int) -> ApplyResult:
        if mutation.increment_version:
            self.version += 1
            self.state_store.put(self.version, mutation.new_state)
            self.model.load_trainable(clone_state(mutation.new_state))
        self.accepted_updates += (
            len(mutation.contributing_update_ids) if mutation.applied_weight > 0 else 0
        )
        return ApplyResult(
            True,
            float(mutation.applied_weight),
            receive_version,
            base_version,
            self.version,
            list(mutation.contributing_update_ids),
            dict(mutation.metadata),
        )
