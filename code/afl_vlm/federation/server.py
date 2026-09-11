"""Versioned server that applies method-produced mutations."""

from __future__ import annotations

from typing import Any

from afl_vlm.federation.state_store import StateStore
from afl_vlm.federation.types import ApplyResult, ServerContext, Update
from afl_vlm.models.base import clone_state


class FederatedServer:
    def __init__(self, model: Any, method: Any, expected_clients: int) -> None:
        self.model = model
        self.method = method
        self.expected_clients = expected_clients
        self.version = 0
        self.state_store = StateStore(model.snapshot_trainable())
        self.received_updates = 0
        self.applied_updates = 0

    @property
    def state(self) -> dict[str, Any]:
        return self.state_store.get(self.version)

    def _context(self) -> ServerContext:
        return ServerContext(
            global_state=self.state,
            version=self.version,
            expected_clients=self.expected_clients,
        )

    def receive(self, update: Update) -> list[ApplyResult]:
        receive_version = self.version
        self.received_updates += 1
        mutations = self.method.on_arrival(update, self._context())
        if not mutations:
            return [
                ApplyResult(
                    applied=False,
                    applied_weight=0.0,
                    receive_version=receive_version,
                    training_version=update.download_version,
                    resulting_version=self.version,
                    contributing_update_ids=[],
                    metadata={"buffered": True},
                )
            ]
        return [
            self._apply(mutation, receive_version, update.download_version)
            for mutation in mutations
        ]

    def finish(self) -> list[ApplyResult]:
        results: list[ApplyResult] = []
        while True:
            mutations = self.method.on_finish(self._context())
            if not mutations:
                break
            for mutation in mutations:
                results.append(self._apply(mutation, self.version, self.version))
        return results

    def _apply(self, mutation: Any, receive_version: int, training_version: int) -> ApplyResult:
        self.version += 1
        self.state_store.put(self.version, mutation.new_state)
        self.model.load_trainable(clone_state(mutation.new_state))
        self.applied_updates += len(mutation.contributing_update_ids)
        return ApplyResult(
            applied=True,
            applied_weight=float(mutation.applied_weight),
            receive_version=receive_version,
            training_version=training_version,
            resulting_version=self.version,
            contributing_update_ids=list(mutation.contributing_update_ids),
            metadata=dict(mutation.metadata),
        )
