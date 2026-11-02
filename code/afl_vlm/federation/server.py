"""Versioned server applying only method-produced trainable-state mutations."""

from __future__ import annotations

from typing import Any

from afl_vlm.federation.state_store import StateStore
from afl_vlm.federation.types import ApplyResult, ServerContext, Update
from afl_vlm.models.base import clone_state


class FederatedServer:
    def __init__(
        self,
        model: Any | None,
        method: Any,
        expected_clients: int,
        initial_state: dict[str, Any] | None = None,
    ) -> None:
        self.model, self.method, self.expected_clients = model, method, expected_clients
        self.version = 0
        if initial_state is None:
            if model is None:
                raise ValueError("A model or an explicit initial_state is required")
            initial_state = model.snapshot_trainable()
        self.state_store = StateStore(initial_state)
        self.received_updates = 0
        self.accepted_updates = 0

    @property
    def state(self) -> dict[str, Any]:
        return self.state_store.get(self.version)

    def context(self) -> ServerContext:
        return ServerContext(self.state, self.version, self.expected_clients)

    def initialize_from_gradients(self) -> ApplyResult | None:
        """Apply an optional pre-Plan bootstrap without inventing client uploads."""
        if self.version != 0 or self.received_updates != 0:
            raise RuntimeError("Server gradient initialization must precede all Plan updates")
        mutation = self.method.initial_server_update(self.context())
        if mutation is None:
            return None
        if mutation.contributing_update_ids:
            raise ValueError("Server initialization cannot count new client uploads")
        return self._apply(mutation, self.version, self.version)

    def initialize_from_calibration(self) -> None:
        """Install an opt-in common starting point without inventing a Plan upload/version."""
        if self.version != 0 or self.received_updates != 0:
            raise RuntimeError("Client calibration must precede all Plan updates")
        state = self.method.calibration_initial_state()
        if state is None:
            return
        if set(state) != set(self.state):
            raise ValueError("Calibrated initialization changed the federated parameter schema")
        candidate = StateStore(state)
        if self.model is not None:
            self.model.load_trainable(clone_state(state))
        self.state_store = candidate

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
                    {"buffered": False, "rejection_reason": update.metadata["rejection_reason"]}
                    if "rejection_reason" in update.metadata
                    else {"buffered": True},
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

    def resolve_job(self, client_id: str, local_round: int, outcome: str,
                    virtual_time: float | None = None) -> dict:
        """Only the scheduler may confirm a known job will never be retried."""
        if not self.method.capabilities.requires_job_resolution:
            return {}
        return self.method.on_job_resolved(client_id, local_round, outcome, virtual_time)

    def _apply(self, mutation: Any, receive_version: int, base_version: int) -> ApplyResult:
        if mutation.increment_version:
            if mutation.on_commit is not None:
                # Opt-in transaction: do not publish a version/history before a
                # required model copy succeeds. Existing algorithms keep their path.
                previous = self.state if self.model is not None else None
                if self.model is not None:
                    try:
                        self.model.load_trainable(clone_state(mutation.new_state))
                    except Exception:
                        self.model.load_trainable(previous)
                        raise
                next_version = self.version + 1
                self.state_store.put(next_version, mutation.new_state)
                self.version = next_version
            else:
                self.version += 1
                self.state_store.put(self.version, mutation.new_state)
                if self.model is not None:
                    self.model.load_trainable(clone_state(mutation.new_state))
        self.accepted_updates += (
            len(mutation.contributing_update_ids) if mutation.applied_weight > 0 else 0
        )
        if mutation.on_commit is not None:
            mutation.on_commit()
        return ApplyResult(
            True,
            float(mutation.applied_weight),
            receive_version,
            base_version,
            self.version,
            list(mutation.contributing_update_ids),
            dict(mutation.metadata),
        )
