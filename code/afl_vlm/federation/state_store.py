"""In-memory versioned LoRA state store with deterministic hashes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from afl_vlm.models.base import LoRAState, clone_state, state_hash


class StateStore:
    def __init__(self, initial_state: Mapping[str, Any]) -> None:
        self._states: dict[int, LoRAState] = {0: clone_state(initial_state)}

    def put(self, version: int, state: Mapping[str, Any]) -> None:
        if version in self._states:
            raise ValueError(f"State version already exists: {version}")
        self._states[version] = clone_state(state)

    def get(self, version: int) -> LoRAState:
        try:
            return clone_state(self._states[version])
        except KeyError as exc:
            raise KeyError(f"Unknown state version: {version}") from exc

    def hash(self, version: int) -> str:
        return state_hash(self._states[version])


class BranchRestoreGuard:
    """Restore model state even if a diagnostic branch raises."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.state = model.snapshot_trainable()
        self.before_hash = state_hash(self.state)

    def __enter__(self) -> BranchRestoreGuard:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.model.load_trainable(self.state)
        after_hash = state_hash(self.model.snapshot_trainable())
        if after_hash != self.before_hash:
            raise RuntimeError("Branch state restoration hash mismatch")
