"""Current trainable server state; active-job snapshots live in the runner."""

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
        self._states = {version: clone_state(state)}

    def get(self, version: int) -> LoRAState:
        try:
            return clone_state(self._states[version])
        except KeyError as exc:
            raise KeyError(f"Unknown state version: {version}") from exc

    def hash(self, version: int) -> str:
        return state_hash(self._states[version])
