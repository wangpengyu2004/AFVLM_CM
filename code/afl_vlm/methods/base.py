"""Full method lifecycle spanning download, local loss, upload, and arrival."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import (
    ClientContext,
    ServerContext,
    ServerMutation,
    Update,
)
from afl_vlm.models.base import LoRAState, clone_state


class Method(ABC):
    name = "base"
    branch_compatible = False
    allowed_params: set[str] = set()

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        self.params = dict(params or {})
        unknown = set(self.params) - self.allowed_params
        if unknown:
            raise ValueError(f"Unknown parameter(s) for {self.name}: {sorted(unknown)}")

    def prepare_download(self, global_state: LoRAState, client_context: ClientContext) -> LoRAState:
        return clone_state(global_state)

    def local_loss(self, base_loss: Any, model: Any, batch: Any, context: Mapping[str, Any]) -> Any:
        return base_loss

    def prepare_upload(self, update: Update, context: ClientContext) -> Update:
        return update

    @abstractmethod
    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        raise NotImplementedError

    def on_finish(self, server_context: ServerContext) -> list[ServerMutation]:
        return []

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy({"params": self.params})

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.params = copy.deepcopy(dict(state["params"]))
