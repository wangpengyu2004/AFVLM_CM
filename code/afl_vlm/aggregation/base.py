"""Aggregation primitive contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import ServerMutation, Update


class Aggregator(ABC):
    @abstractmethod
    def apply(
        self, global_state: Mapping[str, Any], update: Update, context: Mapping[str, Any]
    ) -> ServerMutation:
        """Return a new immutable trainable state and application metadata."""
