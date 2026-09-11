"""Delay model contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from random import Random


class DelayModel(ABC):
    @abstractmethod
    def duration(self, client_id: str, task: str, local_round: int, rng: Random) -> float:
        """Return virtual train plus network duration without sleeping."""
