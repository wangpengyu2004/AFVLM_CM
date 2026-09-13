"""Unified samples and task-adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Sample:
    id: str
    task_name: str
    image: str | None
    instruction: str
    answer: str
    split: str
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskAdapter(ABC):
    task_key: str

    @abstractmethod
    def load_file(self, path: str | Path, split: str = "train") -> list[Sample]:
        """Load a normalized split from an explicit data file."""

    @abstractmethod
    def load_split(self, split: str) -> list[Sample]:
        """Load a normalized split."""

    @abstractmethod
    def format_sample(self, sample: Sample) -> tuple[list[dict[str, Any]], str | None]:
        """Return chat messages and the image path/URL."""

    @abstractmethod
    def collate(self, samples: list[Sample], model_adapter: Any) -> Any:
        """Create a model-specific batch."""

    @abstractmethod
    def metric(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        """Compute task-native metrics without cross-task averaging."""

    @abstractmethod
    def probe_loss(self, model: Any, sample_ids: list[str]) -> float:
        """Compute teacher-forced probe loss."""

    @abstractmethod
    def samples_by_id(self, sample_ids: list[str]) -> list[Sample]:
        """Resolve stable sample identifiers."""
