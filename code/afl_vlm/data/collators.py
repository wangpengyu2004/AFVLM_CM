"""Small collator helpers retained as a stable extension point."""

from __future__ import annotations

from typing import Any

from afl_vlm.data.base import Sample


def collate_with_task_adapter(samples: list[Sample], task_adapter: Any, model_adapter: Any) -> Any:
    return task_adapter.collate(samples, model_adapter)
