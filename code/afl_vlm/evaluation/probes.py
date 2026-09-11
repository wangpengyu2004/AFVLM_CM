"""Probe evaluation without mixing incomparable task units."""

from __future__ import annotations

from typing import Any


def evaluate_tasks(
    model: Any, tasks: dict[str, Any], probe_ids: dict[str, list[str]], mode: str = "probe"
) -> dict[str, dict[str, float]]:
    return {
        task_key: model.evaluate(task, probe_ids[task_key], mode)
        for task_key, task in tasks.items()
    }
