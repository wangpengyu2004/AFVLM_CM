"""Task-local metric utilities."""

from __future__ import annotations

from collections.abc import Mapping


def preferred_score(metrics: Mapping[str, float]) -> float | None:
    for key in ("score", "official_accuracy", "accuracy", "exact", "auc"):
        if key in metrics:
            return float(metrics[key])
    return None


def task_damage(before: Mapping[str, float], after: Mapping[str, float]) -> float:
    before_score = preferred_score(before)
    after_score = preferred_score(after)
    if before_score is not None and after_score is not None:
        return before_score - after_score
    if "loss" not in before or "loss" not in after:
        raise ValueError("Metrics contain neither a comparable score nor loss")
    return float(after["loss"]) - float(before["loss"])
