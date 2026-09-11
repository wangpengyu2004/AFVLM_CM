"""Paired E1/E2 mechanism metrics."""

from __future__ import annotations

from collections.abc import Mapping

from afl_vlm.evaluation.task_metrics import task_damage


def extra_harm(
    before: Mapping[str, float], stale_after: Mapping[str, float], fresh_after: Mapping[str, float]
) -> dict[str, float]:
    stale = task_damage(before, stale_after)
    fresh = task_damage(before, fresh_after)
    return {"damage_stale": stale, "damage_fresh": fresh, "extra_harm": stale - fresh}


def adaptation_gap(
    bias_before: Mapping[str, float],
    balanced_before: Mapping[str, float],
    bias_after: Mapping[str, float],
    balanced_after: Mapping[str, float],
) -> dict[str, float]:
    return {
        "initial_loss_gap": float(bias_before["loss"]) - float(balanced_before["loss"]),
        "post_train_loss_gap": float(bias_after["loss"]) - float(balanced_after["loss"]),
    }
