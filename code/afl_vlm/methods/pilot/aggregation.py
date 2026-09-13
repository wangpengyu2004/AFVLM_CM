"""Pilot task-aware visual and personalized text-side aggregation."""

from __future__ import annotations

from typing import Any

from afl_vlm.models.base import clone_state, state_norm, subtract, weighted_mean_states


def pilot_aggregate(
    updates: list[Any], current: dict[str, Any], neighbors: int
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    result = clone_state(current)
    tasks = sorted({update.task for update in updates})
    text_keys = [key for key in result if "lora_" in key]
    client_keys = [key for key in result if "client_adapters." in key]
    task_keys = [key for key in result if "task_adapters." in key]

    # Pilot Eq. (10): sample-weighted task visual adapters within each task.
    for task in tasks:
        selected = [update for update in updates if update.task == task]
        weights = [update.num_samples for update in selected]
        for key in task_keys:
            if f"task_adapters.{task}." in key:
                result[key] = weighted_mean_states(
                    [{key: update.local_state[key]} for update in selected], weights
                )[key]

    # Shared CT-MoA/router/text state has a well-defined server reference.
    shared_keys = [key for key in result if key not in client_keys and key not in task_keys]
    for key in shared_keys:
        result[key] = weighted_mean_states(
            [{key: update.local_state[key]} for update in updates],
            [update.num_samples for update in updates],
        )[key]

    # Client visual adapters remain attached to their owner.
    for update in updates:
        marker = update.client_id.replace("/", "__")
        for key in client_keys:
            if f"client_adapters.{marker}." in key:
                result[key] = clone_state({key: update.local_state[key]})[key]

    # Pilot Eq. (11--12): nearest updates receive inverse-distance text weights.
    personalized: dict[str, dict[str, Any]] = {}
    for update in updates:
        candidates = []
        for other in updates:
            if other.client_id == update.client_id:
                continue
            difference = subtract(update.local_state, other.local_state)
            candidates.append((state_norm({key: difference[key] for key in text_keys}), other))
        distances = sorted(candidates, key=lambda item: (item[0], item[1].client_id))[:neighbors]
        peers = [update, *(item[1] for item in distances)]
        weights = [
            float(update.num_samples),
            *[1.0 / max(1e-12, item[0]) for item in distances],
        ]
        state = clone_state(result)
        for key in text_keys:
            state[key] = weighted_mean_states(
                [{key: peer.local_state[key]} for peer in peers], weights
            )[key]
        personalized[update.client_id] = state
    return result, personalized
