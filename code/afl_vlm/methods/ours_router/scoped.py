"""Selection overlap switches the ORIGINAL sensitivity rule on/off per component."""

from __future__ import annotations

import copy
import math
from typing import Any

from afl_vlm.methods.ours_hc import payload
from afl_vlm.models.base import clone_state


def validate(update: Any, modules: dict) -> None:
    """Accept any unique index order, validating a reordered view only when needed."""
    selected = update.metadata.get("hc_selection", {})
    if set(selected) != set(modules):
        raise ValueError("Scoped aggregation requires all module selection indices")
    reordered = {}
    for name, item in modules.items():
        indices = selected[name]
        if (
            not isinstance(indices, list)
            or any(type(j) is not int or not 0 <= j < item["rank"] for j in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError("Invalid unique global selection indices")
        if indices != sorted(indices):
            reordered[name] = sorted(range(len(indices)), key=indices.__getitem__)
    if not reordered:
        payload.validate(update, modules)
        return
    view = copy.copy(update)
    view.metadata = dict(update.metadata, hc_selection=copy.deepcopy(selected))
    view.delta = dict(update.delta)
    for name, order in reordered.items():
        view.metadata["hc_selection"][name] = sorted(selected[name])
        for side, key in (("A", "a_name"), ("B", "b_name")):
            value_name = modules[name][key]
            view.delta[value_name] = payload.select(update.delta[value_name], order, side)
    payload.validate(view, modules)


def rank_alpha(
    c: float,
    rho: float,
    frequency: list[float],
    omega: list[float],
    *,
    sensitivity_mask: list[bool],
    eta: float,
    lam: float,
    eps: float,
    maximum: float | None = None,
) -> list[float]:
    if (
        not all(math.isfinite(v) and v >= 0 for v in [c, rho, eta, lam, eps, *frequency, *omega])
        or rho > 1
        or not 0 < eta <= 1
        or lam < 0
        or eps <= 0
        or len(frequency) != len(omega)
        or len(sensitivity_mask) != len(omega)
        or any(type(value) is not bool for value in sensitivity_mask)
        or (maximum is not None and not 0 < maximum <= 1)
    ):
        raise ValueError("Invalid scoped precision inputs")
    result = []
    for weight, protection, matched in zip(frequency, omega, sensitivity_mask, strict=True):
        strength = rho * weight * c
        value = eta * strength / (strength + lam * protection + eps) if matched else eta * rho
        result.append(value if maximum is None else min(maximum, value))
    return result


def precision(
    registry: Any,
    task: str,
    selected: dict,
    history: dict,
    weights: dict,
    finished: dict,
    demands: dict,
    strength: float,
) -> dict:
    """Keep original module base H sum; overlap is ONLY a mechanism switch.

    Completed-task age demands follow each task's own frozen indices. They
    never add an age boost at positions outside that task's frozen support.
    """
    active, exits = registry.task_unions()
    result = {}
    for module, indices in selected.items():
        competition = [
            j for j in indices if any(j in active[q][module] for q in active if q != task)
        ]
        exit_hits = [j for j in indices if any(j in exits[q][module] for q in exits if q != task)]
        matched = set(competition) | set(exit_hits)
        sources, base, age = {}, [0.0] * len(indices), [0.0] * len(indices)
        original_base = strength * sum(
            weight * (finished[q]["history"][module] if q in finished else history[q][module])
            for q, weight in weights.items()
        )
        base = [original_base if j in matched else 0.0 for j in indices]
        for q, weight in weights.items():
            h = finished[q]["history"][module] if q in finished else history[q][module]
            scoped_base = [strength * weight * h if j in matched else 0.0 for j in indices]
            scoped_age = [
                strength * demands.get(q, {}).get(module, 0.0) if j in exits[q][module] else 0.0
                for j in indices
            ]
            age = [a + b for a, b in zip(age, scoped_age, strict=True)]
            if any(scoped_base) or any(scoped_age):
                sources[q] = {
                    "base_applied_indices": [j for j in indices if j in matched],
                    "frozen_indices_hit": [j for j in indices if j in exits[q][module]],
                    "base": scoped_base,
                    "age": scoped_age,
                }
        result[module] = {
            "selected_indices": list(indices),
            "global_rank": registry.ranks[module],
            "competition_indices": competition,
            "competition_fraction": len(competition) / len(indices) if indices else 0.0,
            "exit_history_indices": exit_hits,
            "exit_history_fraction": len(exit_hits) / len(indices) if indices else 0.0,
            "sensitivity_indices": [j for j in indices if j in matched],
            "sensitivity_mask": [j in matched for j in indices],
            "original_module_omega_base": original_base,
            "omega_base": base,
            "omega_age": age,
            "omega": [a + b for a, b in zip(base, age, strict=True)],
            "protection_sources": sources,
        }
    return result


def fuse(current: dict, update: Any, alpha: dict[str, list[float]]) -> dict:
    """One alpha per selected ORIGINAL index, shared by its A row and B column."""
    result = clone_state(current)
    for module, item in update.metadata["lora_metadata"].items():
        indices, values = update.metadata["hc_selection"][module], alpha[module]
        if not indices:
            continue
        for side, key in (("A", "a_name"), ("B", "b_name")):
            name = item[key]
            if hasattr(result[name], "index_add_"):
                import torch

                index = torch.tensor(indices, device=result[name].device, dtype=torch.long)
                weights = result[name].new_tensor(values)
                delta = update.delta[name].to(result[name])
                result[name].index_add_(
                    0 if side == "A" else 1,
                    index,
                    delta * (weights[:, None] if side == "A" else weights[None, :]),
                )
            elif side == "A":
                for offset, index in enumerate(indices):
                    result[name][index] = [
                        x + values[offset] * y
                        for x, y in zip(
                            result[name][index], update.delta[name][offset], strict=True
                        )
                    ]
            else:
                for row in range(item["out_features"]):
                    for offset, index in enumerate(indices):
                        result[name][row][index] += values[offset] * update.delta[name][row][offset]
    return result
