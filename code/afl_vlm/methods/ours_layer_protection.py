"""Ours-only immutable finished-task layer selection and max age protection.

These helpers inspect small server-side sensitivity dictionaries, not model
weights or gradients. They never touch training, scheduling or other methods.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

_LAYER_MODULE = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")
_PROJECTIONS = {"q_proj", "k_proj", "v_proj", "o_proj"}
SELECTION_TOLERANCE = 1e-8


def layer_modules(modules: Sequence[str]) -> dict[int, dict[str, str]]:
    """Require complete Q/K/V/O sets; infer L, never hard-code observed layer IDs."""
    layers: dict[int, dict[str, str]] = {}
    for module in modules:
        match = _LAYER_MODULE.search(module)
        if match is None:
            raise ValueError(f"Selected-layer history requires decoder Q/K/V/O modules: {module}")
        layer, projection = int(match[1]), match[2]
        if projection in layers.setdefault(layer, {}):
            raise ValueError(f"Duplicate selected-layer projection: {layer}/{projection}")
        layers[layer][projection] = module
    if not layers or sorted(layers) != list(range(len(layers))):
        raise ValueError("Selected-layer history requires contiguous decoder layer IDs from zero")
    if any(set(projections) != _PROJECTIONS for projections in layers.values()):
        raise ValueError("Selected-layer history requires all four Q/K/V/O projections per layer")
    return layers


def select_layers(history: Mapping[str, float], floor: float, eps: float) -> dict[str, Any]:
    """Freeze corrected H; subtract its uniform floor ONLY for layer selection."""
    values = {name: float(value) for name, value in history.items()}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("Finished-task sensitivity must be finite and non-negative")
    layers = layer_modules(tuple(values))
    scores = {
        str(layer): math.fsum(
            max(values[group[projection]] - floor, 0.0) / 4
            for projection in sorted(_PROJECTIONS)
        )
        for layer, group in sorted(layers.items())
    }
    total = math.fsum(scores.values())
    threshold = 1.0 / len(layers)
    shares = {layer: value / total if total > eps else 0.0 for layer, value in scores.items()}
    mask = {layer: share > threshold + SELECTION_TOLERANCE for layer, share in shares.items()}
    selected = sorted(int(layer) for layer, active in mask.items() if active)
    return {
        "history": values,
        "layer_score": scores,
        "layer_share": shares,
        "layer_mask": mask,
        "selected_layers": selected,
        "layer_count": len(layers),
        "selection_floor": floor,
        "threshold": threshold,
        "selection_tolerance": SELECTION_TOLERANCE,
        "selection_reason": "above_task_mean"
        if selected
        else ("no_positive_excess" if total <= eps else "no_above_mean_layer"),
    }


def max_extra(
    snapshots: Mapping[str, Mapping[str, Any]],
    weights: Mapping[str, float],
    age_boosts: Mapping[str, float],
    modules: Sequence[str],
    module_layer: Mapping[str, str] | None = None,
) -> tuple[dict[str, float], dict[str, str | None]]:
    """P_m=max_q pi_q*b_q*mask_q(layer(m))*H_end[q,m], no sum/budget rescale."""
    if module_layer is None:
        layers = layer_modules(modules)
        module_layer = {
            module: str(layer) for layer, group in layers.items() for module in group.values()
        }
    extra = dict.fromkeys(modules, 0.0)
    owner: dict[str, str | None] = dict.fromkeys(modules)
    for task in sorted(snapshots):
        row = snapshots[task]
        scale = weights[task] * age_boosts[task]
        for module in modules:
            demand = (
                scale * row["history"][module] if row["layer_mask"][module_layer[module]] else 0.0
            )
            if not math.isfinite(demand) or demand < 0:
                raise ValueError("Finished-task extra protection must be finite and non-negative")
            if demand > extra[module]:
                extra[module] = demand
                owner[module] = task  # Exact positive ties use lexicographically first task.
    return extra, owner
