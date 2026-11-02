"""Detached Router prediction/label ranking diagnostics, never an objective.

Observed-pool Spearman uses average ranks for exact ties. Top-k overlap uses
ceil(25% of observed components), plus top1. Constant vectors, n<2, invalid
supervision and ties across the relevant cutoff do not produce fake perfect
scores. No SciPy, random sampling, model forward, backward or collectives.
"""

from __future__ import annotations

import math
from typing import Any

FIELDS = (
    "spearman", "spearman_valid", "topk_overlap", "topk_valid",
    "top1_overlap", "top1_valid", "observed_components", "topk_k",
    "prediction_has_ties", "label_has_ties", "prediction_cutoff_tie", "label_cutoff_tie",
    "supervision_valid", "finite", "prediction_constant", "label_constant",
)


def average_ranks(values: Any) -> Any:
    """Zero-based average ranks in O(n log n), entirely on the input device."""
    import torch

    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    positions = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    edge = torch.ones(1, device=values.device, dtype=torch.bool)
    change = sorted_values[1:] != sorted_values[:-1]
    starts = torch.where(torch.cat((edge, change)), positions, 0).cummax(0).values
    ends = torch.where(torch.cat((change, edge)), positions, values.numel() - 1)
    ends = ends.flip(0).cummin(0).values.flip(0)
    return torch.empty_like(positions).scatter_(0, order, (starts + ends) * .5)


def ranking_tensor(predicted: Any, labels: Any, supervision_valid: Any) -> Any:
    """Return finite numeric fields; validity flags determine JSON nulls later.

    Called on EXISTING predictions before the Router optimizer update, with
    stored labels from the same pre-LoRA-update state. No new forward pass.
    """
    import torch

    with torch.no_grad():
        predicted, labels = predicted.detach().float(), labels.detach().float()
        supervision_valid = torch.as_tensor(
            supervision_valid, device=predicted.device, dtype=torch.bool
        )
        n = labels.numel()
        k = max(1, min(n - 1, math.ceil(n * .25)))
        finite = torch.isfinite(predicted).all() & torch.isfinite(labels).all()
        p, y = average_ranks(predicted), average_ranks(labels)
        p, y = p - p.mean(), y - y.mean()
        pn, yn = p.square().sum(), y.square().sum()
        valid = finite & supervision_valid & (n >= 2) & (pn > 0) & (yn > 0)
        correlation = ((p * y).sum() / (pn * yn).sqrt().clamp_min(1e-12)).clamp(-1, 1)
        prediction_order = torch.argsort(predicted, descending=True, stable=True)
        label_order = torch.argsort(labels, descending=True, stable=True)
        overlap = (prediction_order[:k, None] == label_order[None, :k]).any(1).float().mean()
        top1 = (prediction_order[0] == label_order[0]).float()
        if n >= 2:
            ptie = predicted[prediction_order[k - 1]] == predicted[prediction_order[k]]
            ytie = labels[label_order[k - 1]] == labels[label_order[k]]
            top1_tie = ((predicted[prediction_order[0]] == predicted[prediction_order[1]]) |
                        (labels[label_order[0]] == labels[label_order[1]]))
        else:
            ptie = ytie = top1_tie = predicted.new_tensor(True, dtype=torch.bool)
        p_sorted, y_sorted = predicted[prediction_order], labels[label_order]
        values = (
            torch.nan_to_num(correlation), valid, overlap, valid & ~ptie & ~ytie,
            top1, valid & ~top1_tie, predicted.new_tensor(n), predicted.new_tensor(k),
            (p_sorted[1:] == p_sorted[:-1]).any(), (y_sorted[1:] == y_sorted[:-1]).any(),
            ptie, ytie, supervision_valid, finite, pn == 0, yn == 0,
        )
        return torch.stack([v.float() for v in values])


def ranking_report(names: list[str], values: list[list[float]]) -> dict:
    """Convert one batched host transfer to small JSON diagnostics."""
    modules = {}
    for name, fields in zip(names, values, strict=True):
        row = dict(zip(FIELDS, fields, strict=True))
        for field in FIELDS:
            if field.endswith("valid") or field in {
                "finite", "prediction_has_ties", "label_has_ties",
                "prediction_cutoff_tie", "label_cutoff_tie",
                "prediction_constant", "label_constant",
            }:
                row[field] = bool(row[field])
        row["observed_components"] = int(row["observed_components"])
        row["topk_k"] = int(row["topk_k"])
        n = row["observed_components"]
        row["topk_chance_overlap"] = row["topk_k"] / n if n >= 2 else None
        row["top1_chance_overlap"] = 1 / n if n >= 2 else None
        for metric, flag in (("spearman", "spearman_valid"), ("topk_overlap", "topk_valid"),
                             ("top1_overlap", "top1_valid")):
            if not row[flag]:
                row[metric] = None
        modules[name] = row
    result = {"topk_fraction": .25, "prediction_time": "before_router_optimizer_step",
              "target": "same_pre_lora_step_signed_u", "modules": modules}
    for metric, flag in (("spearman", "spearman_valid"), ("topk_overlap", "topk_valid"),
                         ("top1_overlap", "top1_valid")):
        entries = [row[metric] for row in modules.values() if row[flag]]
        result[f"{metric}_mean"] = sum(entries) / len(entries) if entries else None
        result[f"{metric}_valid_modules"] = len(entries)
    return result
