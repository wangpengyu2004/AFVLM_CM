"""Private Router objectives; all supervision is detached from the VLM graph."""

from __future__ import annotations

from typing import Any


def pairwise_rank_loss(scores: Any, labels: Any, min_gap: float, valid: Any = True
                       ) -> tuple[Any, Any]:
    """Mean softplus(-(s_j-s_l)) for y_j-y_l > min_gap, one direction per pair.

    Near ties are excluded in normalized-label units. Empty pairs contribute an
    exactly zero, graph-connected loss. No pair sampling or RNG consumption.
    """
    import torch.nn.functional as functional

    labels = labels.detach()
    mask = ((labels[:, None] - labels[None, :]) > min_gap) & valid
    count = mask.sum()
    differences = scores[:, None] - scores[None, :]
    loss = (functional.softplus(-differences) * mask).sum() / count.clamp_min(1)
    return loss, count
