"""Exact nonzero spectrum of scaled BA using only a rank-sized QR core."""

from __future__ import annotations

import math
from typing import Any


def spectral_capacity(
    a: Any, b: Any, scaling: float, threshold: float, minimum: int = 1, maximum: int | None = None
) -> dict:
    import torch

    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise ValueError("Capacity estimation requires compatible LoRA A/B matrices")
    rank = a.shape[0]
    maximum = rank if maximum is None else min(rank, maximum)
    if not 0 < threshold <= 1 or not 1 <= minimum <= maximum:
        raise ValueError("Invalid spectral energy threshold or capacity bounds")
    if not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("Invalid LoRA scaling")
    with torch.no_grad():
        # BA = Q_b (R_b R_a.T) Q_a.T; orthonormal factors preserve singular values.
        _, ra = torch.linalg.qr(a.detach().double().T, mode="reduced")
        _, rb = torch.linalg.qr(b.detach().double(), mode="reduced")
        singular = torch.linalg.svdvals(rb @ ra.T) * scaling
        energy = singular.square()
        values = energy.cpu().tolist()
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Non-finite probe spectrum")
    total = math.fsum(values)
    running, needed = 0.0, minimum
    if total > 0:
        needed = len(values)
        for index, value in enumerate(values, 1):
            running += value
            if running >= threshold * total:
                needed = index
                break
    selected = min(maximum, max(minimum, needed))
    retained = math.fsum(values[:selected]) / total if total > 0 else 0.0
    return {
        "rank": selected,
        "candidate_rank": rank,
        "energy_threshold": threshold,
        "total_energy": total,
        "squared_singular_values": values,
        "retained_energy_fraction": retained,
        "energy_valid": total > 0,
        "threshold_met": total > 0 and retained + 1e-12 >= threshold,
        "bounded_by_maximum": needed > maximum,
    }
