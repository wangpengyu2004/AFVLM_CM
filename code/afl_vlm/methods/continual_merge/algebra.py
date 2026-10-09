"""Functional LoRA algebra without constructing a dense backbone-sized matrix.

QR spans the columns of all B and all A.T factors. Every BA product is then
represented exactly in the same small orthonormal coordinates. Only the final
rank-r truncation is approximate. PyTorch is imported lazily for static tooling.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_PARAMETER = re.compile(r"^(.*)\.lora_([AB])(?:\.([^.]+))?\.weight$")


def lora_pairs(state: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    groups: dict[str, dict[str, str]] = {}
    for key in state:
        match = _PARAMETER.fullmatch(key)
        if match is None:
            raise ValueError(f"Continual merging requires LoRA-only A/B weights: {key}")
        module, side, adapter = match.groups()
        identity = f"{module}:{adapter or 'default'}"
        if side in groups.setdefault(identity, {}):
            raise ValueError(f"Duplicate {side} factor: {identity}")
        groups[identity][side] = key
    if not groups or any(set(group) != {"A", "B"} for group in groups.values()):
        raise ValueError("Continual merging requires complete, nonempty LoRA A/B pairs")
    return {name: (group["A"], group["B"]) for name, group in sorted(groups.items())}


def compact_coordinates(
    factors: Sequence[tuple[Any, Any]],
    device: Any,
) -> tuple[Any, Any, list[Any]]:
    """Return Qo, Qi and cores for factors (A, B); Qo @ core @ Qi.T == B @ A."""
    import torch

    converted = [
        (
            a.detach().to(device=device, dtype=torch.float32),
            b.detach().to(device=device, dtype=torch.float32),
        )
        for a, b in factors
    ]
    rank, inputs = converted[0][0].shape
    outputs = converted[0][1].shape[0]
    for a, b in converted:
        if a.shape != (rank, inputs) or b.shape != (outputs, rank):
            raise ValueError("LoRA shape/rank mismatch across global/local/download states")
        if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
            raise ValueError("Non-finite LoRA factors")
    qo = torch.linalg.qr(torch.cat([b for _, b in converted], dim=1), mode="reduced")[0]
    qi = torch.linalg.qr(torch.cat([a.T for a, _ in converted], dim=1), mode="reduced")[0]
    cores = [(qo.T @ b) @ (a @ qi) for a, b in converted]
    return qo, qi, cores


def spectrum(
    core: Any, retain: float = 1.0, rank_limit: int | None = None, eps: float = 1e-8
) -> tuple[Any, Any, Any]:
    """Nonzero thin SVD; retain refers to cumulative singular-value mass, not variance."""
    import torch

    u, s, vh = torch.linalg.svd(core, full_matrices=False)
    # QR/core rotations of rank-deficient FP32 matrices introduce roundoff-sized
    # singular values. Treating them as genuine directions filters an arbitrary
    # null-space completion, defeating the low-rank protection definition.
    relative_tolerance = max(eps, torch.finfo(core.dtype).eps * max(core.shape))
    nonzero = int((s > relative_tolerance * s.max().clamp_min(eps)).sum())
    count = nonzero
    if count and retain < 1.0:
        count = min(
            count, int(torch.searchsorted(s[:count].cumsum(0), retain * s[:count].sum())) + 1
        )
    if rank_limit is not None:
        count = min(count, rank_limit)
    return u[:, :count], s[:count], vh[:count].T


def refactor(
    core: Any, qo: Any, qi: Any, rank: int, reference_a: Any, reference_b: Any, eps: float
) -> tuple[Any, Any, dict[str, float]]:
    """Best Frobenius rank-r approximation, preserving factor shape/device/dtype.

    Do NOT make both factors zero for discarded singular directions: that would
    prevent their learning. A keeps unit singular directions; B holds singular
    values (rather than splitting sqrt(s) into two zero factors).
    """
    import torch

    u, s, vh = torch.linalg.svd(core, full_matrices=False)
    size = min(rank, len(s))
    norm = s.square().sum()
    error = s[size:].square().sum()
    diagnostics = {
        "functional_norm": float(norm.sqrt()),
        "compression_relative_error": float((error / norm.clamp_min(eps**2)).sqrt()),
        "retained_energy_ratio": float(1.0 - error / norm.clamp_min(eps**2)),
    }
    if float(norm) <= eps**2:
        # Keep a trainable A basis even when there is no functional update.
        return reference_a.detach().clone(), torch.zeros_like(reference_b), diagnostics
    a = torch.zeros_like(reference_a, dtype=torch.float32, device=qo.device)
    b = torch.zeros_like(reference_b, dtype=torch.float32, device=qo.device)
    a[:size] = vh[:size] @ qi.T
    b[:, :size] = (qo @ u[:, :size]) * s[:size]
    if size < rank:
        a[size:] = reference_a.detach().to(a.device).float()[size:]
    return (
        a.to(device=reference_a.device, dtype=reference_a.dtype),
        b.to(device=reference_b.device, dtype=reference_b.dtype),
        diagnostics,
    )


def projection_loss(value: Any, target: Any, u: Any, s: Any, v: Any) -> Any:
    """DOP Eq. (2): ||diag(s) U.T E||² + ||E V diag(s)||²."""
    difference = value - target
    return (s[:, None] * (u.T @ difference)).square().sum() + ((difference @ v) * s).square().sum()


def opcm_filter(incoming: Any, old: Any, retain: float, eps: float) -> tuple[Any, int]:
    """Remove principal block AND singular-direction diagonal (OPCM core step).

    Only nonzero old singular directions are filtered. Unlike the dense original
    implementation, arbitrary completions of its zero-singular-value basis are
    not filtered; doing so would make a rank-deficient LoRA merge gauge-dependent.
    """
    u, s, v = spectrum(old, eps=eps)
    if not len(s):
        return incoming.clone(), 0
    import torch

    cutoff = min(len(s), int(torch.searchsorted(s.cumsum(0), retain * s.sum())) + 1)
    coordinates = u.T @ incoming @ v
    removed = torch.zeros_like(coordinates)
    removed[:cutoff, :cutoff] = coordinates[:cutoff, :cutoff]
    removed.diagonal().copy_(coordinates.diagonal())
    return incoming - u @ removed @ v.T, cutoff
