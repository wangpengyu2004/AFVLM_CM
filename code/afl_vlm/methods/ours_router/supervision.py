"""Fixed-one gate observations; never optimizer parameters or federated state."""

from __future__ import annotations

from typing import Any


def gate_labels(gates: dict[str, Any]) -> dict[str, Any]:
    """Mean accumulated h gradients over DDP ranks, once per optimizer boundary.

    Plain leaf tensors are intentionally outside the DDP reducer. Pack presence
    flags with gradients so every rank also agrees on missing observations.
    No gradients of model Parameters are modified by this collective.
    """
    import torch

    names = sorted(gates)
    if not names:
        raise RuntimeError("Missing full-candidate observation gates")
    values = [gates[m].grad for m in names]
    gradients = torch.cat([v.detach().float() if v is not None else
                           torch.zeros_like(gates[m], dtype=torch.float32)
                           for m, v in zip(names, values, strict=True)])
    present = gradients.new_tensor([float(v is not None) for v in values])
    packed = torch.cat((gradients, present))
    world = 1
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(packed)
            world = torch.distributed.get_world_size()
        if not bool((packed[-len(names):] == world).all()):
            raise RuntimeError("A candidate gate is missing task gradients on a replica")
        if not bool(torch.isfinite(packed).all()):
            raise ValueError("Non-finite candidate gate gradients")
        chunks = (-packed[:gradients.numel()] / world).split(
            [gates[m].numel() for m in names]
        )
        return dict(zip(names, chunks, strict=True))
    finally:
        for gate in gates.values():
            gate.grad = None
