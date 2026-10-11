"""Match existing reference-DDP optimizer windows without changing DDP training."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


def detect_reference_world_size() -> int:
    """Detect visible CUDA devices only at a main-process configuration boundary.

    Importing this module/static YAML validation never probes hardware. Do not
    call this inside a worker with its one-GPU CUDA_VISIBLE_DEVICES mask.
    """
    try:
        import torch
    except ImportError as exc:
        raise ValueError(
            "Automatic reference GPU detection requires PyTorch. Generate the profile "
            "on the training server or pass --reference_world_size to the generator."
        ) from exc
    count = int(torch.cuda.device_count())
    if count < 1:
        raise ValueError(
            "No visible CUDA GPU detected; refusing to silently use a one-GPU budget. "
            "Generate the profile on the training server or pass "
            "--reference_world_size 8 for an eight-GPU reference experiment."
        )
    return count


def freeze_reference_training(
    training: Mapping[str, Any], reference_world_size: int
) -> dict[str, Any]:
    """Record derived batch metadata without mutating the editable source."""
    frozen = copy.deepcopy(dict(training))
    frozen["batch_size_scope"] = "reference_ddp"
    frozen["reference_world_size"] = reference_world_size
    resolve_batch_layout(frozen)
    return frozen


def freeze_runtime_batch_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve an editable config before model load/spawn; never rewrite snapshots.

    Immutable historical configs missing these fields keep per-device behavior.
    New profiles already have explicit frozen metadata and never re-detect GPUs.
    """
    if config.get("experiment_profile", {}).get("immutable_snapshot", False):
        return config
    if "batch_size_scope" in config["training"]:
        return config
    frozen = copy.deepcopy(config)
    frozen["training"] = freeze_reference_training(
        config["training"], detect_reference_world_size()
    )
    return frozen


@dataclass(frozen=True)
class BatchLayout:
    batch_size_scope: str
    reference_batch_size: int
    reference_gradient_accumulation: int
    training_world_size: int
    per_device_batch_size: int
    physical_gradient_accumulation: int
    effective_batch_size: int
    reference_world_size: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MicroBatch:
    indices: tuple[int, ...]


def resolve_batch_layout(training: Mapping[str, Any], world_size: int = 1) -> BatchLayout:
    """Missing scope preserves immutable historical per-device configurations.

    reference_ddp retains B/G on the reference number of real DDP ranks;
    one-GPU clients sequentially accumulate those SAME virtual-rank batches.
    """
    batch = int(training["batch_size"])
    accumulation = int(training["gradient_accumulation"])
    scope = str(training.get("batch_size_scope", "per_device"))
    if min(batch, accumulation, world_size) <= 0:
        raise ValueError("Batch size, accumulation and training world size must be positive")
    if scope not in {"reference_ddp", "per_device"}:
        raise ValueError("training.batch_size_scope must be reference_ddp or per_device")
    reference = (
        int(training.get("reference_world_size", 8)) if scope == "reference_ddp" else world_size
    )
    if reference <= 0:
        raise ValueError("training.reference_world_size must be positive")
    if scope == "reference_ddp" and world_size not in {1, reference}:
        raise ValueError(
            f"reference_ddp requires {reference} real DDP ranks or one training GPU; "
            f"got {world_size}. Keep the reference GPU count to preserve existing experiments."
        )
    physical_accumulation = accumulation * reference // world_size
    return BatchLayout(
        scope,
        batch,
        accumulation,
        world_size,
        batch,
        physical_accumulation,
        batch * accumulation * reference,
        reference,
    )


def optimizer_windows(
    indices: Sequence[int], layout: BatchLayout, rank: int = 0
) -> Iterator[tuple[MicroBatch, ...]]:
    """Use precisely the historical padded reference DistributedSampler shards.

    With real reference-DDP ranks this is the original rank's window. On one
    GPU, flatten the same reference windows (rank-major) before ONE optimizer
    step. Equal window lengths make /len(flattened_window) equal the original
    DDP /len(rank_window) followed by rank averaging, including epoch tails.
    """
    world = layout.training_world_size
    if not 0 <= rank < world:
        raise ValueError("Distributed rank must belong to the training world")
    if not indices:
        return
    reference = layout.reference_world_size
    if world == 1 and reference > 1:
        shards = [_rank_windows(indices, layout, virtual_rank) for virtual_rank in range(reference)]
        for windows in zip(*shards, strict=True):
            yield tuple(batch for window in windows for batch in window)
        return
    yield from _rank_windows(indices, layout, rank)


def _rank_windows(
    indices: Sequence[int], layout: BatchLayout, rank: int
) -> Iterator[tuple[MicroBatch, ...]]:
    padded = list(indices)
    reference = layout.reference_world_size
    padded_size = math.ceil(len(padded) / reference) * reference
    missing = padded_size - len(padded)
    if missing:
        padded.extend((padded * math.ceil(missing / len(padded)))[:missing])
    local = padded[rank:padded_size:reference]
    batches = [
        MicroBatch(tuple(local[start : start + layout.per_device_batch_size]))
        for start in range(0, len(local), layout.per_device_batch_size)
    ]
    accumulation = layout.reference_gradient_accumulation
    for start in range(0, len(batches), accumulation):
        yield tuple(batches[start : start + accumulation])


def planned_optimizer_steps(
    num_samples: int, local_epochs: int, training: Mapping[str, Any], world_size: int = 1
) -> int:
    """Physical epoch steps; reference_ddp matches the historical eight-rank steps."""
    if min(num_samples, local_epochs) <= 0:
        raise ValueError("Sample count and local epochs must be positive")
    layout = resolve_batch_layout(training, world_size)
    per_rank = math.ceil(num_samples / layout.reference_world_size)
    return local_epochs * math.ceil(
        math.ceil(per_rank / layout.per_device_batch_size) / layout.reference_gradient_accumulation
    )
