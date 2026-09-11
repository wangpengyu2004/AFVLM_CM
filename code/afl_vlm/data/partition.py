"""Deterministic, disjoint client partitioning and diagnostic sample selection."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable

from afl_vlm.data.base import Sample


def partition_disjoint(
    samples: list[Sample], client_ids: list[str], per_client: int, seed: int
) -> dict[str, list[Sample]]:
    if len(samples) < len(client_ids) * per_client:
        raise ValueError(f"Need {len(client_ids) * per_client} train samples, found {len(samples)}")
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    partitions = {
        client_id: shuffled[index * per_client : (index + 1) * per_client]
        for index, client_id in enumerate(client_ids)
    }
    assert_disjoint(partitions)
    return partitions


def assert_disjoint(partitions: dict[str, list[Sample]]) -> None:
    seen: set[str] = set()
    for client_id, samples in partitions.items():
        identifiers = {sample.id for sample in samples}
        if len(identifiers) != len(samples):
            raise ValueError(f"Duplicate sample IDs inside client {client_id}")
        overlap = seen.intersection(identifiers)
        if overlap:
            raise ValueError(f"Client partitions overlap: {sorted(overlap)[:3]}")
        seen.update(identifiers)


def select_ids(samples: list[Sample], count: int, seed: int) -> list[str]:
    if len(samples) < count:
        raise ValueError(f"Requested {count} samples, found {len(samples)}")
    selected = list(samples)
    random.Random(seed).shuffle(selected)
    return [sample.id for sample in selected[:count]]


def sample_ids_hash(samples_or_ids: Iterable[Sample | str]) -> str:
    identifiers = [str(item.id if isinstance(item, Sample) else item) for item in samples_or_ids]
    return hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()


def local_batch(
    samples: list[Sample], local_round: int, steps: int, grad_accumulation: int
) -> list[Sample]:
    required = max(1, steps * grad_accumulation)
    start = (local_round * required) % len(samples)
    return [samples[(start + index) % len(samples)] for index in range(required)]
