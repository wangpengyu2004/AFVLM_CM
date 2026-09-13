"""Read-only helpers for identifying the existing client samples."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from afl_vlm.data.base import Sample


def sample_ids_hash(samples_or_ids: Iterable[Sample | str]) -> str:
    identifiers = [str(item.id if isinstance(item, Sample) else item) for item in samples_or_ids]
    return hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()
