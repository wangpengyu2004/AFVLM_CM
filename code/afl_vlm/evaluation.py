"""Shared exact-corpus helpers for multi-GPU evaluation."""

from __future__ import annotations


def merge_evaluation_outputs(
    shards: list[list[tuple[int, str, str]]], total: int
) -> tuple[list[str], list[str]]:
    """Restore corpus order and reject missing/duplicate distributed samples."""

    rows = sorted((row for shard in shards for row in shard), key=lambda row: row[0])
    indices = [row[0] for row in rows]
    if indices != list(range(total)):
        raise RuntimeError(
            "Distributed evaluation outputs do not cover the corpus exactly once: "
            f"expected={total}, received={len(rows)}"
        )
    return [row[1] for row in rows], [row[2] for row in rows]
