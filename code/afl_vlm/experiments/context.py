"""Experiment context and append-only artifact writer."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


class ArtifactWriter:
    JSONL_FILES = ("events.jsonl", "probes.jsonl", "updates.jsonl", "branches.jsonl")

    def __init__(self, directory: Path, overwrite: bool) -> None:
        if directory.exists() and any(directory.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {directory}. "
                "Choose a new output root or set output.overwrite=true."
            )
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        for name in self.JSONL_FILES:
            (directory / name).write_text("", encoding="utf-8")

    def write_json(self, name: str, payload: Any) -> None:
        (self.directory / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
            + "\n",
            encoding="utf-8",
        )

    def append(self, name: str, payload: Any) -> None:
        with (self.directory / name).open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default)
                + "\n"
            )

    def write_csv(self, name: str, rows: list[dict[str, Any]]) -> None:
        path = self.directory / name
        if not rows:
            path.write_text("\n", encoding="utf-8")
            return
        fields = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


@dataclass(slots=True)
class ExperimentContext:
    config: dict[str, Any]
    model: Any
    method: Any
    tasks: dict[str, Any]
    clients: dict[str, Any]
    client_specs: list[Any]
    probe_ids: dict[str, list[str]]
    final_ids: dict[str, list[str]]
    heldout_samples: dict[str, list[Any]]
    train_config: Any
    seed: int
    writer: ArtifactWriter
    counters: dict[str, int] = field(default_factory=dict)

    def derived_seed(self, stream: str, *parts: Any) -> int:
        import hashlib

        material = ":".join([str(self.seed), stream, *(str(part) for part in parts)])
        return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "big")
