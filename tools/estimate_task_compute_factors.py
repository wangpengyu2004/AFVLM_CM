"""Estimate relative per-task compute costs from a completed parallel run."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_directory",
        type=Path,
        help="Run directory containing updates.jsonl",
    )
    args = parser.parse_args()
    source = args.run_directory.resolve() / "updates.jsonl"
    if not source.is_file():
        parser.error(f"Missing update log: {source}")
    seconds_per_step: dict[str, list[float]] = defaultdict(list)
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        start = row.get("worker_start_wall_time")
        finish = row.get("worker_finish_wall_time")
        steps = int(row.get("optimizer_steps", 0))
        if start is None or finish is None or steps <= 0:
            continue
        seconds_per_step[str(row["task"])].append((float(finish) - float(start)) / steps)
    if not seconds_per_step:
        parser.error("No parallel worker timing records were found")
    task_medians = {
        task: statistics.median(values) for task, values in seconds_per_step.items()
    }
    reference = statistics.median(task_medians.values())
    payload = {
        "federation": {
            "task_compute_factors": {
                task: round(value / reference, 6)
                for task, value in sorted(task_medians.items())
            }
        },
        "measurement": {
            "source": str(source),
            "normalization": "median task seconds per optimizer step = 1.0",
            "updates_per_task": {
                task: len(values) for task, values in sorted(seconds_per_step.items())
            },
        },
    }
    print(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
