"""Aggregate numeric result fields across seeds without averaging task-native metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize(output_root: str | Path) -> list[dict[str, Any]]:
    root = Path(output_root).resolve()
    rows: list[dict[str, Any]] = []
    for path in root.glob("*/**/summary.json"):
        if path == root / "summary.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload.get("rows", []))
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["experiment"]), str(row["method"]), str(row["scenario"]))
        grouped[key].append(row)
    output: list[dict[str, Any]] = []
    for (experiment, method, scenario), items in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "experiment": experiment,
            "method": method,
            "scenario": scenario,
            "count": len(items),
        }
        numeric_keys = set.intersection(
            *[
                {key for key, value in item.items() if isinstance(value, int | float)}
                for item in items
            ]
        )
        numeric_keys -= {"seed"}
        for key in sorted(numeric_keys):
            values = [float(item[key]) for item in items]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            summary[f"{key}_mean"] = mean
            summary[f"{key}_std"] = math.sqrt(variance)
        output.append(summary)
    (root / "aggregate_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = sorted({key for row in output for key in row})
    with (root / "aggregate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root")
    arguments = parser.parse_args()
    rows = summarize(arguments.output_root)
    print(f"Wrote {len(rows)} aggregate row(s) under {Path(arguments.output_root).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
