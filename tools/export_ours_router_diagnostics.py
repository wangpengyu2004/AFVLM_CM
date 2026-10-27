"""Export capacity, routing, sensitivity and fusion diagnostics without models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def export(run: Path, output: Path) -> dict:
    updates, modules, steps, capacities = [], [], [], []
    for line in (run / "events.jsonl").read_text("utf-8").splitlines():
        event = json.loads(line)
        for metadata in event.get("result_metadata", []):
            row = metadata.get("router_diagnostics")
            if row is None:
                continue
            identity = {
                key: row[key]
                for key in (
                    "update_id",
                    "client_id",
                    "task",
                    "base_version",
                    "server_version_before",
                    "arrival_time",
                    "accepted",
                )
            }
            updates.append(
                {
                    **identity,
                    **{
                        key: row[key]
                        for key in (
                            "optimizer_steps",
                            "version_staleness",
                            "functional_drift",
                            "functional_local_energy",
                            "relative_staleness",
                            "reliability",
                            "frequency_weight",
                        )
                    },
                    **row["communication"],
                    "local_wall_time_seconds": row["local"].get("local_wall_time_seconds"),
                    "router_parameter_count": row["local"].get("router_parameter_count"),
                    "total_selected_components": sum(row["selected_rank"].values()),
                }
            )
            transform = row["sensitivity_transform"]
            for name, rank in row["selected_rank"].items():
                modules.append(
                    {
                        **identity,
                        "module": name,
                        "selected_rank": rank,
                        "selected_indices": json.dumps(row["selected_indices"][name]),
                        "selection_source": row["local"]["selection_source"][name],
                        "sensitivity": row["module_sensitivity"][name],
                        "raw_sensitivity": transform.get("raw_by_module", {}).get(name),
                        "gate_direction": transform.get("gate_direction", {}).get(name),
                        "history_before": row["history_before"][row["task"]][name],
                        "history_after": row["history_after"][row["task"]][name],
                        "omega_base": row["omega_base"][name],
                        "omega_extra": row["omega_extra"][name],
                        "omega": row["omega"][name],
                        "alpha": row["module_alpha"][name],
                    }
                )
            steps.extend({**identity, **step} for step in row["local"]["trajectory"])
    calibration = run / "client_calibration.json"
    if calibration.is_file():
        for client in json.loads(calibration.read_text("utf-8"))["clients"]:
            for name, record in client["modules"].items():
                capacities.append(
                    {
                        "client_id": client["client_id"],
                        "task": client["task"],
                        "module": name,
                        **{
                            key: value
                            for key, value in record.items()
                            if key != "squared_singular_values"
                        },
                    }
                )
    if not updates:
        raise ValueError("No ours_router arrival diagnostics found")
    output.mkdir(parents=True, exist_ok=False)
    counts = {}
    for name, rows in (
        ("updates", updates),
        ("modules", modules),
        ("local_steps", steps),
        ("capacities", capacities),
    ):
        fields = list(dict.fromkeys(key for record in rows for key in record))
        with (output / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        counts[name] = len(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                args.run_directory,
                args.output_directory or args.run_directory / "router_diagnostics",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
