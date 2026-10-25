"""Export ours_hc's numeric event diagnostics only; no model/checkpoint loading."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export(run: Path, output: Path) -> dict[str, int]:
    updates, modules, steps = [], [], []
    for line in (run / "events.jsonl").read_text("utf-8").splitlines():
        event = json.loads(line)
        for metadata in event.get("result_metadata", []):
            item = metadata.get("hc_diagnostics")
            if item is None:
                continue
            identity = {
                key: item[key]
                for key in (
                    "update_id",
                    "client_id",
                    "task",
                    "base_version",
                    "server_version_before",
                    "arrival_time",
                    "optimizer_steps",
                    "accepted",
                    "sensitivity_valid",
                )
            }
            updates.append(
                {
                    **identity,
                    **{
                        key: item[key]
                        for key in (
                            "version_staleness",
                            "functional_drift",
                            "functional_local_energy",
                            "relative_staleness",
                            "reliability",
                            "frequency_weight",
                        )
                    },
                    **item["communication"],
                    "total_selected_components": sum(item["selected_rank"].values()),
                }
            )
            transform = item["sensitivity_transform"]
            for module in item["module_sensitivity"]:
                modules.append(
                    {
                        **identity,
                        "module": module,
                        "selected_rank": item["selected_rank"][module],
                        "selected_indices": json.dumps(item["selected_indices"][module]),
                        "sensitivity": item["module_sensitivity"][module],
                        "raw_sensitivity": transform.get("raw_by_module", {}).get(module),
                        "signed_gate_mean": transform.get("signed_gate_mean", {}).get(module),
                        "gate_direction": transform.get("gate_direction", {}).get(module),
                        "observations": item["sensitivity_observations"].get(module, 0),
                        **item["local"].get("module_gate_summary", {}).get(module, {}),
                        "omega": item["omega_before"][module],
                        "alpha": item["module_alpha"][module],
                        "task_history_before": item["history_before"][item["task"]][module],
                        "task_history_after": item["history_after"][item["task"]][module],
                    }
                )
            steps.extend({**identity, **row} for row in item["local"].get("trajectory", []))
    if not updates:
        raise ValueError("No ours_hc diagnostics in this run; other methods are not modified")
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("updates", updates), ("modules", modules), ("local_steps", steps)):
        write_csv(output / f"{name}.csv", rows)
    return {"updates": len(updates), "modules": len(modules), "local_steps": len(steps)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    result = export(
        args.run_directory, args.output_directory or args.run_directory / "hc_diagnostics"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
