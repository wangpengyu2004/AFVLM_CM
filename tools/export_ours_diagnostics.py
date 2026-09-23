"""Export Module-Gate diagnostics embedded in an AFVLM-CM events.jsonl file."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def _read_events(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                # A concurrent training process may still be appending the
                # final JSONL record. Skip only that unterminated tail; a bad
                # complete line remains a hard error.
                if not line.endswith("\n") and handle.read() == "":
                    return
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc


def _diagnostics(event: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for result in event.get("result_metadata", []):
        if not isinstance(result, Mapping):
            continue
        diagnostics = result.get("ours_diagnostics")
        if isinstance(diagnostics, Mapping):
            yield dict(diagnostics)


def _summary_value(section: Mapping[str, Any], key: str) -> Any:
    value = section.get(key)
    return "" if value is None else value


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_diagnostics(
    events_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    update_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    for event in _read_events(events_path):
        for item in _diagnostics(event):
            client = item["client"]
            sensitivity = item["sensitivity"]
            functional = item["functional_staleness"]
            aggregation = item["aggregation"]
            memory = item["memory"]
            module_summary = sensitivity["module_summary"]
            transform = sensitivity.get("transform", {})
            raw_summary = transform.get("raw_summary", {})
            raw_by_module = transform.get("raw_by_module", {})
            final_summary = transform.get("final_summary", module_summary)
            alpha_summary = aggregation["alpha_summary"]
            before_summary = memory["task_memory_before_summary"]
            after_summary = memory["task_memory_after_summary"]
            common = {
                "event_id": event.get("event_id"),
                "update_id": client["update_id"],
                "client_id": client["client_id"],
                "task": client["task"],
                "local_round": client["local_round"],
                "base_version": client["base_version"],
                "receive_version": client["receive_version"],
                "server_version_after": event.get("server_version_after"),
            }
            update_rows.append(
                {
                    **common,
                    "accepted": aggregation["accepted"],
                    "rejection_reason": aggregation.get("rejection_reason") or "",
                    "version_staleness": functional["version_staleness"],
                    "server_drift": functional["server_drift"],
                    "local_update": functional["local_update"],
                    "parameter_delta_norm": functional["parameter_delta_norm"],
                    "relative_staleness": functional["relative_staleness"],
                    "reliability": _summary_value(functional, "reliability"),
                    "reliability_function": functional.get("reliability_function", ""),
                    "gamma": _summary_value(functional, "gamma"),
                    "sensitivity_valid": sensitivity["valid"],
                    "sensitivity_gate": transform.get("gate", "module_gate"),
                    "sensitivity_statistic": transform.get("statistic", ""),
                    "sensitivity_normalization": transform.get("normalization", ""),
                    "sensitivity_uniform_mix": _summary_value(transform, "uniform_mix"),
                    "sensitivity_uniform_floor": _summary_value(
                        transform, "uniform_floor"
                    ),
                    "raw_sensitivity_mean": _summary_value(raw_summary, "mean"),
                    "final_sensitivity_mean": _summary_value(final_summary, "mean"),
                    "module_sensitivity_mean": _summary_value(module_summary, "mean"),
                    "module_sensitivity_std": _summary_value(module_summary, "std"),
                    "module_sensitivity_min": _summary_value(module_summary, "min"),
                    "module_sensitivity_max": _summary_value(module_summary, "max"),
                    "alpha_mean": _summary_value(alpha_summary, "mean"),
                    "alpha_min": _summary_value(alpha_summary, "min"),
                    "alpha_max": _summary_value(alpha_summary, "max"),
                    "task_memory_before_mean": _summary_value(before_summary, "mean"),
                    "task_memory_after_mean": _summary_value(after_summary, "mean"),
                    "task_memory_count_before": memory.get("task_memory_count_before", ""),
                    "task_memory_count_after": memory.get("task_memory_count_after", ""),
                    "history_strength": memory.get("history_strength", ""),
                }
            )
            modules = sensitivity["module_by_module"]
            observations = sensitivity["observations_by_module"]
            alphas = aggregation["module_alpha_by_module"]
            for module, module_sensitivity in sorted(modules.items()):
                module_rows.append(
                    {
                        **common,
                        "module": module,
                        "raw_module_sensitivity": raw_by_module.get(module, ""),
                        "module_sensitivity": module_sensitivity,
                        "observations": observations[module],
                        "module_alpha": alphas.get(module, ""),
                        "task_memory_raw_before": memory.get(
                            "task_memory_raw_before", memory["task_memory_before"]
                        )[module],
                        "task_memory_raw_after": memory.get(
                            "task_memory_raw_after", memory["task_memory_after"]
                        )[module],
                        "task_memory_before": memory["task_memory_before"][module],
                        "task_memory_after": memory["task_memory_after"][module],
                        "historical_precision_before": memory[
                            "historical_precision_before"
                        ][module],
                        "historical_precision_after": memory[
                            "historical_precision_after"
                        ][module],
                    }
                )

    if not update_rows:
        raise ValueError(f"No ours_diagnostics records found in {events_path}")
    output_directory.mkdir(parents=True, exist_ok=True)
    update_fields = list(update_rows[0])
    module_fields = list(module_rows[0])
    update_path = output_directory / "ours_update_diagnostics.csv"
    module_path = output_directory / "ours_module_diagnostics.csv"
    _write_csv(update_path, update_rows, update_fields)
    _write_csv(module_path, module_rows, module_fields)
    return {
        "updates": len(update_rows),
        "modules": len(module_rows),
        "update_csv": str(update_path),
        "module_csv": str(module_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    run_directory = args.run_directory.resolve()
    events_path = run_directory / "events.jsonl"
    if not events_path.is_file():
        raise FileNotFoundError(f"Missing events log: {events_path}")
    output = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else run_directory / "ours_diagnostics"
    )
    print(
        json.dumps(
            export_diagnostics(events_path, output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
