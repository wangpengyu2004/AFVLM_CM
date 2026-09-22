"""Export Rank-Gate diagnostics embedded in an AFVLM-CM events.jsonl file."""

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
    include_ranks: bool = False,
) -> dict[str, Any]:
    update_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    for event in _read_events(events_path):
        for item in _diagnostics(event):
            client = item["client"]
            sensitivity = item["sensitivity"]
            functional = item["functional_staleness"]
            aggregation = item["aggregation"]
            memory = item["memory"]
            rank_summary = sensitivity["rank_summary"]
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
                    "sensitivity_valid": sensitivity["valid"],
                    "rank_sensitivity_mean": _summary_value(rank_summary, "mean"),
                    "rank_sensitivity_std": _summary_value(rank_summary, "std"),
                    "rank_sensitivity_min": _summary_value(rank_summary, "min"),
                    "rank_sensitivity_max": _summary_value(rank_summary, "max"),
                    "rank_zero_fraction": _summary_value(rank_summary, "zero_fraction"),
                    "rank_clipped_fraction": _summary_value(
                        rank_summary, "clipped_fraction"
                    ),
                    "alpha_mean": _summary_value(alpha_summary, "mean"),
                    "alpha_min": _summary_value(alpha_summary, "min"),
                    "alpha_max": _summary_value(alpha_summary, "max"),
                    "task_memory_before_mean": _summary_value(before_summary, "mean"),
                    "task_memory_after_mean": _summary_value(after_summary, "mean"),
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
                        "module_sensitivity": module_sensitivity,
                        "observations": observations[module],
                        "module_alpha": alphas.get(module, ""),
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
                if include_ranks:
                    for rank, rank_sensitivity in enumerate(
                        sensitivity["rank_by_module"][module]
                    ):
                        rank_rows.append(
                            {
                                **common,
                                "module": module,
                                "rank": rank,
                                "rank_sensitivity": rank_sensitivity,
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
    rank_path = None
    if include_ranks:
        rank_path = output_directory / "ours_rank_diagnostics.csv"
        _write_csv(rank_path, rank_rows, list(rank_rows[0]))
    return {
        "updates": len(update_rows),
        "modules": len(module_rows),
        "ranks": len(rank_rows),
        "update_csv": str(update_path),
        "module_csv": str(module_path),
        "rank_csv": str(rank_path) if rank_path is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--include-ranks", action="store_true")
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
            export_diagnostics(events_path, output, include_ranks=args.include_ranks),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
