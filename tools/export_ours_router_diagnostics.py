"""Export capacity, routing, sensitivity and fusion diagnostics without models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def ranking_rows(trajectory: list[dict], identity: dict, phase: str) -> tuple[list, list]:
    """Flatten old/new trajectories; never infer missing historical measurements."""
    steps, modules = [], []
    for step in trajectory:
        summary = {**identity, **{key: value for key, value in step.items()
                                  if key != "router_ranking"}}
        ranking = step.get("router_ranking")
        if ranking is not None:
            summary["ranking_phase"] = ranking.get("phase", phase)
            summary["ranking_scope"] = ranking.get("scope")
            for name, detail in ranking["modules"].items():
                modules.append({
                    **identity, "phase": ranking.get("phase", phase),
                    "scope": ranking.get("scope"), "optimizer_step": step["optimizer_step"],
                    "module": name, "prediction_time": ranking["prediction_time"],
                    "target": ranking["target"], "topk_fraction": ranking["topk_fraction"],
                    **{key: value for key, value in detail.items() if key != "component_indices"},
                    "component_indices": json.dumps(detail.get("component_indices")),
                })
        steps.append(summary)
    return steps, modules


def export(run: Path, output: Path) -> dict:
    updates, modules, steps, capacities, components, resolutions = [], [], [], [], [], []
    conflicts, windows, probe_steps = [], [], []
    rankings = []
    events = run / "events.jsonl"
    for line in events.read_text("utf-8").splitlines() if events.is_file() else []:
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
                        key: row.get(key)
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
                    "router_sharing": row["local"].get("router_sharing"),
                    "router_head_count": row["local"].get("router_head_count"),
                    "total_selected_components": sum(row["selected_rank"].values()),
                    "aggregation_mode": row.get("aggregation_mode", "legacy_or_overlap_switch"),
                    "uniform_alpha": row.get("uniform_alpha"),
                    "server_step_size": row.get("server_step_size"),
                }
            )
            transform = row["sensitivity_transform"]
            scoped = row.get("scoped_aggregation", {}).get("modules", {})
            selection = row.get("conflict_selection", {})
            for name, detail in selection.get("modules", {}).items():
                for index, score in enumerate(detail["scores"]):
                    conflicts.append({
                        **identity, "module": name, "component_index": index,
                        "download_accepted_clock": selection["clock"], "router_score": score,
                        "normalized_score": detail["normalized_scores"][index],
                        "conflict": detail["penalty"][index], "valid": detail["valid"][index],
                        "peer_count": detail["peer_count"][index],
                        "own_direction_valid": detail["own_direction_valid"][index],
                        "final_score": detail["final_scores"][index],
                        "penalty_strength": detail["penalty_strength"],
                        "selected": index in row["selected_indices"][name],
                        "router_only_selected": index in detail["router_only_indices"],
                    })
            window = row.get("conflict_window_after")
            if window is not None:
                for task, records in window["tasks"].items():
                    windows.append({**identity, "reference_task": task,
                                    "clock": window["successful_updates"],
                                    "record_count": len(records),
                                    "per_task_limit": window["per_task_limit"],
                                    "max_age": window["max_age"],
                                    "records": json.dumps(records, ensure_ascii=False),
                                    "total_factor_bytes": window["factor_bytes"],
                                    "total_gram_bytes": window["gram_bytes"]})
            if "job_resolution" in row:
                resolutions.append(
                    {
                        **identity,
                        "local_round": row.get("local_round"),
                        "resolution": json.dumps(row["job_resolution"], ensure_ascii=False),
                    }
                )
            for name, rank in row["selected_rank"].items():
                detail = scoped.get(name)
                if detail is not None:
                    for offset, index in enumerate(row["selected_indices"][name]):
                        components.append(
                            {
                                **identity,
                                "local_round": row["local_round"],
                                "module": name,
                                "global_rank": detail["global_rank"],
                                "component_index": index,
                                "competition": index in detail["competition_indices"],
                                "sensitivity_aggregation": detail["sensitivity_mask"][offset],
                                "exit_history": index in detail["exit_history_indices"],
                                "effective_frequency": detail["effective_frequency"][offset],
                                "sensitivity": row["module_sensitivity"][name],
                                "reliability": row["reliability"],
                                "omega_base": detail["omega_base"][offset],
                                "original_module_omega_base": detail["original_module_omega_base"],
                                "omega_age": detail["omega_age"][offset],
                                "omega": detail["omega"][offset],
                                "alpha": detail["alpha"][offset],
                                "protection_sources": json.dumps(
                                    detail["protection_sources"], ensure_ascii=False
                                ),
                            }
                        )
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
                        "history_before": row.get("history_before", {}).get(
                            row["task"], {}
                        ).get(name),
                        "history_after": row.get("history_after", {}).get(
                            row["task"], {}
                        ).get(name),
                        "omega_base": row.get("omega_base", {}).get(name),
                        "omega_extra": row.get("omega_extra", {}).get(name),
                        "omega": row.get("omega", {}).get(name),
                        "alpha": row["module_alpha"][name],
                        "conflict_valid_coverage": selection.get("modules", {}).get(
                            name, {}
                        ).get("selected_valid_comparison_fraction"),
                        "selection_changed_components": selection.get("modules", {}).get(
                            name, {}
                        ).get("changed_components"),
                        "summary_scope": row.get("summary_scope", "one_coefficient_per_module"),
                        "scoped_details": json.dumps(detail, ensure_ascii=False)
                        if detail is not None
                        else "",
                    }
                )
            step_rows, module_rows = ranking_rows(row["local"]["trajectory"], identity, "formal")
            steps.extend(step_rows)
            rankings.extend(module_rows)
    calibration = run / "client_calibration.json"
    if calibration.is_file():
        for client in json.loads(calibration.read_text("utf-8"))["clients"]:
            probe_identity = {"client_id": client["client_id"], "task": client["task"]}
            step_rows, module_rows = ranking_rows(client.get("trajectory", []), probe_identity,
                                                 "probe")
            probe_steps.extend(step_rows)
            rankings.extend(module_rows)
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
    terminal_log = run / "job_resolutions.jsonl"
    if terminal_log.is_file():
        for line in terminal_log.read_text("utf-8").splitlines():
            terminal = json.loads(line)
            if terminal.get("method") == "ours_router":
                resolutions.append(
                    {
                        "client_id": terminal.get("client_id"),
                        "local_round": terminal.get("local_round"),
                        "resolution": json.dumps(terminal, ensure_ascii=False),
                    }
                )
    if not updates and not resolutions:
        raise ValueError("No ours_router arrival diagnostics found")
    output.mkdir(parents=True, exist_ok=False)
    counts = {}
    for name, rows in (
        ("updates", updates),
        ("modules", modules),
        ("local_steps", steps),
        ("capacities", capacities),
        ("components", components),
        ("resolutions", resolutions),
        ("conflict_selection", conflicts),
        ("conflict_windows", windows),
        ("probe_steps", probe_steps),
        ("router_ranking", rankings),
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
