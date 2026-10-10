from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.export_ours_diagnostics import export_diagnostics


class OursDiagnosticsExportTests(unittest.TestCase):
    def test_exports_update_and_module_rows(self) -> None:
        module = "model.layers.0.self_attn.q_proj"
        diagnostics = {
            "schema_version": 4,
            "method_variant": "module_gate",
            "client": {
                "update_id": "vqa/client_0-r0",
                "client_id": "vqa/client_0",
                "task": "vqa",
                "local_round": 0,
                "base_version": 0,
                "receive_version": 2,
                "arrival_time": 1.0,
                "num_samples": 4,
                "optimizer_steps": 2,
            },
            "sensitivity": {
                "valid": True,
                "module_by_module": {module: 1.0},
                "observations_by_module": {module: 2},
                "module_summary": {
                    "count": 1,
                    "mean": 1.0,
                    "std": 0.0,
                    "min": 1.0,
                    "max": 1.0,
                    "zero_fraction": 0.0,
                    "clipped_fraction": None,
                },
                "transform": {
                    "gate": "module_gate",
                    "statistic": "mean_absolute_gate",
                    "normalization": "client_module_mean",
                    "uniform_mix": 0.5,
                    "uniform_floor": 0.5,
                    "raw_by_module": {module: 2.0},
                    "raw_summary": {"mean": 2.0},
                    "final_summary": {"mean": 1.0},
                    "signed_gate_mean": {module: -1.0},
                    "gate_direction": {module: 0.5},
                    "gate_direction_abs_mean": 0.5,
                },
            },
            "functional_staleness": {
                "version_staleness": 2,
                "server_drift": 3.0,
                "local_update": 4.0,
                "parameter_delta_norm": 5.0,
                "relative_staleness": 0.75,
                "reliability": 0.4,
            },
            "frequency": {
                "window_size": 18,
                "window_length_before": 4,
                "window_length_after": 5,
                "current_task_count_before": 1,
                "current_task_count_after": 2,
                "raw_weight": 1.2,
                "weight": 1.2,
            },
            "aggregation": {
                "rule": "base_relative_delta",
                "accepted": True,
                "rejection_reason": None,
                "module_alpha_by_module": {module: 0.2},
                "alpha_summary": {
                    "count": 1,
                    "mean": 0.2,
                    "std": 0.0,
                    "min": 0.2,
                    "max": 0.2,
                    "zero_fraction": 0.0,
                    "clipped_fraction": None,
                },
            },
            "memory": {
                "task": "vqa",
                "task_memory_before": {module: 0.1},
                "task_memory_after": {module: 0.15},
                "historical_precision_before": {module: 1.05},
                "historical_precision_after": {module: 1.075},
                "task_memory_before_summary": {"mean": 0.1},
                "task_memory_after_summary": {"mean": 0.15},
                "historical_precision_unweighted_before": {module: 1.0},
                "historical_precision_unweighted_after": {module: 1.0},
                "history_age_before": {
                    "enabled": True, "boost": 1.0, "time_scale": 18.0,
                    "accepted_updates": 2, "finished_at": {"cls": 1},
                    "pending_updates": {"cls": 0, "vqa": 2},
                    "age_by_task": {"cls": 1, "vqa": 0},
                    "weight_by_task": {"cls": 1.05, "vqa": 1.0},
                },
                "history_age_after": {
                    "accepted_updates": 3, "finished_at": {"cls": 1},
                    "pending_updates": {"cls": 0, "vqa": 1},
                    "age_by_task": {"cls": 2, "vqa": 0},
                    "weight_by_task": {"cls": 1.1, "vqa": 1.0},
                },
            },
        }
        event = {
            "event_id": 3,
            "server_version_after": 3,
            "result_metadata": [{"ours_diagnostics": diagnostics}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events = root / "events.jsonl"
            events.write_text(json.dumps(event) + "\n", encoding="utf-8")

            result = export_diagnostics(events, root / "derived")

            self.assertEqual(result["updates"], 1)
            self.assertEqual(result["modules"], 1)
            with Path(result["update_csv"]).open(encoding="utf-8-sig", newline="") as handle:
                update_rows = list(csv.DictReader(handle))
            with Path(result["module_csv"]).open(encoding="utf-8-sig", newline="") as handle:
                module_rows = list(csv.DictReader(handle))
            self.assertEqual(update_rows[0]["relative_staleness"], "0.75")
            self.assertEqual(update_rows[0]["aggregation_rule"], "base_relative_delta")
            self.assertEqual(module_rows[0]["aggregation_rule"], "base_relative_delta")
            self.assertEqual(update_rows[0]["sensitivity_statistic"], "mean_absolute_gate")
            self.assertEqual(update_rows[0]["frequency_weight"], "1.2")
            self.assertEqual(update_rows[0]["gate_direction_abs_mean"], "0.5")
            self.assertEqual(module_rows[0]["raw_module_sensitivity"], "2.0")
            self.assertEqual(module_rows[0]["signed_gate_mean"], "-1.0")
            self.assertEqual(module_rows[0]["gate_direction"], "0.5")
            self.assertEqual(module_rows[0]["module_alpha"], "0.2")
            self.assertEqual(update_rows[0]["history_age_enabled"], "True")
            self.assertEqual(update_rows[0]["history_accepted_updates_after"], "3")
            self.assertEqual(
                json.loads(update_rows[0]["history_task_weights_before"]),
                {"cls": 1.05, "vqa": 1.0},
            )
            self.assertEqual(
                json.loads(update_rows[0]["history_tasks_finished_at_after"]), {"cls": 1}
            )
            self.assertEqual(module_rows[0]["historical_precision_unweighted_before"], "1.0")

            # Old events lack the additive rule label; do not mislabel their semantics.
            diagnostics["aggregation"].pop("rule")
            for key in (
                "history_age_before", "history_age_after",
                "historical_precision_unweighted_before", "historical_precision_unweighted_after",
            ):
                diagnostics["memory"].pop(key)
            events.write_text(json.dumps(event) + "\n", encoding="utf-8")
            legacy = export_diagnostics(events, root / "legacy")
            for key in ("update_csv", "module_csv"):
                with Path(legacy[key]).open(encoding="utf-8-sig", newline="") as handle:
                    row = next(csv.DictReader(handle))
                    self.assertEqual(row["aggregation_rule"], "")
                    field = (
                        "history_task_weights_before" if key == "update_csv"
                        else "historical_precision_unweighted_before"
                    )
                    self.assertEqual(row[field], "")

    def test_ignores_only_an_incomplete_trailing_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events = root / "events.jsonl"
            events.write_text('{"event_id": 1}\n{"event_id":', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "No ours_diagnostics"):
                export_diagnostics(events, root / "derived")


if __name__ == "__main__":
    unittest.main()
