"""Recorded-result arithmetic and preservation checks; no model or training execution."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml
from afl_vlm.config import load_config
from afl_vlm.experiments.rank_probe import (
    append_json,
    build_jobs,
    hash_file,
    steps_per_epoch,
    validate_probe_config,
    write_json,
)
from afl_vlm.experiments.rank_probe_history import (
    SHARED_SOURCES,
    merge_probe_runs,
    read_completed_run,
    validate_extension,
)

ROOT = Path(__file__).resolve().parents[1]


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def file_snapshot(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): hash_file(path)
            for path in root.rglob("*") if path.is_file()}


def make_recorded_run(root: Path, ranks: list[int], *, learning_rate: float = 2e-5) -> None:
    """Small arithmetic JSON records only, never synthetic model execution/results."""
    root.mkdir()
    config = load_config(ROOT / "configs/probes/local_rank_probe.yaml")
    config.pop("_config_path")
    config["experiment"]["ranks"] = ranks
    config["training"]["learning_rate"] = learning_rate
    jobs = build_jobs(config, task="cls")
    data = {"partition_root": "unused-recorded-path", "source_files": {}, "tasks": {}}
    for task in config["dataset"]["tasks"]:
        for client in config["experiment"]["clients"]:
            data["source_files"][f"{task}/{client}.json"] = hashlib.sha256(
                client.encode()).hexdigest()
        data["source_files"][f"{task}/val.json"] = "validation-file-hash"
        data["tasks"][task] = {"client_records": {"client_0": 129, "client_1": 130},
            "validation_ids": ["v0", "v1"], "validation_ids_sha256": "validation-ids",
            "primary_protocol": "validation", "subsets": {"validation": [0, 1]},
            "subset_ids_sha256": {"validation": "validation-ids"},
            "overlapping_validation_records": 0}
    for job in jobs:
        folder = root / job.key
        folder.mkdir(parents=True)
        task_data = data["tasks"][job.task]
        count = task_data["client_records"][job.client]
        per_epoch = steps_per_epoch(count, 4, 4)
        for epoch in range(1, 6):
            value = {2: 10, 4: 14, 8: 15, 16: 16, 32: 16.5}[job.lora_rank]
            value += epoch + (2 if job.client == "client_1" else 0)
            append_json(folder / "metrics.jsonl", {**job.as_dict(), "epoch": epoch,
                "optimizer_steps": per_epoch * epoch, "num_train_samples": count,
                "num_eval_samples": 2, "protocol": "validation", "primary": True,
                "metrics": {"accuracy": value}, "train_annotation_sha256":
                    data["source_files"][f"{job.task}/{job.client}.json"],
                "validation_ids_sha256": "validation-ids"})
        write_json(folder / "metadata.json", {**job.as_dict(),
            "versions": {"torch": "recorded-version", "transformers": "recorded-version"}})
        write_json(folder / "result.json", {**job.as_dict(), "status": "completed",
            "optimizer_steps": 5 * per_epoch, "completed_epochs": [1, 2, 3, 4, 5]})
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    write_json(root / "data_manifest.json", data)
    write_json(root / "manifest.json", {"kind": "local_rank_probe", "status": "completed",
        "local_epochs": 5, "world_size": 8,
        "config_sha256": hash_file(root / "resolved_config.yaml"),
        "data_sha256": hash_file(root / "data_manifest.json"),
        "source_sha256": {path: hash_file(ROOT / path) for path in SHARED_SOURCES},
        "jobs": [{**job.as_dict(), "status": "completed"} for job in jobs]})


class RankProbeHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.old, self.new, self.merged = (self.root / name for name in ("old", "new", "combined"))
        make_recorded_run(self.old, [2, 4, 8])
        make_recorded_run(self.new, [16, 32])

    def test_only_additional_ranks_have_twelve_jobs_and_original_scaling(self) -> None:
        _, config, _ = read_completed_run(self.new)
        jobs = build_jobs(config)
        self.assertEqual(len(jobs), 12)
        self.assertEqual({(job.lora_rank, job.lora_alpha) for job in jobs}, {(16, 32), (32, 64)})
        self.assertEqual(config["training"]["local_epochs"], 5)
        self.assertEqual(config["training"]["batch_size"] * 8
                         * config["training"]["gradient_accumulation"], 128)
        self.assertEqual(len(build_jobs(load_config(
            ROOT / "configs/probes/local_rank_probe.yaml"))), 18)

    def test_reject_invalid_rank_lists(self) -> None:
        for ranks in ([16, 16], [32, 16], [], [0], [16.0], [True]):
            config = load_config(ROOT / "configs/probes/local_rank_probe.yaml")
            config["experiment"]["ranks"] = ranks
            with self.subTest(ranks=ranks), self.assertRaises(ValueError):
                validate_probe_config(config)

    def test_extension_preserves_original_and_refuses_retraining_completed_rank(self) -> None:
        before = file_snapshot(self.old)
        _, config, data = read_completed_run(self.new)
        jobs = build_jobs(config, task="cls")
        validate_extension(self.old, config, data, jobs, ROOT)
        self.assertEqual(file_snapshot(self.old), before)
        _, old_config, old_data = read_completed_run(self.old)
        with self.assertRaisesRegex(ValueError, "Refusing to retrain"):
            validate_extension(self.old, old_config, old_data,
                               build_jobs(old_config, task="cls"), ROOT)

    def test_combined_five_rank_metrics_and_gains_leave_both_sources_byte_identical(self) -> None:
        before = (file_snapshot(self.old), file_snapshot(self.new))
        report = merge_probe_runs([self.old, self.new], self.merged)
        self.assertEqual(report["jobs"], 10)  # cls two clients x five ranks in this record fixture.
        self.assertEqual(report["ranks"], 5)
        self.assertEqual(report["metric_rows"], 50)
        self.assertEqual(report["gain_rows"], 10)
        final = csv_rows(self.merged / "final_epoch_metrics.csv")
        self.assertEqual({int(row["lora_rank"]) for row in final}, {2, 4, 8, 16, 32})
        gain = csv_rows(self.merged / "rank_gains.csv")[0]
        for field, value in {"gain_2_to_4": 4, "gain_4_to_8": 1,
                             "gain_8_to_16": 1, "gain_16_to_32": .5,
                             "gain_per_added_rank_16_to_32": .5 / 16}.items():
            self.assertEqual(float(gain[field]), value)
        self.assertEqual((file_snapshot(self.old), file_snapshot(self.new)), before)
        self.assertFalse((self.old / "metrics_long.csv").exists())
        self.assertFalse((self.merged / "adapters").exists())
        self.assertFalse((self.merged / "predictions").exists())
        with self.assertRaises(FileExistsError):
            merge_probe_runs([self.old, self.new], self.merged)

    def test_different_learning_rate_is_not_silently_combined(self) -> None:
        other = self.root / "different_lr"
        make_recorded_run(other, [16, 32], learning_rate=1e-5)
        with self.assertRaisesRegex(ValueError, "Different training"):
            merge_probe_runs([self.old, other], self.merged)
        self.assertFalse(self.merged.exists())

    def test_different_package_versions_are_not_silently_combined(self) -> None:
        path = self.new / "cls/client_0/rank16/seed42/metadata.json"
        record = json.loads(path.read_text("utf-8"))
        record["versions"]["torch"] = "different-version"
        write_json(path, record)
        with self.assertRaisesRegex(ValueError, "Package versions differ"):
            merge_probe_runs([self.old, self.new], self.merged)
        self.assertFalse(self.merged.exists())

    def test_missing_evaluation_or_changed_frozen_config_fails_before_output_creation(self) -> None:
        path = self.new / "cls/client_0/rank16/seed42/metrics.jsonl"
        path.write_text("\n".join(path.read_text("utf-8").splitlines()[:-1]) + "\n",
                        encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing epoch"):
            merge_probe_runs([self.old, self.new], self.merged)
        self.assertFalse(self.merged.exists())
        config_path = self.old / "resolved_config.yaml"
        config_path.write_text(config_path.read_text("utf-8") + "# changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Frozen input changed"):
            read_completed_run(self.old)

    def test_duplicate_rank_or_output_inside_old_run_is_rejected(self) -> None:
        duplicate = self.root / "duplicate"
        make_recorded_run(duplicate, [8, 16])
        with self.assertRaisesRegex(ValueError, "Duplicate job"):
            merge_probe_runs([self.old, duplicate], self.merged)
        with self.assertRaises(FileExistsError):
            merge_probe_runs([self.old, self.new], self.old / "combined")

    def test_incomplete_paired_matrix_is_rejected(self) -> None:
        path = self.new / "manifest.json"
        manifest = json.loads(path.read_text("utf-8"))
        manifest["jobs"] = [row for row in manifest["jobs"]
                            if not (row["client"] == "client_1" and row["lora_rank"] == 32)]
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "Incomplete paired rank matrix"):
            merge_probe_runs([self.old, self.new], self.merged)

    def test_changed_validation_signature_prevents_extension(self) -> None:
        _, config, data = read_completed_run(self.new)
        changed = copy.deepcopy(data)
        changed["tasks"]["cls"]["subset_ids_sha256"]["validation"] = "different-validation"
        with self.assertRaisesRegex(ValueError, "data/validation differs"):
            validate_extension(self.old, config, changed, build_jobs(config), ROOT)


if __name__ == "__main__":
    unittest.main()
