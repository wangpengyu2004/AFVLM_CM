"""Non-training checks: experiment budgets, frozen annotations and honest CSV exports.

These tests never load a checkpoint, execute a forward/backward or initialize DDP.
The CSV fixtures are arithmetic records, not simulated training/metric results.
"""

from __future__ import annotations

import ast
import copy
import csv
import io
import json
import math
import random
import runpy
import tempfile
import unittest
from pathlib import Path

from afl_vlm.config import load_config
from afl_vlm.experiments.rank_probe import (
    EPOCH_FIELDS,
    STEP_FIELDS,
    append_json,
    build_jobs,
    export_probe_results,
    hash_file,
    hash_ids,
    job_config,
    sampler_manifest,
    steps_per_epoch,
    validate_probe_config,
    validation_protocols,
    verify_sources,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/probes/local_rank_probe.yaml"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class RankProbeConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG)

    def test_isolated_eighteen_jobs_with_fixed_ddp_budget_and_scaling(self) -> None:
        jobs = build_jobs(self.config)
        self.assertEqual(len(jobs), 18)
        self.assertEqual(len({(job.task, job.client) for job in jobs}), 6)
        self.assertEqual(len({job.key for job in jobs}), 18)
        self.assertNotIn("inherits", CONFIG.read_text(encoding="utf-8"))
        for forbidden in ("federation", "method", "experiment_profile"):
            self.assertNotIn(forbidden, self.config)
        before = copy.deepcopy(self.config)
        for job in jobs:
            config = job_config(self.config, job)
            training = config["training"]
            self.assertEqual(training["local_epochs"], 5)
            self.assertEqual(training["batch_size"] * training["gradient_accumulation"] * 8, 128)
            self.assertEqual(config["runtime"]["world_size"], 8)
            self.assertEqual(config["model"]["lora"]["alpha"] / job.lora_rank, 2)
            self.assertEqual(training["max_text_length"], 2048)
            self.assertEqual(job.seed, 42)
        self.assertEqual(before, self.config)

    def test_selectors_do_not_repartition_or_change_other_jobs(self) -> None:
        selected = build_jobs(self.config, task="vqa", client="client_0")
        self.assertEqual([job.lora_rank for job in selected], [2, 4, 8])
        self.assertEqual(len(build_jobs(self.config, lora_rank=4)), 6)
        for options in ({"task": "grounding"}, {"seed": 43}, {"client": "client_2"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                build_jobs(self.config, **options)

    def test_rejects_federated_and_step_limited_or_single_card_configuration(self) -> None:
        variants = []
        config = copy.deepcopy(self.config)
        config["method"] = {"name": "local"}
        variants.append(config)
        config = copy.deepcopy(self.config)
        config["training"]["max_local_steps"] = 2
        variants.append(config)
        config = copy.deepcopy(self.config)
        config["runtime"]["world_size"] = 1
        variants.append(config)
        config = copy.deepcopy(self.config)
        config["model"]["lora"]["train_mm_projector"] = True
        variants.append(config)
        for config in variants:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_probe_config(config)

    def test_original_sampler_epoch_tail_and_five_epoch_optimizer_budgets(self) -> None:
        counts = [7006, 5994, 7796, 7204, 5435, 5565]
        self.assertEqual([5 * steps_per_epoch(n, 4, 4) for n in counts],
                         [275, 235, 305, 285, 215, 220])
        self.assertEqual(steps_per_epoch(129, 4, 4), 2)
        # A non-full accumulation window still performs its original optimizer step.
        self.assertEqual(steps_per_epoch(33, 4, 4), 1)
        with self.assertRaises(ValueError):
            steps_per_epoch(0, 4, 4)

    def test_expected_sampler_uses_continuous_shuffle_rng_and_repeat_padding(self) -> None:
        before = random.getstate()
        manifest = sampler_manifest(19, 5, 42, 4, 4)
        self.assertEqual(random.getstate(), before)
        rng = random.Random(42)
        for row in manifest:
            indices = list(range(19))
            rng.shuffle(indices)
            indices.extend(indices[:5])
            self.assertEqual(row["padded_records"], 24)
            self.assertEqual(row["padding_duplicates"], 5)
            self.assertEqual(row["global_indices_sha256"], hash_ids(indices))
            shards = [indices[rank::8] for rank in range(8)]
            self.assertEqual(row["per_rank_indices_sha256"], [hash_ids(s) for s in shards])
        self.assertEqual(len({row["global_indices_sha256"] for row in manifest}), 5)

    def test_validation_subset_is_fixed_from_union_of_both_clients(self) -> None:
        rows = [{"image": "COCO2014/shared.jpg"}, {"image": "./COCO2014\\unseen.jpg"}]
        original = copy.deepcopy(rows)
        result = validation_protocols("vqa", rows, {"COCO2014/shared.jpg"}, self.config)
        self.assertEqual(result["subsets"], {"validation_image_disjoint": [1],
                                            "validation": [0, 1]})
        self.assertEqual(result["overlapping_validation_records"], 1)
        self.assertEqual(rows, original)
        cls_result = validation_protocols("cls", rows, set(), self.config)
        self.assertEqual(cls_result["subsets"], {"validation": [0, 1]})
        with self.assertRaises(ValueError):
            validation_protocols("vqa", rows, {"COCO2014/shared.jpg", "COCO2014/unseen.jpg"},
                                 self.config)

    def test_frozen_source_detects_changed_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "client_0.json"
            path.write_text("[]", encoding="utf-8")
            manifest = {"partition_root": folder, "source_files": {path.name: hash_file(path)}}
            verify_sources(manifest)
            path.write_text('[{"changed": true}]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Annotation changed"):
                verify_sources(manifest)

    def test_runtime_uses_single_continuous_historical_training_call(self) -> None:
        source = (ROOT / "code/afl_vlm/experiments/rank_probe_runtime.py").read_text("utf-8")
        tree = ast.parse(source)
        train_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and node.func.attr == "local_train"]
        self.assertEqual(len(train_calls), 1)
        configs = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == "TrainConfig"]
        self.assertEqual(len(configs), 1)
        keywords = {kw.arg for kw in configs[0].keywords}
        self.assertTrue({"step_hook", "progress_hook", "local_epochs"} <= keywords)
        self.assertTrue({"max_local_steps", "loss_hook", "gradient_hook"}.isdisjoint(keywords))
        self.assertIn("with preserved_training_state(model, torch):", source)
        self.assertIn("merge_evaluation_outputs(", source)
        self.assertIn("model.enable_distributed_data_parallel(local_rank)", source)


class RankProbeExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        config = load_config(CONFIG)
        jobs = build_jobs(config, task="cls")
        self.manifest = {"kind": "local_rank_probe", "status": "completed", "local_epochs": 5,
                         "jobs": [{**job.as_dict(), "status": "completed"} for job in jobs]}
        self.paths = []
        for job in jobs:
            directory = self.root / job.key
            directory.mkdir(parents=True)
            path = directory / "metrics.jsonl"
            self.paths.append(path)
            for epoch in range(1, 6):
                # Arithmetic fixture: Δ24=4 and Δ48=1; client offset must survive pooling.
                value = {2: 10, 4: 14, 8: 15}[job.lora_rank]
                value += epoch + (2 if job.client == "client_1" else 0)
                append_json(path, {**job.as_dict(), "epoch": epoch, "optimizer_steps": epoch * 55,
                    "num_train_samples": 7006, "num_eval_samples": 1000,
                    "protocol": "validation", "primary": True, "metrics": {"accuracy": value},
                    "train_annotation_sha256": "train", "validation_ids_sha256": "validation"})
            step = {key: 1 for key in STEP_FIELDS if key != "job_status"}
            step.update(job.as_dict())
            append_json(directory / "training_steps.jsonl", step)
            epoch_row = {key: 1 for key in EPOCH_FIELDS if key != "job_status"}
            epoch_row.update(job.as_dict())
            append_json(directory / "epoch_training.jsonl", epoch_row)
        write_json(self.root / "manifest.json", self.manifest)

    def test_raw_final_metrics_equal_client_curves_paired_gains_and_losses(self) -> None:
        counts = export_probe_results(self.root)
        self.assertEqual(counts["metric_rows"], 30)
        self.assertEqual(counts["curve_rows"], 15)
        self.assertEqual(counts["gain_rows"], 10)
        final = read_csv(self.root / "final_epoch_metrics.csv")
        self.assertEqual(len(final), 6)
        self.assertTrue(all(row["epoch"] == "5" for row in final))
        curves = read_csv(self.root / "task_rank_curves.csv")
        row = next(row for row in curves if row["lora_rank"] == "2" and row["epoch"] == "5")
        self.assertEqual(float(row["mean"]), 16)
        self.assertEqual(row["complete"], "True")
        gains = read_csv(self.root / "rank_gains.csv")
        self.assertTrue(all(float(row["gain_2_to_4"]) == 4 for row in gains))
        self.assertTrue(all(float(row["gain_4_to_8"]) == 1 for row in gains))
        self.assertTrue(all(float(row["gain_ratio_48_over_24"]) == .25 for row in gains))
        self.assertEqual(len(read_csv(self.root / "training_steps.csv")), 6)
        self.assertEqual(len(read_csv(self.root / "epoch_training.csv")), 6)

    def test_failed_job_keeps_records_but_is_excluded_from_averages(self) -> None:
        self.manifest["status"] = "failed"
        self.manifest["jobs"][0]["status"] = "failed"
        write_json(self.root / "manifest.json", self.manifest)
        export_probe_results(self.root)
        raw = read_csv(self.root / "metrics_long.csv")
        self.assertEqual(sum(row["job_status"] == "failed" for row in raw), 5)
        curves = read_csv(self.root / "task_rank_curves.csv")
        self.assertTrue(all(row["client_count"] == "1" and row["complete"] == "False"
                            for row in curves if row["lora_rank"] == "2"))
        self.assertEqual(len(read_csv(self.root / "rank_gains.csv")), 5)

    def test_zero_or_negative_small_rank_gain_has_no_misleading_ratio(self) -> None:
        for path in self.paths:
            records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
            for record in records:
                record["metrics"]["accuracy"] = {2: 10, 4: 10, 8: 11}[record["lora_rank"]]
            path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        export_probe_results(self.root)
        self.assertTrue(all(row["gain_ratio_48_over_24"] == ""
                            for row in read_csv(self.root / "rank_gains.csv")))

    def test_duplicate_metric_is_rejected(self) -> None:
        path = self.paths[0]
        record = json.loads(path.read_text("utf-8").splitlines()[0])
        append_json(path, record)
        with self.assertRaisesRegex(ValueError, "Duplicate/non-finite"):
            export_probe_results(self.root)

    def test_non_finite_metric_is_rejected(self) -> None:
        path = self.paths[0]
        records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        records[0]["metrics"]["accuracy"] = math.inf
        path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate/non-finite"):
            export_probe_results(self.root)


class RankProbeProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = runpy.run_path(str(ROOT / "scripts/run_local_rank_probe.py"))

    def test_relay_preserves_in_place_refreshes_and_tracebacks_exactly(self) -> None:
        payload = (b"loading model\n\rrank probe total: 0/18\n"
                   b"\rlocal: 1/275\x1b[A\n\rlocal: 2/275\x1b[A"
                   b"\nTraceback (most recent call last):\nRuntimeError: example\n")
        source = io.BufferedReader(io.BytesIO(payload), buffer_size=7)
        terminal, log = io.BytesIO(), io.BytesIO()
        self.launcher["_relay_output"](source, terminal, log)
        self.assertEqual(terminal.getvalue(), payload)
        self.assertEqual(log.getvalue(), payload)
        # The former Popen(text=True) path produced extra lines from these same CRs.
        translated = io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8").read()
        self.assertGreater(translated.count("\n"), payload.count(b"\n"))

    def test_relay_handles_split_utf8_without_inserting_line_breaks(self) -> None:
        payload = ("\r总进度 0/18\n\r训练 1/275\x1b[A" * 1200).encode("utf-8")
        source = io.BufferedReader(io.BytesIO(payload), buffer_size=3)
        terminal, log = io.BytesIO(), io.BytesIO()
        self.launcher["_relay_output"](source, terminal, log)
        self.assertEqual(terminal.getvalue().decode("utf-8"), payload.decode("utf-8"))
        self.assertEqual(log.getvalue(), payload)


if __name__ == "__main__":
    unittest.main()
