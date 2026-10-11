"""Non-training reference-DDP index/step/objective checks; no model forward."""

from __future__ import annotations

import math
import random
import unittest
from pathlib import Path
from unittest.mock import patch

from afl_vlm.batching import (
    detect_reference_world_size,
    freeze_reference_training,
    freeze_runtime_batch_config,
    optimizer_windows,
    planned_optimizer_steps,
    resolve_batch_layout,
)
from afl_vlm.config import load_config, validate_config
from afl_vlm.federation.types import ClientSpec
from afl_vlm.methods.registry import method_names
from afl_vlm.scheduling.train_plan import build_train_plan, optimizer_steps_for_epochs

ROOT = Path(__file__).resolve().parents[1]


def original_rank_windows(indices: list[int], rank: int, world: int = 8) -> list:
    """Independent transcription of the published pre-change B4/G4 sampler."""
    padded = list(indices)
    per_rank = math.ceil(len(padded) / world)
    missing = per_rank * world - len(padded)
    if missing:
        padded.extend((padded * math.ceil(missing / len(padded)))[:missing])
    local = padded[rank : per_rank * world : world]
    batches = [tuple(local[offset : offset + 4]) for offset in range(0, len(local), 4)]
    return [tuple(batches[offset : offset + 4]) for offset in range(0, len(batches), 4)]


class ReferenceBatchBudgetTests(unittest.TestCase):
    def training(self) -> dict:
        return {
            "batch_size": 4,
            "gradient_accumulation": 4,
            "batch_size_scope": "reference_ddp",
            "reference_world_size": 8,
        }

    def test_existing_ddp_layout_is_not_changed(self) -> None:
        layout = resolve_batch_layout(self.training(), 8)
        self.assertEqual(layout.per_device_batch_size, 4)
        self.assertEqual(layout.physical_gradient_accumulation, 4)
        self.assertEqual(layout.effective_batch_size, 128)
        self.assertEqual(layout.reference_world_size, 8)

    def test_worker_uses_batch4_accumulation32(self) -> None:
        layout = resolve_batch_layout(self.training(), 1)
        self.assertEqual(layout.per_device_batch_size, 4)
        self.assertEqual(layout.physical_gradient_accumulation, 32)
        self.assertEqual(layout.effective_batch_size, 128)

    def test_every_ddp_batch_matches_original_including_padded_tails(self) -> None:
        for size in (1, 3, 7, 8, 9, 15, 16, 17, 127, 128, 129, 432, 6071, 10000):
            indices = list(range(size))
            random.Random(42).shuffle(indices)
            for rank in range(8):
                with self.subTest(size=size, rank=rank):
                    actual = list(
                        optimizer_windows(indices, resolve_batch_layout(self.training(), 8), rank)
                    )
                    self.assertEqual(
                        [tuple(batch.indices for batch in window) for window in actual],
                        original_rank_windows(indices, rank),
                    )

    def test_worker_step_contains_exactly_all_reference_rank_micro_batches(self) -> None:
        for size in (1, 7, 33, 127, 128, 129, 6071, 10000):
            indices = list(range(size))
            random.Random(99).shuffle(indices)
            reference = [original_rank_windows(indices, rank) for rank in range(8)]
            worker = list(optimizer_windows(indices, resolve_batch_layout(self.training(), 1)))
            self.assertEqual(len(worker), len(reference[0]))
            for step, window in enumerate(worker):
                expected = tuple(batch for shard in reference for batch in shard[step])
                self.assertEqual(tuple(batch.indices for batch in window), expected)
                self.assertTrue(all(len(batch.indices) <= 4 for batch in window))

    def test_loss_and_gradient_weighting_equal_reference_ddp_algebra(self) -> None:
        # A deterministic scalar stands for each ORIGINAL micro-batch loss/gradient.
        # Average each rank's micro-batches, then ranks, vs. one flattened window.
        for size in (1, 15, 127, 128, 129, 432):
            indices = list(range(size))
            shards = [original_rank_windows(indices, rank) for rank in range(8)]
            worker = list(optimizer_windows(indices, resolve_batch_layout(self.training())))

            def batch_value(batch: tuple[int, ...]) -> float:
                return sum((i % 11 - 5) * (i + 1) for i in batch) / (1 + len(batch))

            for step, window in enumerate(worker):
                ddp = (
                    sum(
                        sum(batch_value(batch) for batch in shard[step]) / len(shard[step])
                        for shard in shards
                    )
                    / 8
                )
                single = sum(batch_value(batch.indices) for batch in window) / len(window)
                self.assertAlmostEqual(single, ddp)

    def test_padding_contributions_match_existing_experiment(self) -> None:
        for size in (1, 17, 129, 6071):
            windows = optimizer_windows(list(range(size)), resolve_batch_layout(self.training()))
            count = sum(len(batch.indices) for window in windows for batch in window)
            self.assertEqual(count, 8 * math.ceil(size / 8))

    def test_epoch_optimizer_steps_match_existing_ddp(self) -> None:
        for size in (1, 16, 127, 128, 129, 432, 6071, 10000):
            for world in (1, 8):
                self.assertEqual(
                    planned_optimizer_steps(size, 2, self.training(), world),
                    2 * math.ceil(math.ceil(math.ceil(size / 8) / 4) / 4),
                )

    def test_tiny_tail_uses_actual_accumulation_not_fixed32(self) -> None:
        window = next(optimizer_windows([0], resolve_batch_layout(self.training())))
        self.assertEqual(len(window), 8)
        self.assertEqual([batch.indices for batch in window], [(0,)] * 8)

    def test_legacy_profile_without_scope_is_not_reinterpreted(self) -> None:
        legacy = {"batch_size": 4, "gradient_accumulation": 4}
        self.assertEqual(resolve_batch_layout(legacy, 8).effective_batch_size, 128)
        self.assertEqual(resolve_batch_layout(legacy, 1).effective_batch_size, 16)
        self.assertEqual(planned_optimizer_steps(129, 1, legacy, 1), 9)

    def test_incompatible_real_gpu_count_is_rejected(self) -> None:
        for world in (0, 2, 4, 16):
            with self.assertRaises(ValueError):
                resolve_batch_layout(self.training(), world)
        with self.assertRaises(ValueError):
            list(optimizer_windows([0], resolve_batch_layout(self.training(), 8), 8))

    def test_virtual_nominal_plan_budget_stays_separate(self) -> None:
        self.assertEqual(optimizer_steps_for_epochs(6071, 1, 4, 4), 380)
        self.assertEqual(planned_optimizer_steps(6071, 1, self.training(), 8), 48)
        self.assertEqual(planned_optimizer_steps(6071, 1, self.training(), 1), 48)

    def test_compass_step_cost_conversion_does_not_create_virtual_speedup(self) -> None:
        clients = [
            ClientSpec("slow", "vqa", "AOKVQA", 128, 0.5, 0.25, 0.0, 1.0),
            ClientSpec("fast", "cls", "ImageNet-R", 256, 2.0, 0.5, 0.0, 1.0),
        ]
        params = {"min_local_steps": 1, "max_local_steps": 10000}
        old = build_train_plan(clients, 2, 1, 4, 4, "semi_asynchronous", params)
        matched = build_train_plan(
            clients,
            2,
            1,
            4,
            32,
            "semi_asynchronous",
            params,
            optimizer_step_cost_scale=8,
        )
        # Dynamic scheduler bounds are physical optimizer steps. The cost
        # conversion must not silently reduce those bounds or alter profiles.
        old_initial = sorted((e for e in old if e.local_round == 0), key=lambda e: e.client_id)
        matched_initial = sorted(
            (e for e in matched if e.local_round == 0), key=lambda e: e.client_id
        )
        for before, after in zip(old_initial, matched_initial, strict=True):
            self.assertEqual(before.client_id, after.client_id)
            self.assertEqual(before.local_steps, after.local_steps)
            self.assertEqual(8 * before.estimated_train_time, after.estimated_train_time)
            self.assertEqual(before.start_time, after.start_time)
            self.assertEqual(
                after.arrival_time,
                after.start_time + after.estimated_train_time + before.network_delay,
            )
            self.assertEqual(before.speed_factor, after.speed_factor)
            self.assertEqual(before.network_delay, after.network_delay)
            self.assertEqual(before.group_id, after.group_id)

    def test_invalid_optimizer_step_cost_scale_is_rejected(self) -> None:
        client = ClientSpec("k", "vqa", "AOKVQA", 128, 1.0, 0.0, 0.0, 1.0)
        for scale in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                build_train_plan(
                    [client],
                    1,
                    1,
                    4,
                    4,
                    "asynchronous",
                    optimizer_step_cost_scale=scale,
                )

    def test_all_method_templates_share_reference128(self) -> None:
        reference = load_config(ROOT / "configs/base.yaml")["training"]
        self.assertNotIn("batch_size_scope", reference)
        self.assertNotIn("reference_world_size", reference)
        for setting in (2, 5, 10):
            for method in method_names():
                path = (
                    ROOT
                    / "configs/experiments/llava/afvlm_cm"
                    / f"{setting}clients"
                    / f"{method}.yaml"
                )
                config = load_config(path)
                validate_config(config, check_paths=False)
                self.assertEqual(config["training"], reference)
                frozen = freeze_reference_training(config["training"], 8)
                for world in (1, 8):
                    self.assertEqual(
                        resolve_batch_layout(frozen, world).effective_batch_size,
                        reference["batch_size"] * reference["gradient_accumulation"] * 8,
                    )

    def test_visible_gpu_detection_uses_cuda_count(self) -> None:
        with patch("torch.cuda.device_count", return_value=8) as count:
            self.assertEqual(detect_reference_world_size(), 8)
            count.assert_called_once_with()

    def test_no_cuda_does_not_silently_freeze_reference1(self) -> None:
        with patch("torch.cuda.device_count", return_value=0):
            with self.assertRaisesRegex(ValueError, "No visible CUDA GPU"):
                detect_reference_world_size()

    def test_freeze_preserves_training_controls_without_mutating_input(self) -> None:
        original = {"batch_size": 4, "gradient_accumulation": 4, "local_epochs": 1}
        frozen = freeze_reference_training(original, 8)
        self.assertEqual(
            frozen, {**original, "batch_size_scope": "reference_ddp", "reference_world_size": 8}
        )
        self.assertNotIn("batch_size_scope", original)
        with self.assertRaises(ValueError):
            freeze_reference_training(original, 0)

    def test_runtime_freezes_editable_config_once_before_worker_dispatch(self) -> None:
        original = {"training": {"batch_size": 4, "gradient_accumulation": 4}}
        with patch("afl_vlm.batching.detect_reference_world_size", return_value=8) as detect:
            frozen = freeze_runtime_batch_config(original)
            self.assertEqual(frozen["training"]["reference_world_size"], 8)
            self.assertEqual(resolve_batch_layout(frozen["training"]).effective_batch_size, 128)
            self.assertIs(freeze_runtime_batch_config(frozen), frozen)
            detect.assert_called_once_with()
        self.assertNotIn("reference_world_size", original["training"])

    def test_legacy_immutable_profile_never_detects_or_changes_semantics(self) -> None:
        legacy = {
            "training": {"batch_size": 4, "gradient_accumulation": 4},
            "experiment_profile": {"immutable_snapshot": True},
        }
        with patch("afl_vlm.batching.detect_reference_world_size") as detect:
            self.assertIs(freeze_runtime_batch_config(legacy), legacy)
            detect.assert_not_called()
        self.assertEqual(resolve_batch_layout(legacy["training"], 1).effective_batch_size, 16)
        self.assertEqual(resolve_batch_layout(legacy["training"], 8).effective_batch_size, 128)

    def test_frozen_profile_on_another_machine_keeps_reference8(self) -> None:
        profile = {"training": self.training(), "experiment_profile": {"immutable_snapshot": True}}
        with patch("afl_vlm.batching.detect_reference_world_size", return_value=4) as detect:
            self.assertIs(freeze_runtime_batch_config(profile), profile)
            detect.assert_not_called()
        self.assertEqual(resolve_batch_layout(profile["training"], 1).effective_batch_size, 128)
        with self.assertRaises(ValueError):
            resolve_batch_layout(profile["training"], 4)

    def test_static_yaml_validation_does_not_detect_cuda(self) -> None:
        with patch("afl_vlm.batching.detect_reference_world_size") as detect:
            config = load_config(ROOT / "configs/experiments/llava/afvlm_cm/2clients/ours.yaml")
            validate_config(config, check_paths=False)
            detect.assert_not_called()

    def test_configured_scope_and_reference_are_propagated(self) -> None:
        for name in ("runner.py", "parallel_runner.py", "federation/client.py"):
            source = (ROOT / "code/afl_vlm" / name).read_text(encoding="utf-8")
            self.assertIn("batch_size_scope=", source)
            self.assertIn("reference_world_size=", source)

    def test_original_backward_and_collation_are_retained(self) -> None:
        source = (ROOT / "code/afl_vlm/models/llava15.py").read_text(encoding="utf-8")
        self.assertIn("(loss / len(window)).backward()", source)
        self.assertNotIn("text_sequences", source)
        self.assertNotIn("torch.distributed.all_reduce", source)


if __name__ == "__main__":
    unittest.main()
