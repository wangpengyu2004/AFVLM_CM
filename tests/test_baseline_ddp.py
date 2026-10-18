"""Method-owned DDP algebra/config/AST checks; no forward/backward or training."""

from __future__ import annotations

import ast
import copy
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from afl_vlm.batching import optimizer_windows, resolve_batch_layout
from afl_vlm.config import load_config, validate_config
from afl_vlm.execution import resolve_runtime_backend
from afl_vlm.federation.types import ClientContext
from afl_vlm.methods import fedasmu_ddp
from afl_vlm.methods.registry import create_method, method_names
from afl_vlm.models.base import TrainConfig

ROOT = Path(__file__).resolve().parents[1]
NEW_DDP = ("fedcompass", "fedasmu", "masfl", "adamasfl", "aced", "unifed_lora")


class BaselineDDPChecks(unittest.TestCase):
    def context(self, initial=False):
        return ClientContext(
            "cls/client_0", "cls", "ImageNet-R", 6071, -1 if initial else 0, 0, 10, 42
        )

    def config(self):
        return TrainConfig(
            1,
            4,
            4,
            2e-5,
            2048,
            42,
            planned_optimizer_steps=48,
            batch_size_scope="reference_ddp",
            reference_world_size=8,
            context={"client": self.context(True)},
            collect_mean_gradient=True,
        )

    def test_all_compatible_baselines_select_ddp_and_pilot_stays_safe(self):
        runtime = {"runtime": {"backend": "client_parallel", "executor_policy": "capability"}}
        for name in method_names():
            self.assertEqual(
                resolve_runtime_backend(runtime, create_method(name)),
                "client_parallel" if name == "pilot" else "client_ddp",
                name,
            )
        for name in NEW_DDP:
            for setting in (2, 5, 10):
                config = load_config(
                    ROOT / f"configs/experiments/llava/afvlm_cm/{setting}clients/{name}.yaml"
                )
                validate_config(config)
                config["runtime"]["backend"] = "client_ddp"
                validate_config(config)

    def test_fixed_client_parallel_is_still_available(self):
        runtime = {"runtime": {"backend": "client_parallel", "executor_policy": "fixed"}}
        for name in NEW_DDP:
            self.assertEqual(
                resolve_runtime_backend(runtime, create_method(name)), "client_parallel"
            )

    def test_real_rank_initialization_preserves_reference_budget_and_no_step(self):
        for name in ("masfl", "adamasfl", "aced"):
            method = create_method(name)
            method.load_client_runtime_state({"initial_gradient_only": True}, self.context(True))
            config = self.config()
            before = copy.deepcopy(config)
            shards = []
            for rank in range(8):
                with (
                    patch("torch.distributed.is_available", return_value=True),
                    patch("torch.distributed.is_initialized", return_value=True),
                    patch("torch.distributed.get_rank", return_value=rank),
                    patch("torch.distributed.get_world_size", return_value=8),
                ):
                    actual = method.configure_local_training(config)
                self.assertEqual(actual.distributed_rank, rank)
                self.assertEqual(actual.distributed_world_size, 8)
                self.assertTrue(actual.gradient_estimation_only)
                self.assertEqual(actual.optimizer, "adamw" if name == "aced" else "sgd")
                self.assertEqual(actual.planned_optimizer_steps, 1 if name == "aced" else 48)
                self.assertEqual(actual.max_local_steps, 1 if name == "aced" else None)
                layout = resolve_batch_layout(
                    {
                        "batch_size": actual.batch_size,
                        "gradient_accumulation": actual.gradient_accumulation,
                        "batch_size_scope": actual.batch_size_scope,
                        "reference_world_size": actual.reference_world_size,
                    },
                    8,
                )
                self.assertEqual(layout.effective_batch_size, 128)
                window = next(optimizer_windows(list(range(6071)), layout, rank))
                shards.append({index for batch in window for index in batch.indices})
            self.assertEqual(len(set.union(*shards)), 128)
            self.assertEqual(config, before)

    def test_ordinary_masfl_still_sgd_and_aced_local_config_unchanged(self):
        config = self.config()
        for name in ("masfl", "adamasfl"):
            actual = create_method(name).configure_local_training(config)
            self.assertEqual(actual.optimizer, "sgd")
            self.assertFalse(actual.gradient_estimation_only)
            self.assertEqual(actual.gradient_accumulation, 4)
        self.assertIs(create_method("aced").configure_local_training(config), config)

    def test_reduced_meta_observations_use_equal_rank_mean_not_repeated_updates(self):
        # Emulate eight ranks' SUM of three loss/derivative scalars and records.
        peers = torch.tensor([14.0, 21.0, -7.0, 28.0])

        def sum_peers(value):
            value.add_(peers)

        local = torch.tensor([1.0, 2.0, -1.0])
        before = local.clone()
        with (
            patch("torch.distributed.get_world_size", return_value=8),
            patch("torch.distributed.all_reduce", side_effect=sum_peers) as reduce,
        ):
            result, records, world = fedasmu_ddp.reduce_observations(local, 4)
        torch.testing.assert_close(result, torch.tensor([15 / 8, 23 / 8, -1.0]))
        torch.testing.assert_close(local, before)
        self.assertEqual((records, world), (32, 8))
        reduce.assert_called_once()

    def test_nonfinite_feedback_fails_after_collective_for_all_ranks(self):
        with (
            patch("torch.distributed.get_world_size", return_value=8),
            patch("torch.distributed.all_reduce") as reduce,
            self.assertRaisesRegex(ValueError, "Non-finite"),
        ):
            fedasmu_ddp.reduce_observations(torch.tensor([1, float("nan"), 2]), 4)
        reduce.assert_called_once()

    def test_policy_selected_only_on_rank_zero_and_broadcast_to_every_rank(self):
        decision = {"kind": "lstm", "step": 4, "slot": 1}
        controller = SimpleNamespace(select=Mock(return_value=decision))
        for rank in (0, 7):
            controller.select.reset_mock()

            def deliver(message, src):
                self.assertEqual(src, 0)
                message[0] = decision

            with (
                patch("afl_vlm.methods.fedasmu_ddp.distributed_active", return_value=True),
                patch("torch.distributed.get_rank", return_value=rank),
                patch("torch.distributed.broadcast_object_list", side_effect=deliver),
            ):
                result = fedasmu_ddp.select_decision(controller, self.context(), 48)
            self.assertEqual(result, decision)
            self.assertEqual(controller.select.call_count, int(rank == 0))

    def test_non_owner_keeps_local_controls_but_does_not_update_rl_policy_twice(self):
        for owner in (False, True):
            method = create_method("fedasmu")
            method.controller = SimpleNamespace(observe=Mock())
            decision = {"kind": "q_learning", "step": 4}
            method.decisions[self.context().client_id, 0] = decision
            row = SimpleNamespace(
                metadata={
                    "fedasmu_local": {
                        "controls": {"gamma": 0.9, "upsilon": 0.4},
                        "reward": 1.0,
                    }
                }
            )
            with patch("afl_vlm.methods.fedasmu_ddp.controller_owner", return_value=owner):
                method.prepare_upload(row, self.context())
            self.assertEqual(method.controller.observe.call_count, int(owner))
            self.assertEqual(method.local_controls[self.context().client_id]["gamma"], 0.9)
            self.assertEqual(row.metadata["fedasmu_request_decision"], decision)

    def test_feedback_bridge_is_instance_local_and_preserves_single_gpu_call(self):
        first = SimpleNamespace(
            _ddp_enabled=False, mixing_objective_feedback=Mock(return_value={"x": 1})
        )
        second = SimpleNamespace(_ddp_enabled=False, mixing_objective_feedback=Mock())
        original = first.mixing_objective_feedback
        untouched = second.mixing_objective_feedback
        fedasmu_ddp.install_feedback(first)
        bound = first.mixing_objective_feedback
        fedasmu_ddp.install_feedback(first)
        self.assertIs(first.mixing_objective_feedback, bound)
        self.assertEqual(first.mixing_objective_feedback("left", "right", 0.5), {"x": 1})
        original.assert_called_once_with("left", "right", 0.5)
        self.assertIs(second.mixing_objective_feedback, untouched)
        first._ddp_enabled = True
        with patch("afl_vlm.methods.fedasmu_ddp.feedback", return_value={"ddp": True}) as oracle:
            self.assertEqual(first.mixing_objective_feedback("l", "r", 0.5), {"ddp": True})
        oracle.assert_called_once_with(first, "l", "r", 0.5)

    def test_feedback_never_calls_ddp_forward_backward_or_optimizer(self):
        # Inspect production code rather than constructing a fake VLM forward.
        tree = ast.parse(inspect.getsource(fedasmu_ddp.feedback))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        forbidden = {"backward", "step", "zero_grad", "enable_distributed_data_parallel"}
        self.assertFalse(
            any(
                isinstance(node.func, ast.Attribute) and node.func.attr in forbidden
                for node in calls
            )
        )
        self.assertEqual(
            sum(isinstance(node.func, ast.Name) and node.func.id == "network" for node in calls), 2
        )
        self.assertFalse(
            any(
                isinstance(node.func, ast.Attribute) and node.func.attr == "model" for node in calls
            )
        )
        self.assertTrue(
            any(isinstance(node, ast.Try) and node.finalbody for node in ast.walk(tree))
        )


if __name__ == "__main__":
    unittest.main()
