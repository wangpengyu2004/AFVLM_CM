"""ACED algebra, lifecycle and config regressions; no VLM/training/forward pass."""

from __future__ import annotations

import ast
import copy
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from afl_vlm.config import load_config, validate_config
from afl_vlm.evaluation_schedule import PeriodicEvaluationSchedule
from afl_vlm.execution import resolve_runtime_backend
from afl_vlm.federation.initialization import initial_server_update_record
from afl_vlm.federation.server import FederatedServer
from afl_vlm.federation.types import ClientContext, ServerContext, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import create_method, method_names, profile_method_names
from afl_vlm.models.base import TrainConfig, clone_state, subtract

ROOT = Path(__file__).resolve().parents[1]


def upload(client="a", gradient=2.0, local_round=0, base=0.0, version=1, samples=1):
    anchor = {"x": torch.tensor([base], dtype=torch.float32)}
    local = {"x": anchor["x"] + 999}  # Deliberately unrelated AdamW endpoint.
    return Update(
        f"{client}-r{local_round}",
        client,
        "cls",
        "dataset",
        samples,
        local_round,
        version,
        float(local_round + 1),
        anchor,
        local,
        subtract(local, anchor),
        42,
        "hash",
        [],
        7,
        mean_gradient={"x": torch.tensor([gradient], dtype=torch.float32)},
    )


def initial_upload(client, gradient, initial):
    row = upload(client, gradient if isinstance(gradient, (float, int)) else 0, -1, version=0)
    row.base_state = clone_state(initial)
    row.local_state = clone_state(initial)
    row.delta = subtract(initial, initial)
    row.optimizer_steps = 1
    row.metadata = {"gradient_estimation_only": True}
    if not isinstance(gradient, (float, int)):
        row.mean_gradient = gradient
    return row


class ACEDChecks(unittest.TestCase):
    def setup_method(self, params=None, gradients=(1.0, 3.0), initial=None, bootstrap=True):
        ids = tuple(chr(ord("a") + index) for index in range(len(gradients)))
        state = initial if initial is not None else {"x": torch.zeros(1)}
        method = create_method("aced", {"server_lr": 0.1, **(params or {})})
        method.configure_server(state, [SimpleNamespace(id=cid) for cid in ids])
        method.configure_schedule([SimpleNamespace(client_id=cid) for cid in ids] * 10)
        server = FederatedServer(None, method, len(ids), state)
        if bootstrap:
            for cid, value in zip(ids, gradients, strict=True):
                method.initialize_client_gradient(initial_upload(cid, value, state))
            method.finalize_initial_gradients()
            server.initialize_from_gradients()
        return method, server

    def test_full_client_initialization_and_bootstrap_do_not_count_uploads(self):
        method, server = self.setup_method(bootstrap=False)
        with self.assertRaisesRegex(RuntimeError, "completed initialization"):
            server.initialize_from_gradients()
        method.initialize_client_gradient(initial_upload("a", 1, server.state))
        with self.assertRaisesRegex(ValueError, "every client"):
            method.finalize_initial_gradients()
        method.initialize_client_gradient(initial_upload("b", 3, server.state))
        method.finalize_initial_gradients()
        record = initial_server_update_record(server)["server_initialization"]
        self.assertAlmostEqual(float(server.state["x"]), -0.2)
        self.assertEqual(
            (server.version, server.received_updates, server.accepted_updates), (1, 0, 0)
        )
        self.assertFalse(record["counted_as_upload"])
        self.assertEqual(record["gradient_observations"], 2)
        self.assertEqual(record["scheduled_server_iterations"], 21)
        with self.assertRaisesRegex(RuntimeError, "precede"):
            server.initialize_from_gradients()

    def test_independent_reference_algorithm_a1_and_true_origin_versions(self):
        method, server = self.setup_method({"delay_threshold": 1}, gradients=(1, 3, 5))
        cache = dict(zip(("a", "b", "c"), (1, 3, 5), strict=True))
        stamps = dict.fromkeys(cache, 1)
        expected, clock = -0.3, 1
        rounds = dict.fromkeys(cache, 0)
        for cid, grad in (("a", 2), ("a", 4), ("a", 6), ("b", 8), ("c", 10), ("b", 12)):
            cache[cid] = grad
            active = [key for key in cache if clock - stamps[key] <= 1]
            if active:
                expected -= 0.1 * sum(cache[key] for key in active) / len(active)
            result = server.receive(upload(cid, grad, rounds[cid], version=1))[0]
            rounds[cid] += 1
            self.assertEqual(result.metadata["active_clients"], active)
            self.assertEqual(result.metadata["algorithm_clock_before"], clock)
            self.assertAlmostEqual(float(server.state["x"]), expected, places=5)
            self.assertEqual(result.metadata["cached_gradient_base_versions"][cid], 1)
            stamps[cid] = clock + 1
            self.assertEqual(method.activity_versions, stamps)
            clock += 1

    def test_large_threshold_is_ace_and_no_sample_or_optimizer_step_weight(self):
        _, server = self.setup_method({"delay_threshold": 100})
        row = upload("a", 7, samples=99999)
        row.optimizer_steps = 200
        result = server.receive(row)[0]
        self.assertAlmostEqual(float(server.state["x"]), -0.7)
        self.assertEqual(result.metadata["active_clients"], ["a", "b"])
        self.assertEqual(result.contributing_update_ids, ["a-r0"])
        # Reuse of a-r0 in this next mean must not count that upload twice.
        server.receive(upload("b", 9))
        self.assertAlmostEqual(float(server.state["x"]), -1.5)
        self.assertEqual(server.accepted_updates, 2)

    def test_age_boundary_and_post_aggregation_rejoin(self):
        _, server = self.setup_method({"delay_threshold": 1})
        server.receive(upload("a", 2, 0))  # t1: all active.
        boundary = server.receive(upload("a", 4, 1))[0]  # t2: b age1 included.
        self.assertIn("b", boundary.metadata["active_clients"])
        excluded = server.receive(upload("b", 8, 0))[0]  # t3: b age2 excluded.
        self.assertFalse(excluded.metadata["arriving_client_included"])
        self.assertTrue(excluded.metadata["rejoin_after_this_iteration"])
        self.assertEqual(excluded.contributing_update_ids, [])
        self.assertEqual(excluded.metadata["pending_cached_update_ids"], {"b": "b-r0"})
        resumed = server.receive(upload("a", 6, 2))[0]
        self.assertEqual(resumed.contributing_update_ids, ["a-r2", "b-r0"])
        self.assertEqual(resumed.metadata["cached_gradient_base_versions"]["b"], 1)
        self.assertGreater(resumed.metadata["cached_gradient_true_staleness"]["b"], 1)
        self.assertEqual(server.accepted_updates, 4)

    def test_empty_set_advances_logical_clock_without_model_update_and_can_rejoin(self):
        method, server = self.setup_method({"delay_threshold": 1})
        method.clock = 5
        before = clone_state(server.state)
        skipped = server.receive(upload("a", 2))[0]
        self.assertEqual((server.version, server.accepted_updates, method.clock), (1, 0, 6))
        self.assertEqual(skipped.applied_weight, 0)
        self.assertTrue(skipped.metadata["empty_active_set"])
        torch.testing.assert_close(server.state["x"], before["x"])
        resumed = server.receive(upload("b", 6))[0]
        self.assertEqual(resumed.metadata["active_clients"], ["a"])
        self.assertEqual(resumed.contributing_update_ids, ["a-r0"])
        self.assertEqual(server.accepted_updates, 1)

    def test_pending_overwrite_and_final_pending_are_explicit_not_extra_training(self):
        method, server = self.setup_method({"delay_threshold": 1})
        method.clock = 5
        server.receive(upload("a", 2))
        result = server.receive(upload("a", 7, 1))[0]
        self.assertEqual(result.metadata["overwritten_unincorporated_update_id"], "a-r0")
        self.assertEqual(result.contributing_update_ids, ["a-r1"])
        method.clock += 10
        server.receive(upload("b", 3))
        self.assertEqual(method.pending_ids, {"b": "b-r0"})
        version = server.version
        self.assertEqual(server.finish(), [])
        self.assertEqual(server.version, version)
        self.assertEqual(method.state_dict()["pending_ids"], {"b": "b-r0"})

    def test_evaluation_upload_budget_excludes_bootstrap_and_cache_reuse(self):
        _, server = self.setup_method({"delay_threshold": 100})
        schedule = PeriodicEvaluationSchedule(
            unit="incorporated_client_updates", interval=10, next_threshold=10
        )
        milestones = []
        for index in range(20):
            server.receive(upload("ab"[index % 2], 2, index // 2))
            if schedule.observe(server.accepted_updates):
                milestones.append((server.accepted_updates, server.version))
        self.assertEqual(milestones, [(10, 11), (20, 21)])

    def test_initializer_rejects_changed_origin_delta_missing_and_duplicate(self):
        method, server = self.setup_method(bootstrap=False)
        original = initial_upload("a", 1, server.state)
        for field, value in (("optimizer_steps", 2), ("mean_gradient", None), ("local_round", 0)):
            bad = copy.deepcopy(original)
            setattr(bad, field, value)
            with self.assertRaises(ValueError):
                method.initialize_client_gradient(bad)
        changed = copy.deepcopy(original)
        changed.local_state["x"] += 1
        with self.assertRaisesRegex(ValueError, "origins"):
            method.initialize_client_gradient(changed)
        method.initialize_client_gradient(original)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            method.initialize_client_gradient(original)

    def test_arrival_validation_no_delta_surrogate(self):
        method, server = self.setup_method()
        for field, value in (
            ("mean_gradient", None),
            ("optimizer_steps", 0),
            ("base_version", 2),
            ("client_id", "unknown"),
            ("local_round", -1),
        ):
            bad = upload()
            setattr(bad, field, value)
            with self.assertRaises(ValueError):
                method.on_arrival(bad, server.context())
        for value in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                method.on_arrival(upload(gradient=value), server.context())
        server.receive(upload())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            server.receive(upload())
        bad = upload(local_round=0)
        bad.update_id = "another-id"
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            method.on_arrival(bad, server.context())

    def test_lr_and_plan_budget_parameters(self):
        method, _ = self.setup_method({"server_lr": None})
        self.assertAlmostEqual(method.server_lr, 0.2 * math.sqrt(2 / 21))
        with self.assertRaisesRegex(ValueError, "budget"):
            method.configure_schedule([SimpleNamespace(client_id=cid) for cid in "ab"])
        for params in (
            {"server_lr": 0},
            {"server_lr": float("nan")},
            {"lr_scale": -1},
            {"delay_threshold": 0},
            {"delay_threshold": 1.5},
            {"cache_bits": 4},
            {"cache_seed": -1},
            {"made_up": 1},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                create_method("aced", params)

    def test_ordinary_local_config_untouched_initialization_one_real_batch(self):
        method = create_method("aced")
        config = TrainConfig(1, 4, 4, 2e-5, 2048, 42, planned_optimizer_steps=28)
        self.assertIs(method.configure_local_training(config), config)
        context = ClientContext("a", "cls", "data", 2000, -1, 0, 0, 42)
        method.load_client_runtime_state({"initial_gradient_only": True}, context)
        probe = method.configure_local_training(config)
        self.assertEqual((probe.max_local_steps, probe.planned_optimizer_steps), (1, 1))
        self.assertTrue(probe.gradient_estimation_only)
        self.assertTrue(probe.collect_mean_gradient)
        for field in (
            "local_epochs",
            "batch_size",
            "gradient_accumulation",
            "learning_rate",
            "max_text_length",
            "optimizer",
            "batch_size_scope",
            "reference_world_size",
        ):
            self.assertEqual(getattr(probe, field), getattr(config, field))
        method.load_client_runtime_state({}, context)
        self.assertIs(method.configure_local_training(config), config)
        self.assertIsNone(config.max_local_steps)

    def test_multiple_lora_shapes_and_state_roundtrip(self):
        initial = {"lora_A": torch.zeros(2, 3), "lora_B": torch.zeros(4, 2)}
        gradients = [
            {key: torch.ones_like(value) * v for key, value in initial.items()} for v in (1, 3)
        ]
        for bits in (32, 8):
            method, server = self.setup_method({"cache_bits": bits}, gradients, initial)
            for key in initial:
                torch.testing.assert_close(server.state[key], torch.full_like(initial[key], -0.2))
            restored = create_method("aced")
            restored.load_state_dict(method.state_dict())
            row = upload()
            row.mean_gradient = gradients[0]
            one = method.on_arrival(row, server.context())[0]
            two = restored.on_arrival(row, server.context())[0]
            for key in initial:
                self.assertEqual(one.new_state[key].shape, initial[key].shape)
                torch.testing.assert_close(one.new_state[key], two.new_state[key])
            restored.cache.clear()
            self.assertEqual(set(method.cache), {"a", "b"})
            bad = method.state_dict()
            bad["server_lr"] = 2
            with self.assertRaisesRegex(ValueError, "server_lr"):
                create_method("aced").load_state_dict(bad)

    def test_quantization_private_rng_seeded_packed_and_unbiased(self):
        method = create_method("aced", {"cache_bits": 8})
        vector = torch.linspace(-2, 3, 1003).square()
        before = torch.random.get_rng_state().clone()
        encoded = method._encode(vector, "same")
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(encoded["codes"].dtype, torch.uint8)
        self.assertLess(method._storage_bytes(encoded), vector.numel() * 4)
        again = method._encode(vector, "same")
        self.assertTrue(torch.equal(encoded["codes"], again["codes"]))
        total = torch.zeros_like(vector)
        for index in range(500):
            total += method._decode(method._encode(vector, str(index)), vector)
        self.assertLess(float((total / 500 - vector).abs().max()), 0.006)
        constant = torch.ones(5) * 2.5
        torch.testing.assert_close(
            method._decode(method._encode(constant, "constant"), constant), constant
        )

    def test_existing_algorithms_bootstrap_hook_noop(self):
        context = ServerContext({"x": torch.zeros(1)}, 0, 2)
        for name in method_names():
            if name == "aced":
                continue
            method = create_method(name)
            self.assertIs(type(method).initial_server_update, Method.initial_server_update, name)
            self.assertIsNone(method.initial_server_update(context), name)
        # Existing initial-gradient algorithms keep server version zero and state.
        for name in ("masfl", "adamasfl"):
            method = create_method(name)
            server = FederatedServer(None, method, 2, context.global_state)
            self.assertEqual(initial_server_update_record(server), {})
            self.assertEqual((server.version, server.accepted_updates), (0, 0))
            torch.testing.assert_close(server.state["x"], context.global_state["x"])

    def test_bootstrap_integration_is_capability_guarded_in_both_runners(self):
        for name in ("runner.py", "parallel_runner.py"):
            tree = ast.parse((ROOT / "code/afl_vlm" / name).read_text(encoding="utf-8"))
            calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "initial_server_update_record"
            ]
            self.assertEqual(len(calls), 1, name)
            guarded = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.If)
                and ast.unparse(node.test) == "method.capabilities.requires_initial_gradient"
            ]
            self.assertTrue(any(calls[0] in list(ast.walk(node)) for node in guarded), name)

    def test_pending_quantized_state_roundtrip_preserves_delayed_acceptance(self):
        method, server = self.setup_method({"cache_bits": 8, "delay_threshold": 1})
        method.clock = 5
        server.receive(upload("a", 2))
        restored = create_method("aced")
        restored.load_state_dict(method.state_dict())
        self.assertEqual(restored.pending_ids, {"a": "a-r0"})
        left = method.on_arrival(upload("b", 6), server.context())[0]
        right = restored.on_arrival(upload("b", 6), server.context())[0]
        self.assertEqual(left.contributing_update_ids, ["a-r0"])
        self.assertEqual(right.contributing_update_ids, left.contributing_update_ids)
        torch.testing.assert_close(left.new_state["x"], right.new_state["x"])

    def test_explicit_ddp_accepted_but_changed_gradient_shape_rejected(self):
        config = load_config(ROOT / "configs/experiments/llava/afvlm_cm/2clients/aced.yaml")
        config["runtime"]["backend"] = "client_ddp"
        validate_config(config)
        method, server = self.setup_method()
        bad = upload()
        bad.mean_gradient = {"x": torch.ones(2)}
        with self.assertRaisesRegex(ValueError, "shapes"):
            method.on_arrival(bad, server.context())
        self.assertEqual(method.arrivals, 0)
        with self.assertRaisesRegex(ValueError, "every configured client"):
            method.configure_schedule([SimpleNamespace(client_id="a")])

    def test_configs_registry_backend_and_shared_training(self):
        self.assertIn("aced", profile_method_names())
        self.assertNotIn("ace", method_names())
        for setting in (2, 5, 10):
            base = ROOT / f"configs/experiments/llava/afvlm_cm/{setting}clients"
            config = load_config(base / "aced.yaml")
            validate_config(config)
            reference = load_config(base / "fedavg.yaml")
            for key in ("training", "model", "dataset", "run", "evaluation", "runtime"):
                self.assertEqual(config[key], reference[key], key)
            self.assertEqual(
                resolve_runtime_backend(config, create_method("aced")), "client_ddp"
            )
            self.assertEqual(config["federation"]["mode"], "asynchronous")


if __name__ == "__main__":
    unittest.main()
