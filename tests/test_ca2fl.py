"""CA2FL equation/config/state checks; no model, optimizer or training job."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from afl_vlm.config import load_config, validate_config
from afl_vlm.evaluation_schedule import PeriodicEvaluationSchedule
from afl_vlm.execution import resolve_runtime_backend
from afl_vlm.federation.server import FederatedServer
from afl_vlm.federation.types import ClientContext, Update
from afl_vlm.methods.registry import create_method, profile_method_names
from afl_vlm.models.base import TrainConfig, clone_state, subtract

ROOT = Path(__file__).resolve().parents[1]


def upload(client="a", delta=2.0, local_round=0, base=0.0, version=0, samples=1):
    anchor = {"x": torch.tensor([base], dtype=torch.float32)}
    local = {"x": anchor["x"] + delta}
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
        2,
    )


class CA2FLChecks(unittest.TestCase):
    def setup_method(self, params=None, clients=("a", "b", "c"), initial=None):
        state = initial if initial is not None else {"x": torch.zeros(1)}
        method = create_method("ca2fl", params)
        method.configure_server(state, [SimpleNamespace(id=cid) for cid in clients])
        server = FederatedServer(None, method, len(clients), state)
        return method, server

    def test_equation_41_independent_reference_and_prebuffer_mean(self):
        method, server = self.setup_method({"buffer_size": 2})
        # Independently compute Eq4.1 at buffer boundaries with zero unseen caches.
        cache = {cid: 0.0 for cid in ("a", "b", "c")}
        mean, expected, rows = 0.0, 0.0, []
        rounds = {cid: 0 for cid in cache}
        for cid, value in (("a", 2), ("b", 4), ("a", 6), ("c", 9), ("a", 7), ("b", 1)):
            rows.append(value - cache[cid])
            cache[cid] = value
            result = server.receive(upload(cid, value, rounds[cid]))
            rounds[cid] += 1
            if len(rows) == 2:
                expected += mean + sum(rows) / 2
                mean = sum(cache.values()) / 3
                rows = []
                self.assertTrue(result[0].applied)
                self.assertAlmostEqual(float(server.state["x"]), expected, places=5)
                self.assertAlmostEqual(float(method.global_cache_mean), mean, places=5)
        self.assertEqual(server.version, 3)
        self.assertEqual(server.accepted_updates, 6)

    def test_unseen_clients_stay_zero_and_population_does_not_shrink(self):
        method, server = self.setup_method({"buffer_size": 1}, clients=("a", "b", "c", "d"))
        server.receive(upload("a", 4))
        self.assertEqual(float(method.global_cache_mean), 1)
        result = server.receive(upload("a", 4, 1))[0]
        self.assertEqual(float(server.state["x"]), 5)
        self.assertEqual(result.metadata["cache_population"], 4)
        self.assertEqual(result.metadata["cached_clients"], 1)

    def test_repeated_client_before_flush_uses_distinct_denominator_and_latest_cache(self):
        method, server = self.setup_method({"buffer_size": 2}, clients=("a", "b"))
        server.receive(upload("a", 2))
        result = server.receive(upload("a", 5, 1))[0]
        self.assertEqual(float(server.state["x"]), 5)
        self.assertEqual(float(method.global_cache_mean), 2.5)
        self.assertEqual(result.metadata["buffer_distinct_clients"], 1)
        self.assertEqual(result.metadata["buffer_occupancy"], 2)

    def test_sample_counts_and_version_gap_do_not_add_unpublished_weights(self):
        _, server = self.setup_method({"buffer_size": 2}, clients=("a", "b"))
        server.receive(upload("a", 2, samples=1))
        server.receive(upload("b", 6, samples=1000))
        self.assertEqual(float(server.state["x"]), 4)
        # Old base0 is retained, without interpolation toward its local endpoint.
        server.receive(upload("a", 4, 1, base=10, version=0))
        server.receive(upload("b", 8, 1, base=-20, version=1))
        self.assertEqual(float(server.state["x"]), 10)

    def test_partial_buffer_counts_and_final_flush(self):
        method, server = self.setup_method({"buffer_size": 3, "server_lr": 0.1})
        server.receive(upload("a", 2))
        server.receive(upload("b", 4))
        self.assertEqual(server.version, 0)
        self.assertEqual(server.accepted_updates, 0)
        results = server.finish()
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(float(server.state["x"]), 0.3)
        self.assertEqual(server.accepted_updates, 2)
        self.assertTrue(results[0].metadata["residual_flush"])
        self.assertEqual(method.occupancy_sum / method.arrivals, 1.5)
        self.assertEqual(server.finish(), [])

    def test_disabling_flush_never_silently_drops_updates(self):
        _, server = self.setup_method({"flush_last_buffer": False})
        server.receive(upload())
        with self.assertRaisesRegex(RuntimeError, "residual"):
            server.finish()

    def test_evaluation_budget_counts_uploads_not_cached_reuse_or_server_versions(self):
        _, server = self.setup_method({"buffer_size": 3})
        schedule = PeriodicEvaluationSchedule(
            unit="incorporated_client_updates", interval=10, next_threshold=10
        )
        evaluations = []
        for index in range(21):
            server.receive(upload(("a", "b", "c")[index % 3], 1.0, index // 3))
            if schedule.observe(server.accepted_updates):
                evaluations.append((index + 1, server.accepted_updates, server.version))
        self.assertEqual(evaluations, [(12, 12, 4), (21, 21, 7)])

    def test_multiple_lora_tensors_keep_shapes_and_current_state_origin(self):
        initial = {"lora_A": torch.zeros(2, 3), "lora_B": torch.zeros(4, 2)}
        method, server = self.setup_method({"buffer_size": 1}, initial=initial)
        current = {key: torch.full_like(value, 10) for key, value in initial.items()}
        row = upload()
        row.base_state = {key: torch.full_like(value, -3) for key, value in initial.items()}
        row.local_state = {key: torch.full_like(value, -1) for key, value in initial.items()}
        row.delta = subtract(row.local_state, row.base_state)
        from afl_vlm.federation.types import ServerContext

        result = method.on_arrival(row, ServerContext(current, 5, server.expected_clients))[0]
        for key in initial:
            self.assertEqual(result.new_state[key].shape, initial[key].shape)
            torch.testing.assert_close(result.new_state[key], torch.full_like(initial[key], 12))

    def test_pending_state_roundtrip_including_quantized_cache(self):
        for bits in (32, 8, 4):
            with self.subTest(bits=bits):
                method, server = self.setup_method({"buffer_size": 2, "cache_bits": bits})
                server.receive(upload("a", 2))
                state = method.state_dict()
                restored = create_method("ca2fl")
                restored.load_state_dict(state)
                restored.configure_server(
                    server.state, [SimpleNamespace(id=c) for c in ("a", "b", "c")]
                )
                context = server.context()
                left = method.on_arrival(upload("b", 6), context)[0]
                right = restored.on_arrival(upload("b", 6), context)[0]
                torch.testing.assert_close(left.new_state["x"], right.new_state["x"])
                torch.testing.assert_close(method.global_cache_mean, restored.global_cache_mean)
                # Returned state owns storage and cannot mutate live method memory.
                state["cache_sum"].zero_()
                self.assertGreater(float(method.cache_sum), 0)

    def test_invalid_state_or_population_rejected(self):
        method, _ = self.setup_method()
        state = method.state_dict()
        for changed in ({"schema_version": 2}, {"server_lr": 0.3}):
            with self.assertRaises(ValueError):
                create_method("ca2fl").load_state_dict({**state, **changed})
        with self.assertRaisesRegex(ValueError, "population changed"):
            method.configure_server({"x": torch.zeros(1)}, [SimpleNamespace(id="z")])

    def test_quantizer_is_packed_bounded_seeded_and_training_rng_independent(self):
        vector = torch.linspace(-2, 3, 1001)
        for bits in (4, 8):
            method = create_method("ca2fl", {"cache_bits": bits})
            before = torch.random.get_rng_state().clone()
            encoded = method._encode(vector, "upload1")
            torch.testing.assert_close(before, torch.random.get_rng_state())
            decoded = method._decode(encoded, vector)
            self.assertEqual(encoded["codes"].dtype, torch.uint8)
            self.assertEqual(encoded["codes"].numel(), 501 if bits == 4 else 1001)
            self.assertLessEqual(float((decoded - vector).abs().max()), 5 / (2**bits - 1) + 1e-6)
            self.assertLess(method._storage_bytes(encoded), vector.numel() * 4)
            torch.testing.assert_close(
                decoded, method._decode(method._encode(vector, "upload1"), vector)
            )
            self.assertFalse(
                torch.equal(encoded["codes"], method._encode(vector, "upload2")["codes"])
            )

    def test_quantizer_unbiased_rounding_and_constant_zero_vectors(self):
        # Each middle coordinate has a .5 rounding probability; large independent
        # coordinates test the expectation, not a trained model or tiny workflow.
        for bits in (4, 8):
            method = create_method("ca2fl", {"cache_bits": bits})
            levels = 2**bits - 1
            vector = torch.cat((torch.tensor([0.0, 1.0]), torch.full((10000,), 0.5 / levels)))
            decoded = method._decode(method._encode(vector, "unbiased"), vector)
            self.assertAlmostEqual(float(decoded[2:].mean()) * levels, 0.5, delta=0.025)
            for constant in (0.0, -3.0):
                v = torch.full((7,), constant)
                torch.testing.assert_close(method._decode(method._encode(v, "constant"), v), v)

    def test_mf_keeps_raw_correction_but_quantizes_only_stored_cache(self):
        vector = torch.tensor([0.0, 0.13, 0.61, 1.0])
        for bits in (4, 8):
            method, server = self.setup_method(
                {"buffer_size": 1, "cache_bits": bits},
                clients=("a", "b"),
                initial={"x": torch.zeros(4)},
            )
            row = upload()
            row.base_state = {"x": torch.zeros(4)}
            row.delta = {"x": vector.clone()}
            row.local_state = clone_state(row.delta)
            server.receive(row)
            torch.testing.assert_close(server.state["x"], vector)
            cached = method._decode(method.client_cache["a"], vector)
            torch.testing.assert_close(method.global_cache_mean, cached / 2)
            row2 = copy.deepcopy(row)
            row2.update_id, row2.local_round = "a-r1", 1
            server.receive(row2)
            torch.testing.assert_close(server.state["x"], 2 * vector - cached / 2)

    def test_inputs_detached_and_not_modified(self):
        method, server = self.setup_method({"buffer_size": 1})
        row = upload()
        row.delta["x"].requires_grad_(True)
        source = row.delta["x"].clone()
        context_before = server.state["x"].clone()
        result = method.on_arrival(row, server.context())[0]
        self.assertFalse(result.new_state["x"].requires_grad)
        torch.testing.assert_close(row.delta["x"], source)
        torch.testing.assert_close(server.state["x"], context_before)
        self.assertIsNone(row.delta["x"].grad)

    def test_bad_arrivals_do_not_change_caches(self):
        method, server = self.setup_method()
        bad = [upload("unknown"), upload(version=1), upload(delta=float("nan"))]
        wrong_shape = upload()
        wrong_shape.delta = {"x": torch.zeros(2)}
        bad.append(wrong_shape)
        for row in bad:
            with self.assertRaises(ValueError):
                method.on_arrival(row, server.context())
            self.assertEqual(method.arrivals, 0)
            self.assertEqual(method.client_cache, {})
        server.receive(upload())
        for row in (upload(), upload(local_round=0, delta=5)):
            with self.assertRaises(ValueError):
                method.on_arrival(row, server.context())

    def test_parameter_validation(self):
        for params in (
            {"server_lr": 0},
            {"server_lr": float("nan")},
            {"buffer_size": 0},
            {"buffer_size": 1.5},
            {"buffer_size": True},
            {"cache_bits": 16},
            {"cache_bits": 4.0},
            {"cache_seed": -1},
            {"flush_last_buffer": "true"},
            {"staleness_weighting": True},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                create_method("ca2fl", params)

    def test_local_training_download_and_evaluation_remain_common(self):
        method = create_method("ca2fl")
        configured = TrainConfig(1, 4, 4, 2e-5, 2048, 42)
        self.assertIs(method.configure_local_training(configured), configured)
        context = ClientContext("a", "cls", "dataset", 8, 0, 0, 1.0, 42)
        state = {"x": torch.ones(1)}
        downloaded = method.prepare_download(state, context)
        downloaded["x"].zero_()
        self.assertEqual(float(state["x"]), 1)
        self.assertEqual(method.client_runtime_state(context), {})
        self.assertEqual(set(method.evaluation_states(state)), {"global"})
        self.assertEqual(method.secondary_evaluation_states(state), {})

    def test_three_inherited_configs_ddp_and_profile_registration(self):
        for setting in (2, 5, 10):
            root = ROOT / f"configs/experiments/llava/afvlm_cm/{setting}clients"
            config = load_config(root / "ca2fl.yaml")
            reference = load_config(root / "fedavg.yaml")
            validate_config(config, check_paths=False)
            for key in ("run", "training", "model", "dataset", "runtime", "evaluation"):
                self.assertEqual(config[key], reference[key])
            method = create_method("ca2fl", config["method"]["params"])
            self.assertEqual(resolve_runtime_backend(config, method), "client_ddp")
            self.assertEqual(method.buffer_size, 3)
            self.assertEqual(method.cache_bits, 32)
        self.assertIn("ca2fl", profile_method_names())


if __name__ == "__main__":
    unittest.main()
