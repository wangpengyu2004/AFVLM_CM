"""Exact tensor/protocol checks only: no VLM, images, model weights or training."""

from __future__ import annotations

import ast
import copy
import csv
import json
import math
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from afl_vlm.config import load_config, validate_config
from afl_vlm.execution import resolve_runtime_backend
from afl_vlm.federation.server import FederatedServer
from afl_vlm.federation.types import ClientContext, ClientSpec, Update
from afl_vlm.methods.ours_router.conflict import TaskConflictWindow
from afl_vlm.methods.ours_router.method import OursRouter
from afl_vlm.methods.ours_router.probe import initialization_weights
from afl_vlm.methods.ours_router.router import conflict_topk
from afl_vlm.methods.registry import create_method, method_names
from afl_vlm.models.base import clone_state

ROOT = Path(__file__).resolve().parents[1]


def module_map(rank=3):
    return {"m": {"rank": rank, "in_features": 2, "out_features": 2,
                  "a_name": "a", "b_name": "b", "scaling": 2.0}}


class ExactConflictChecks(unittest.TestCase):
    def zero(self):
        return {"a": torch.tensor([[1., 0], [0, 1], [1, 1]], dtype=torch.float64),
                "b": torch.zeros(2, 3, dtype=torch.float64)}

    def append(self, w, q, sign=1, indices=None):
        before = self.zero()
        after = clone_state(before)
        indices = [0] if indices is None else indices
        after["b"][:, indices] += sign
        return w.append(before, after, {"m": indices}, q, f"{q}-{w.clock+1}", w.clock+1)

    def test_exact_cached_gram_matches_dense_actual_BA_changes_and_cross_terms(self):
        g = torch.Generator().manual_seed(12)
        w = TaskConflictWindow(module_map(), ["a", "b", "c"])
        dense = []
        for i, task in enumerate(("a", "b", "a", "c")):
            before = {"a": torch.randn(3, 2, generator=g, dtype=torch.float64),
                      "b": torch.randn(2, 3, generator=g, dtype=torch.float64)}
            after = {key: value + torch.randn(value.shape, generator=g, dtype=value.dtype) * .1
                     for key, value in before.items()}
            selected = [0, 2] if i != 1 else [1, 2]
            changes = {}
            for j in selected:
                changes[j] = 2 * (torch.outer(after["b"][:, j], after["a"][j])
                                  - torch.outer(before["b"][:, j], before["a"][j]))
            dense.append(changes)
            w = w.append(before, after, {"m": selected}, task, str(i), i+1)
        for j in range(3):
            for u in range(4):
                for v in range(4):
                    expected = ((dense[u][j] * dense[v][j]).sum().item()
                                if j in dense[u] and j in dense[v] else 0.0)
                    self.assertAlmostEqual(w.grams["m"][j, u, v].item(), expected, places=12)
        own = sum((d.get(2, torch.zeros(2, 2, dtype=torch.float64))
                   for i, d in enumerate(dense) if w.rows[i]["task"] == "a"))
        peer_changes = [dense[i][2] for i in (1, 3)]
        expected = sum(max(0, -(own * peer).sum().item() /
                           (own.norm().item() * peer.norm().item() + w.eps))
                       for peer in peer_changes) / 2
        self.assertAlmostEqual(w.scores("a")["modules"]["m"]["penalty"][2], expected, places=12)

    def test_same_direction_zero_opposite_direction_symmetric_and_task_specific(self):
        w = TaskConflictWindow(module_map(), ["a", "b", "c", "unknown"])
        w = self.append(w, "a")
        w = self.append(w, "b")
        w = self.append(w, "c", -1)
        for task, expected in (("a", .5), ("b", .5), ("c", 1)):
            scores = w.scores(task)["modules"]["m"]
            self.assertAlmostEqual(scores["penalty"][0], expected, places=10)
            self.assertEqual(scores["peer_count"][0], 2)
        self.assertFalse(any(w.scores("unknown")["modules"]["m"]["valid"]))

    def test_each_task_keeps_three_uploads_others_do_not_evict_its_records(self):
        w = TaskConflictWindow(module_map(), ["a", "b"], size=3, max_age=24)
        w = self.append(w, "b")
        for _ in range(7):
            w = self.append(w, "a")
        self.assertEqual([r["accepted_at"] for r in w.rows if r["task"] == "a"], [6, 7, 8])
        self.assertEqual([r["accepted_at"] for r in w.rows if r["task"] == "b"], [1])
        # The limit counts uploads, even an upload not selecting an old component.
        w = self.append(w, "b", indices=[1])
        w = self.append(w, "b", indices=[2])
        w = self.append(w, "b", indices=[2])
        self.assertFalse(w.scores("b")["modules"]["m"]["own_direction_valid"][0])

    def test_age24_inclusive_age25_expires_and_new_record_does_not_refresh_old(self):
        w = TaskConflictWindow(module_map(), ["a", "b"], size=3, max_age=24)
        w = self.append(w, "b")
        for _ in range(24):
            w = self.append(w, "a")
        self.assertEqual(w.summary()["tasks"]["b"][0]["age"], 24)
        self.assertTrue(w.scores("a")["modules"]["m"]["valid"][0])
        w = self.append(w, "b", indices=[1])
        self.assertEqual([r["accepted_at"] for r in w.rows if r["task"] == "b"], [26])
        self.assertFalse(w.scores("a")["modules"]["m"]["valid"][0])
        self.assertEqual(w.summary()["tasks"]["b"][0]["age"], 0)

    def test_accumulated_direction_cancels_instead_of_last_update_only(self):
        w = TaskConflictWindow(module_map(), ["a", "b", "c"])
        w = self.append(w, "a", 1)
        w = self.append(w, "a", -1)
        w = self.append(w, "b", 1)
        w = self.append(w, "c", -1)
        self.assertFalse(w.scores("a")["modules"]["m"]["valid"][0])
        # Cancelled a is excluded, not counted as a zero-conflict vote.
        b = w.scores("b")["modules"]["m"]
        self.assertEqual(b["peer_count"][0], 1)
        self.assertAlmostEqual(b["penalty"][0], 1.0, places=10)

    def test_checkpoint_roundtrip_retains_exact_factors_cache_clocks_and_expiry(self):
        w = TaskConflictWindow(module_map(), ["a", "b"])
        w = self.append(self.append(w, "a"), "b", -1)
        snapshot = w.state_dict()
        restored = TaskConflictWindow(module_map(), ["a", "b"])
        restored.load_state_dict(snapshot)
        self.assertEqual(w.scores("a"), restored.scores("a"))
        for _ in range(25):
            restored = self.append(restored, "a")
        self.assertEqual(w.clock, 2)
        self.assertEqual(snapshot["clock"], 2)
        self.assertEqual(restored.summary()["tasks"]["b"], [])
        bad = copy.deepcopy(snapshot)
        bad["rows"][0]["accepted_at"] = -1
        with self.assertRaises(ValueError):
            TaskConflictWindow(module_map(), ["a", "b"]).load_state_dict(bad)


class ConflictSelectionChecks(unittest.TestCase):
    def test_controller_uses_warmed_router_and_freezes_virtual_download_scores(self):
        from afl_vlm.methods.ours_router.local import LocalRouter
        from test_ours_router import FactorRegistry

        fixture = FactorRegistry()
        controller = LocalRouter(fixture, {"rank_aggregation_mode": "conflict_router"})
        ctx = ClientContext("a/client_0", "a", "a", 128, 0, 0, 1., 42)
        controller.capacities[ctx.client_id] = {
            m: {"rank": 1} for m in controller.full_metadata
        }
        router = controller.router(ctx.client_id, 42)
        router.steps = 1  # Selection fixture only; no model/optimizer training.
        unknown = {"penalty": [1., 0., 0.], "valid": [True, False, False],
                   "peer_count": [1, 0, 0], "own_direction_valid": [True, False, False]}
        snapshot = {"clock": 4, "window": {}, "modules":
                    {m: copy.deepcopy(unknown) for m in controller.full_metadata}}
        with patch.object(router, "predict", return_value=torch.tensor([1., .9, .1])):
            choice = controller.select(fixture.snapshot_trainable(), ctx, snapshot)
        self.assertTrue(all(v == [1] for v in choice["indices"].values()))
        self.assertEqual(choice["conflict_selection"]["clock"], 4)
        snapshot["clock"] = 999
        self.assertEqual(choice["conflict_selection"]["clock"], 4)
        self.assertEqual(fixture.model.layers[0].self_attn.q_proj.lora_A["default"].out_features, 3)

    def test_probe_observer_reads_all_full_rank_pre_step_gradients_without_tracker(self):
        from afl_vlm.methods.ours_router.local import LocalRouter
        from test_ours_router import FactorRegistry

        fixture = FactorRegistry()
        controller = LocalRouter(fixture, {})
        controller.probing = True
        before = {}
        for name, parameter in fixture.named_federated_parameters():
            parameter.grad = torch.full_like(parameter, .25)
            before[name] = parameter.grad.clone()
        controller.observe(0)
        self.assertEqual(set(controller.pending), set(controller.full_metadata))
        for _name, (_head, features, labels) in controller.pending.items():
            self.assertEqual(features.shape[0], 3)
            self.assertEqual(labels.shape[0], 3)
            self.assertFalse(features.requires_grad or labels.requires_grad)
        for name, parameter in fixture.named_federated_parameters():
            self.assertTrue(torch.equal(parameter.grad, before[name]))

    def test_penalty_can_change_topk_unknown_is_unpenalized_and_rng_preserved(self):
        rng = torch.get_rng_state().clone()
        observed = {"penalty": [1., 0., 0.], "valid": [True, False, False],
                    "peer_count": [1, 0, 0], "own_direction_valid": [True, False, False]}
        chosen, row = conflict_topk(torch.tensor([1., .9, .1]), observed, 1, .3, 1e-12, 42)
        self.assertEqual(chosen.tolist(), [1])
        self.assertEqual(row["changed_components"], 1)
        self.assertEqual(row["router_only_indices"], [0])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        observed["valid"] = [False, False, False]
        chosen, _ = conflict_topk(torch.tensor([1., .9, .1]), observed, 1, .3, 1e-12, 42)
        self.assertEqual(chosen.tolist(), [0])

    def test_negative_scores_and_zero_scores_are_finite_deterministic_without_replacement(self):
        unknown = {"penalty": [0.] * 5, "valid": [False] * 5, "peer_count": [0] * 5,
                   "own_direction_valid": [False] * 5}
        for scores in (torch.zeros(5), -torch.arange(1., 6)):
            x, row = conflict_topk(scores, unknown, 3, .3, 1e-12, 7)
            y, _ = conflict_topk(scores, unknown, 3, .3, 1e-12, 7)
            self.assertTrue(torch.equal(x, y))
            self.assertEqual(len(set(x.tolist())), 3)
            self.assertTrue(all(math.isfinite(v) for v in row["final_scores"]))


class ConflictServerChecks(unittest.TestCase):
    def setUp(self):
        from test_ours_router import FactorRegistry

        self.fixture = FactorRegistry()
        self.base = self.fixture.snapshot_trainable()
        self.clients = [ClientSpec(f"{q}/client_0", q, q, 128, 1, 0, 0, 1)
                        for q in ("a", "b")]

    def setup_method(self):
        method = OursRouter({"rank_aggregation_mode": "conflict_router"})
        method.configure_server(self.base, self.clients)
        method.configure_schedule([SimpleNamespace(client_id=c.id, task=c.task, local_round=n)
                                   for c in self.clients for n in range(4)])
        return method, FederatedServer(None, method, 2, self.base)

    def packet(self, method, q="a", number=0, delta=.1, selected=None, current=None):
        chosen = {m: [0, 2] if selected is None else selected for m in method.modules}
        base = clone_state(self.base if current is None else current)
        local = clone_state(base)
        metadata = copy.deepcopy(method.modules)
        for m, item in metadata.items():
            item["scaling"] = 2.
            local[item["b_name"]][:, chosen[m]] += delta
        ctx = ClientContext(f"{q}/client_0", q, q, 128, number, 0, number+1., 42)
        update = Update(f"{ctx.client_id}-r{number}", ctx.client_id, q, q, 128, number, 0,
                        ctx.arrival_time, base, local, {}, 42, "ids", [], 4,
                        metadata={"lora_metadata": metadata, "hc_selection": chosen,
                                  "module_sensitivity": {m: 1.5 if ".0." in m else .5
                                                         for m in metadata},
                                  "sensitivity_valid": True, "sensitivity_transform": {},
                                  "sensitivity_observations": {},
                                  "hc_local": {"trajectory": [], "selection_source":
                                               dict.fromkeys(metadata, "router")}})
        method.downloads[ctx.client_id, number] = (0, clone_state(base))
        method.choices[ctx.client_id, number] = {"indices": chosen}
        return method.pack_local_update(update, ctx)

    def test_uniform_alpha_uses_base_relative_delta_no_C_history_frequency_or_overlap_factor(self):
        method, server = self.setup_method()
        first = self.packet(method)
        stale = self.packet(method, "b")
        result = server.receive(first)[0]
        self.assertEqual(result.applied_weight, 1.)
        before = server.state
        result = server.receive(stale)[0]
        row = result.metadata["router_diagnostics"]
        self.assertAlmostEqual(row["uniform_alpha"], math.exp(-1.), places=6)
        self.assertEqual(len(set(row["module_alpha"].values())), 1)
        self.assertEqual(method.history, {})
        self.assertEqual(method.window, [])
        for item in method.modules.values():
            name = item["b_name"]
            self.assertTrue(torch.allclose(server.state[name][:, [0, 2]],
                                           before[name][:, [0, 2]] + .1 * math.exp(-1)))
            self.assertTrue(torch.equal(server.state[name][:, 1], self.base[name][:, 1]))
        self.assertEqual(method.conflict_window.clock, server.accepted_updates)

    def test_records_are_actual_server_before_after_not_raw_client_proposals(self):
        method, server = self.setup_method()
        first, stale = self.packet(method), self.packet(method, "b", delta=-.1)
        server.receive(first)
        before = server.state
        server.receive(stale)
        after = server.state
        record = method.conflict_window.rows[-1]
        for m, item in method.modules.items():
            factors = record["modules"][m]
            for offset, j in enumerate(factors["indices"]):
                actual = 2 * (torch.outer(after[item["b_name"]][:, j], after[item["a_name"]][j])
                              - torch.outer(before[item["b_name"]][:, j],
                                            before[item["a_name"]][j]))
                represented = factors["left"][offset].T @ factors["right"][offset]
                self.assertTrue(torch.allclose(actual, represented, atol=1e-7))

    def test_rejection_and_model_copy_failure_do_not_add_records_or_advance_clock(self):
        method, server = self.setup_method()
        update = self.packet(method)
        mutation = method.on_arrival(update, server.context())[0]
        self.assertEqual(method.conflict_window.clock, 0)
        model = SimpleNamespace(load_trainable=lambda state: None)
        server.model = model
        with patch.object(model, "load_trainable", side_effect=[RuntimeError("copy failed"), None]):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                server._apply(mutation, 0, 0)
        self.assertEqual((server.version, method.accepted_count), (0, 0))
        self.assertFalse(method.seen or method.conflict_window.rows)
        server.model = None
        server.receive(update)
        snapshot = method.conflict_window.summary()
        rejected = self.packet(method, "b", delta=0)
        result = server.receive(rejected)[0]
        self.assertEqual(result.applied_weight, 0)
        self.assertEqual(method.conflict_window.summary(), snapshot)
        with self.assertRaises(ValueError):
            server.receive(update)

    def test_method_state_roundtrip_and_cross_mode_migration_rejected(self):
        method, server = self.setup_method()
        server.receive(self.packet(method))
        saved = method.state_dict()
        restored = OursRouter(method.params)
        restored.load_state_dict(saved)
        restored.configure_server(server.state, self.clients)
        self.assertEqual(restored.conflict_window.scores("a"), method.conflict_window.scores("a"))
        self.assertEqual(restored.accepted_count, 1)
        for params in ({}, {"rank_aggregation_mode": "overlap_switch"}):
            with self.assertRaises(ValueError):
                OursRouter(params).load_state_dict(saved)
        bad = copy.deepcopy(saved)
        bad["router_protocol"].pop("conflict_window")
        with self.assertRaises(ValueError):
            OursRouter(method.params).load_state_dict(bad)

    def test_new_diagnostics_export_without_fake_history_columns(self):
        method, server = self.setup_method()
        row = server.receive(self.packet(method))[0].metadata
        export = runpy.run_path(str(ROOT / "tools/export_ours_router_diagnostics.py"))["export"]
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "events.jsonl").write_text(json.dumps({"result_metadata": [row]}) + "\n",
                                             encoding="utf-8")
            counts = export(run, run / "export")
            self.assertEqual(counts["updates"], 1)
            self.assertEqual(counts["conflict_windows"], 2)
            with (run / "export/modules.csv").open(encoding="utf-8-sig", newline="") as stream:
                entries = list(csv.DictReader(stream))
            self.assertTrue(all(e["history_before"] == "" and e["alpha"] == "1.0"
                                for e in entries))


class PreparationAndIsolationChecks(unittest.TestCase):
    def test_initialization_is_task_equal_and_sample_weighted_within_task(self):
        clients = {str(i): SimpleNamespace(id=str(i), task=q, samples=range(n))
                   for i, (q, n) in enumerate((("a", 1), ("a", 3), ("b", 20)))}
        weights = initialization_weights(clients)
        self.assertEqual(weights, {"0": .125, "1": .375, "2": .5})

    def test_initialization_updates_version_zero_only_and_other_methods_noop(self):
        from afl_vlm.methods.ours_router.local import LocalRouter
        from test_ours_router import FactorRegistry

        fixture = FactorRegistry()
        controller = LocalRouter(fixture, {})
        method = OursRouter({"rank_aggregation_mode": "conflict_router"})
        method.controller = controller
        initial = fixture.snapshot_trainable()
        merged = {k: v + .125 for k, v in initial.items()}
        controller.initialized_state = merged
        server = FederatedServer(fixture, method, 1, initial)
        server.initialize_from_calibration()
        self.assertEqual(
            (server.version, server.received_updates, server.accepted_updates), (0, 0, 0)
        )
        self.assertTrue(all(torch.equal(v, merged[k]) for k, v in server.state.items()))
        self.assertIsNone(controller.initialized_state)
        for name in method_names():
            if name == "ours_router":
                continue
            method = create_method(name)
            self.assertIsNone(method.calibration_initial_state())
            server = FederatedServer(None, method, 1, initial)
            server.initialize_from_calibration()
            self.assertTrue(all(torch.equal(v, initial[k]) for k, v in server.state.items()))

    def test_three_new_configs_keep_training_data_evaluation_and_ddp_interfaces(self):
        for setting in (2, 5, 10):
            root = ROOT / f"configs/experiments/llava/afvlm_cm/{setting}clients"
            config, old = load_config(root / "ours_router.yaml"), load_config(root / "ours.yaml")
            validate_config(config, check_paths=False)
            method = create_method("ours_router", config["method"]["params"])
            self.assertTrue(method.conflict_mode)
            self.assertEqual(config["method"]["params"]["conflict_per_task_window"], 3)
            self.assertEqual(config["method"]["params"]["conflict_max_age"], 24)
            for section in ("training", "dataset", "evaluation", "federation", "runtime"):
                self.assertEqual(config[section], old[section])
            self.assertEqual(resolve_runtime_backend(config, method), "client_ddp")

    def test_probe_uses_existing_loss_gradient_step_hooks_and_leaves_no_extra_vlm_pass(self):
        tree = ast.parse((ROOT / "code/afl_vlm/methods/ours_router/probe.py").read_text("utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertEqual(sum(isinstance(c.func, ast.Attribute) and c.func.attr == "local_train"
                             for c in calls), 1)
        self.assertFalse(any(isinstance(c.func, ast.Attribute) and c.func.attr in
                             {"backward", "forward", "generate"} for c in calls))
        hooks = [c for c in calls if isinstance(c.func, ast.Name) and c.func.id == "replace"][0]
        self.assertTrue({"loss_hook", "gradient_hook", "step_hook"}.issubset(
            {key.arg for key in hooks.keywords}
        ))


if __name__ == "__main__":
    unittest.main()
