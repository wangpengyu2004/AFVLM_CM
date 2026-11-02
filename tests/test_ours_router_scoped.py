"""Non-training set, transaction, tensor-equation and compatibility checks."""

from __future__ import annotations

import ast
import copy
import csv
import json
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from afl_vlm.federation.server import FederatedServer
from afl_vlm.federation.types import ClientContext, ClientSpec, Update
from afl_vlm.methods.ours_router.method import OursRouter
from afl_vlm.methods.ours_router.recent import RecentSelectionRegistry
from afl_vlm.methods.ours_router.scoped import fuse, precision, rank_alpha
from afl_vlm.methods.registry import create_method, method_names
from afl_vlm.models.base import clone_state

ROOT = Path(__file__).resolve().parents[1]


def event(client, number):
    return SimpleNamespace(client_id=client, local_round=number, task=client.split("/")[0])


class RecentWindowChecks(unittest.TestCase):
    def registry(self, ranks=None):
        r = RecentSelectionRegistry(ranks or {"m": 10}, {"a/0": "a", "a/1": "a", "b/0": "b"})
        r.configure([event(c, n) for c in r.clients for n in range(6)])
        return r

    def accept(self, r, client, number, indices):
        r.resolve(
            client,
            number,
            "accepted",
            accepted_count=number + 1,
            virtual_time=number + 0.5,
            job_id=f"{client}-r{number}",
            selected={m: list(indices) for m in r.ranks},
        )

    def test_recent_union_is_not_majority_and_old_positions_expire(self):
        r = self.registry()
        for n, indices in enumerate(([2, 3, 4], [2, 3, 5], [2, 3, 6])):
            self.accept(r, "a/0", n, indices)
        self.assertEqual(r.client_union("a/0")["m"], [2, 3, 4, 5, 6])
        self.accept(r, "a/0", 3, [7])
        self.assertEqual(r.client_union("a/0")["m"], [2, 3, 5, 6, 7])

    def test_windows_follow_each_clients_round_not_global_arrivals(self):
        r = self.registry()
        self.accept(r, "a/0", 4, [4])
        for n in range(6):
            self.accept(r, "b/0", n, [n])
        self.assertEqual(r.client_union("a/0")["m"], [4])
        for n in (2, 5, 1):
            self.accept(r, "a/0", n, [n])
        self.assertEqual([x["local_round"] for x in r.recent["a/0"]], [2, 4, 5])

    def test_module_windows_include_empty_records_and_use_variable_rank_map(self):
        r = self.registry({"small": 5, "medium": 10, "large": 32})
        for n in range(4):
            selected = {"small": [4] if n == 0 else [], "medium": [9], "large": [31 - n]}
            r.resolve(
                "a/0",
                n,
                "accepted",
                accepted_count=n + 1,
                virtual_time=n,
                job_id=str(n),
                selected=selected,
            )
        self.assertEqual(r.client_union("a/0"), {"small": [], "medium": [9], "large": [28, 29, 30]})

    def test_same_task_dedup_and_pairwise_cross_task_competition(self):
        r = self.registry()
        self.accept(r, "a/0", 0, [3])
        self.accept(r, "a/1", 0, [3])
        p = precision(
            r,
            "a",
            {"m": [2, 3, 4]},
            {"a": {"m": 1}, "b": {"m": 1}},
            {"a": 0.5, "b": 0.5},
            {},
            {},
            1,
        )
        self.assertEqual(p["m"]["competition_indices"], [])
        self.accept(r, "b/0", 0, [3, 5])
        p = precision(
            r,
            "a",
            {"m": [2, 3, 4]},
            {"a": {"m": 1}, "b": {"m": 1}},
            {"a": 0.5, "b": 0.5},
            {},
            {},
            1,
        )
        self.assertEqual(p["m"]["competition_indices"], [3])
        self.assertEqual(p["m"]["omega"], [0, 1, 0])

    def test_three_tasks_need_pairwise_union_not_total_intersection(self):
        r = RecentSelectionRegistry({"m": 5}, {"a/0": "a", "b/0": "b", "c/0": "c"})
        r.configure([event(c, n) for c in r.clients for n in range(2)])
        for c, indices in zip(r.clients, ([1, 2], [2, 3], [3, 4]), strict=True):
            self.accept(r, c, 0, indices)
        active, _ = r.task_unions()
        overlapping = [j for j in range(5) if sum(j in active[q]["m"] for q in active) >= 2]
        self.assertEqual(overlapping, [2, 3])

    def test_overlap_only_switches_original_module_history_without_reweighting_tasks(self):
        r = self.registry()
        self.accept(r, "a/0", 0, [1])
        self.accept(r, "b/0", 0, [2])
        p = precision(
            r,
            "c",
            {"m": [1, 2, 4]},
            {"a": {"m": 100}, "b": {"m": 2}},
            {"a": 0.5, "b": 0.5},
            {},
            {},
            1,
        )
        self.assertEqual(p["m"]["original_module_omega_base"], 51)
        self.assertEqual(p["m"]["omega"], [51, 51, 0])
        self.assertEqual(p["m"]["sensitivity_mask"], [True, True, False])

    def test_one_client_finishing_does_not_end_task_or_discard_its_task_selection_evidence(self):
        r = self.registry()
        self.accept(r, "a/1", 0, [4, 8])
        for n in range(6):
            self.accept(r, "a/0", n, [2, 3, 4 + n % 3])
        active, exited = r.task_unions()
        self.assertEqual(active["a"]["m"], {2, 3, 4, 5, 6, 8})
        self.assertEqual(exited["a"]["m"], set())
        saved = copy.deepcopy(r.exits["a/0"])
        self.accept(r, "a/1", 1, [9])
        self.assertEqual(r.exits["a/0"], saved)
        self.assertFalse(r.task_finished("a"))

    def test_same_task_evidence_never_enables_sensitivity_switch_for_itself(self):
        r = self.registry()
        for n in range(6):
            self.accept(r, "a/0", n, [2])
        self.accept(r, "a/1", 0, [2, 3])
        p = precision(
            r, "b", {"m": [2, 3]}, {"a": {"m": 4}, "b": {"m": 1}}, {"a": 0.5, "b": 0.5}, {}, {}, 1
        )
        self.assertEqual(p["m"]["omega_base"], [2.5, 2.5])
        p = precision(
            r, "a", {"m": [2, 3]}, {"a": {"m": 4}, "b": {"m": 1}}, {"a": 0.5, "b": 0.5}, {}, {}, 1
        )
        self.assertEqual(p["m"]["omega_base"], [0, 0])
        self.assertEqual(p["m"]["competition_indices"], [])

    def test_terminal_failed_cancelled_or_rejected_jobs_exit_without_fabricated_selections(self):
        for outcome in ("failed", "cancelled", "terminated", "rejected"):
            r = RecentSelectionRegistry({"m": 10}, {"a/0": "a"})
            r.configure([event("a/0", n) for n in range(2)])
            self.accept(r, "a/0", 0, [7])
            r.resolve("a/0", 1, outcome, accepted_count=1, virtual_time=9)
            self.assertEqual(r.exits["a/0"]["selected_indices"]["m"], [7])
            self.assertTrue(r.task_finished("a"))
        r = RecentSelectionRegistry({"m": 10}, {"a/0": "a"})
        r.configure([event("a/0", 0)])
        r.resolve("a/0", 0, "cancelled", accepted_count=0, virtual_time=1)
        self.assertEqual(r.exits["a/0"]["selected_indices"]["m"], [])

    def test_duplicate_terminal_is_idempotent_unknown_and_conflicting_outcomes_fail(self):
        r = self.registry()
        self.accept(r, "a/0", 0, [1])
        before = r.state_dict()
        self.accept(r, "a/0", 0, [1])
        self.assertEqual(r.state_dict(), before)
        for c, n, outcome in (
            ("a/0", 0, "failed"),
            ("unknown", 0, "cancelled"),
            ("a/0", 99, "failed"),
        ):
            with self.assertRaises(ValueError):
                r.resolve(c, n, outcome, accepted_count=0, virtual_time=0)
        self.assertEqual(r.state_dict(), before)

    def test_registry_roundtrip_validates_frozen_masks_and_global_rank_bounds(self):
        r = self.registry()
        for n in range(6):
            self.accept(r, "a/0", n, [n])
        saved = r.state_dict()
        restored = self.registry()
        restored.load_state_dict(saved)
        self.assertEqual(restored.state_dict(), saved)
        for mutate in (
            lambda x: x["exits"]["a/0"]["selected_indices"]["m"].append(9),
            lambda x: x["recent_jobs"]["a/0"][-1]["selected_indices"]["m"].append(10),
            lambda x: x.update(schema_version=0),
        ):
            broken = copy.deepcopy(saved)
            mutate(broken)
            with self.assertRaises(ValueError):
                restored.load_state_dict(broken)
            self.assertEqual(restored.state_dict(), saved)


class AlphaAndScatterChecks(unittest.TestCase):
    def test_reliability_never_cancels_on_unprotected_positions(self):
        self.assertEqual(
            rank_alpha(0, 0.2, [1], [0], sensitivity_mask=[False], eta=1, lam=1, eps=1e-12), [0.2]
        )
        values = rank_alpha(
            1,
            0.5,
            [0.5, 1, 1],
            [1, 0, 1],
            sensitivity_mask=[True, False, True],
            eta=1,
            lam=1,
            eps=1e-12,
        )
        for actual, expected in zip(values, (1 / 5, 0.5, 1 / 3), strict=True):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            rank_alpha(0, 0.5, [1], [1], sensitivity_mask=[True], eta=1, lam=1, eps=1e-12), [0]
        )
        self.assertEqual(
            rank_alpha(
                1, 1, [1], [0], sensitivity_mask=[False], eta=1, lam=1, eps=1e-12, maximum=0.3
            ),
            [0.3],
        )

    def test_noncontinuous_unsorted_indices_share_correct_a_b_coefficients(self):
        current = {"a": [[0.0, 0.0] for _ in range(5)], "b": [[0.0] * 5 for _ in range(3)]}
        update = SimpleNamespace(
            metadata={
                "lora_metadata": {"m": {"a_name": "a", "b_name": "b", "out_features": 3}},
                "hc_selection": {"m": [4, 1]},
            },
            delta={"a": [[2.0, 4.0], [6.0, 8.0]], "b": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]},
        )
        result = fuse(current, update, {"m": [0.1, 0.5]})
        self.assertEqual(result["a"][4], [0.2, 0.4])
        self.assertEqual(result["a"][1], [3, 4])
        self.assertEqual(result["b"][0], [0, 1, 0, 0, 0.1])
        self.assertEqual(result["a"][0], current["a"][0])
        self.assertEqual(current["a"][4], [0, 0])

    def test_exit_masks_restrict_age_contributions_by_task(self):
        r = RecentSelectionRegistry({"m": 5}, {"a/0": "a", "b/0": "b"})
        r.configure([event(c, 0) for c in r.clients])
        for c, indices in (("a/0", [1]), ("b/0", [2])):
            r.resolve(
                c,
                0,
                "accepted",
                accepted_count=1,
                virtual_time=1,
                job_id=c,
                selected={"m": indices},
            )
        p = precision(
            r,
            "c",
            {"m": [1, 2, 3]},
            {"a": {"m": 99}, "b": {"m": 99}},
            {"a": 0.5, "b": 0.5},
            {"a": {"history": {"m": 4}}, "b": {"history": {"m": 2}}},
            {"a": {"m": 3}, "b": {"m": 1}},
            1,
        )
        self.assertEqual(p["m"]["omega_base"], [3, 3, 0])
        self.assertEqual(p["m"]["omega_age"], [3, 1, 0])
        self.assertEqual(p["m"]["competition_indices"], [])


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Tensor equations need torch, never a VLM forward")
class ScopedTransactionChecks(unittest.TestCase):
    def setUp(self):
        from test_ours_router import FactorRegistry

        self.fixture = FactorRegistry()
        self.base = self.fixture.snapshot_trainable()
        self.clients = [
            ClientSpec(f"{q}/client_{i}", q, q, 128, 1, 0, 0, 1)
            for q in ("vqa", "cls")
            for i in (0, 1)
        ]

    def setup_method(self, plan=None, **params):
        method = OursRouter({"rank_aggregation_mode": "overlap_switch", **params})
        method.configure_server(self.base, self.clients)
        method.configure_schedule(plan or [event(c.id, n) for c in self.clients for n in range(4)])
        return method, FederatedServer(None, method, len(self.clients), self.base)

    def packet(self, method, client="vqa/client_0", number=0, indices=None, c=None, changed=True):
        q = client.split("/")[0]
        ctx = ClientContext(client, q, q, 128, number, 0, number + 1.0, 42)
        local = clone_state(self.base)
        metadata = copy.deepcopy(method.modules)
        for item in metadata.values():
            item["scaling"] = 2.0
        selection = {m: list(indices if indices is not None else [0, 2]) for m in metadata}
        sensitivity = {m: (1.5 if ".0." in m else 0.5) if c is None else c for m in metadata}
        if changed:
            for m, item in metadata.items():
                local[item["b_name"]][:, selection[m]] += 0.1
        update = Update(
            f"{client}-r{number}",
            client,
            q,
            q,
            128,
            number,
            0,
            number + 1.0,
            clone_state(self.base),
            local,
            {},
            42,
            "ids",
            [],
            4,
            metadata={
                "lora_metadata": metadata,
                "hc_selection": selection,
                "module_sensitivity": sensitivity,
                "sensitivity_valid": True,
                "sensitivity_transform": {},
                "sensitivity_observations": {},
                "hc_local": {
                    "trajectory": [],
                    "selection_source": dict.fromkeys(metadata, "router"),
                },
            },
        )
        method.downloads[(client, number)] = (0, clone_state(self.base))
        return method.pack_local_update(update, ctx)

    def test_first_arrival_has_no_fabricated_history_scope(self):
        method, server = self.setup_method()
        result = server.receive(self.packet(method))[0]
        d = result.metadata["router_diagnostics"]
        self.assertEqual(server.version, 1)
        self.assertTrue(all(value == 1 for value in d["module_alpha"].values()))
        self.assertTrue(all(value == 0 for value in d["omega"].values()))
        self.assertEqual(method.accepted_count, 1)

    def test_mutation_does_not_commit_history_before_model_and_server_state(self):
        method, server = self.setup_method()
        packet = self.packet(method)
        before = method.selection_registry.state_dict()
        mutation = method.on_arrival(packet, server.context())[0]
        self.assertEqual(method.selection_registry.state_dict(), before)
        self.assertEqual(method.history_counts["vqa"], 0)
        self.assertEqual(method.accepted_count, 0)
        server._apply(mutation, 0, 0)
        self.assertEqual(method.history_counts["vqa"], 1)
        self.assertEqual(method.accepted_count, 1)

    def test_failed_model_state_copy_does_not_publish_registry_or_version(self):
        method, server = self.setup_method()
        mutation = method.on_arrival(self.packet(method), server.context())[0]
        model = SimpleNamespace(load_trainable=lambda state: None)
        server.model = model
        with patch.object(model, "load_trainable", side_effect=[RuntimeError("copy failed"), None]):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                server._apply(mutation, 0, 0)
        self.assertEqual(
            (server.version, server.accepted_updates, method.accepted_count), (0, 0, 0)
        )
        self.assertFalse(method.selection_registry.resolved)

    def test_second_tasks_first_upload_immediately_detects_overlap(self):
        method, server = self.setup_method()
        server.receive(self.packet(method))
        packet = self.packet(method, "cls/client_0", indices=[1, 2])
        d = server.receive(packet)[0].metadata["router_diagnostics"]
        self.assertTrue(
            all(
                row["competition_indices"] == [2]
                for row in d["scoped_aggregation"]["modules"].values()
            )
        )
        self.assertEqual(d["history_before"]["cls"], dict.fromkeys(method.modules, 1.0))

    def test_overlap_coefficient_is_exact_original_rule_unmatched_is_only_reliability(self):
        method, server = self.setup_method()
        server.receive(self.packet(method, "cls/client_0", indices=[0, 2]))
        packet = self.packet(method, "vqa/client_0", indices=[1, 2])
        old = OursRouter({"history_age_enabled": False})
        old.configure_server(self.base, self.clients)
        old.history, old.history_counts = copy.deepcopy(method.history), dict(method.history_counts)
        old.window = list(method.window)
        old.downloads = dict(method.downloads)
        previous = old.on_arrival(copy.deepcopy(packet), server.context())[0].metadata[
            "router_diagnostics"
        ]
        result = server.receive(packet)[0].metadata["router_diagnostics"]
        self.assertEqual(result["reliability"], previous["reliability"])
        self.assertEqual(result["frequency_weight"], previous["frequency_weight"])
        for name, row in result["scoped_aggregation"]["modules"].items():
            self.assertEqual(row["sensitivity_mask"], [False, True])
            self.assertEqual(row["alpha"][0], result["reliability"])
            self.assertEqual(row["alpha"][1], previous["module_alpha"][name])

    def test_task_freezes_complete_union_and_history_only_after_all_clients_finish(self):
        plan = [
            event(c.id, n)
            for c in self.clients
            for n in (range(1) if c.task == "vqa" else range(3))
        ]
        method, server = self.setup_method(plan)
        server.receive(self.packet(method, "vqa/client_0", indices=[0]))
        self.assertNotIn("vqa", method.finished)
        self.assertNotIn("vqa", method.selection_registry.finished_tasks)
        active, frozen = method.selection_registry.task_unions()
        self.assertTrue(all(s == {0} for s in active["vqa"].values()))
        self.assertTrue(all(not s for s in frozen["vqa"].values()))
        server.receive(self.packet(method, "vqa/client_1", indices=[1, 2]))
        saved = copy.deepcopy(method.finished["vqa"])
        saved_set = copy.deepcopy(method.selection_registry.finished_tasks["vqa"])
        self.assertTrue(all(indices == [0, 1, 2] for indices in saved["selected_indices"].values()))
        active, frozen = method.selection_registry.task_unions()
        self.assertTrue(all(not s for s in active["vqa"].values()))
        self.assertTrue(all(s == {0, 1, 2} for s in frozen["vqa"].values()))
        for n in range(2):
            server.receive(self.packet(method, "cls/client_0", n))
        self.assertEqual(method.finished["vqa"], saved)
        self.assertEqual(method.selection_registry.finished_tasks["vqa"], saved_set)
        restored = OursRouter({"rank_aggregation_mode": "overlap_switch"})
        restored.load_state_dict(method.state_dict())
        self.assertEqual(restored.finished["vqa"], saved)
        self.assertEqual(restored.selection_registry.finished_tasks["vqa"], saved_set)
        broken = method.state_dict()
        first_module = next(iter(method.modules))
        broken["router_protocol"]["finished"]["vqa"]["selected_indices"][first_module] = []
        with self.assertRaisesRegex(ValueError, "Frozen task history"):
            restored.load_state_dict(broken)

    def test_last_rejection_exits_client_without_updating_history_frequency_or_count(self):
        plan = [event(c.id, 0) for c in self.clients]
        method, server = self.setup_method(plan)
        packet = self.packet(method, changed=False)
        d = server.receive(packet)[0].metadata["router_diagnostics"]
        self.assertTrue(d["job_resolution"]["client_finished"])
        self.assertEqual(server.version, 0)
        self.assertEqual(method.history_counts["vqa"], 0)
        self.assertEqual(method.window, [])
        self.assertNotIn("vqa", method.finished)
        self.assertTrue(
            all(
                not indices
                for indices in method.selection_registry.exits["vqa/client_0"][
                    "selected_indices"
                ].values()
            )
        )
        server.resolve_job("vqa/client_1", 0, "cancelled", 2.0)
        self.assertIn("vqa", method.finished)
        self.assertEqual(method.finished["vqa"]["observed_updates"], 0)
        self.assertEqual(method.accepted_count, 0)

    def test_last_success_is_in_exit_snapshot_age_uses_prior_success_count(self):
        plan = [
            event(c.id, n)
            for c in self.clients
            for n in (range(1) if c.task == "vqa" else range(3))
        ]
        method, server = self.setup_method(plan)
        server.receive(self.packet(method))
        d = server.receive(self.packet(method, "vqa/client_1", indices=[1, 2]))[0].metadata[
            "router_diagnostics"
        ]
        self.assertIn("vqa", d["job_resolution"]["finished_tasks"])
        self.assertEqual(method.finished["vqa"]["finished_at"], 2)
        self.assertEqual(method.finished["vqa"]["history"], method.history["vqa"])
        d = server.receive(self.packet(method, "cls/client_0"))[0].metadata["router_diagnostics"]
        self.assertEqual(d["history_age_boosts_before"]["vqa"], 0)
        d = server.receive(self.packet(method, "cls/client_0", 1))[0].metadata["router_diagnostics"]
        for m, row in d["scoped_aggregation"]["modules"].items():
            self.assertTrue(all(v > 0 if ".0." in m else v == 0 for v in row["omega_age"]))
            self.assertEqual(row["competition_indices"], [])
            self.assertEqual(row["effective_frequency"], [d["frequency_weight"]] * 2)

    def test_restore_next_update_reproduces_masks_precision_alpha_and_state(self):
        method, server = self.setup_method()
        server.receive(self.packet(method))
        incoming = self.packet(method, "cls/client_0", indices=[1, 2])
        saved = method.state_dict()
        restored = OursRouter({"rank_aggregation_mode": "overlap_switch"})
        restored.load_state_dict(saved)
        other = FederatedServer(None, restored, len(self.clients), server.state)
        other.version = server.version
        # StateStore's supplied initial state is associated with version 0.
        other.state_store._states = {server.version: clone_state(server.state)}
        first = server.receive(incoming)[0].metadata["router_diagnostics"]
        second = other.receive(copy.deepcopy(incoming))[0].metadata["router_diagnostics"]
        for key in ("scoped_aggregation", "history_after", "job_resolution", "module_alpha"):
            self.assertEqual(first[key], second[key])
        for name in server.state:
            self.assertTrue(torch.equal(server.state[name], other.state[name]))

    def test_legacy_checkpoint_cannot_be_silently_migrated_to_scoped(self):
        old = OursRouter()
        old.configure_server(self.base, self.clients)
        with self.assertRaisesRegex(ValueError, "selection history"):
            OursRouter({"rank_aggregation_mode": "overlap_switch"}).load_state_dict(
                old.state_dict()
            )

    def test_invalid_indices_unknown_job_duplicate_do_not_change_algorithm_state(self):
        method, server = self.setup_method()
        packet = self.packet(method)
        before = method.selection_registry.state_dict()
        for indices in ([0, 0], [3], [True]):
            broken = copy.deepcopy(packet)
            broken.metadata["hc_selection"][next(iter(method.modules))] = indices
            with self.assertRaises(ValueError):
                server.receive(broken)
        self.assertEqual(method.selection_registry.state_dict(), before)
        with self.assertRaises(ValueError):
            server.resolve_job("unknown", 0, "failed")
        server.receive(packet)
        once = method.selection_registry.state_dict()
        with self.assertRaises(ValueError):
            server.receive(packet)
        self.assertEqual(method.selection_registry.state_dict(), once)

    def test_unsorted_tensor_payload_keeps_original_a_row_b_column_mapping(self):
        from afl_vlm.methods.ours_router.scoped import validate

        method, server = self.setup_method()
        packet = self.packet(method, indices=[2, 0])
        validate(packet, method.modules)
        d = server.receive(packet)[0].metadata["router_diagnostics"]
        for m in method.modules:
            self.assertEqual(d["selected_indices"][m], [2, 0])
            self.assertEqual(d["scoped_aggregation"]["modules"][m]["alpha"], [1, 1])

    def test_functional_energy_uses_unchanged_original_same_support_calculation(self):
        from afl_vlm.methods.ours_hc.payload import functional_energies

        method, server = self.setup_method()
        packet = self.packet(method)
        _, raw = functional_energies(self.base, self.base, packet)
        report = server.receive(packet)[0].metadata["router_diagnostics"]
        self.assertEqual(report["functional_local_energy"], raw)

    def test_export_records_real_per_component_coefficients_and_exit_events(self):
        method, server = self.setup_method()
        result = server.receive(self.packet(method))[0]
        exporter = runpy.run_path(str(ROOT / "tools/export_ours_router_diagnostics.py"))["export"]
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "events.jsonl").write_text(
                json.dumps({"result_metadata": [result.metadata]}) + "\n"
            )
            count = exporter(run, run / "export")
            self.assertEqual(count["components"], 2 * len(method.modules))
            self.assertEqual(count["resolutions"], 1)
            with (run / "export/components.csv").open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertTrue(all(float(row["alpha"]) == 1 for row in rows))

    def test_legacy_on_arrival_math_remains_historical(self):
        method = OursRouter({"history_age_enabled": False})
        method.configure_server(self.base, self.clients)
        packet = self.packet(method)
        mutation = method.on_arrival(packet, SimpleNamespace(global_state=self.base, version=0))[0]
        self.assertIsNone(mutation.on_commit)
        for m, c in packet.metadata["module_sensitivity"].items():
            self.assertAlmostEqual(
                mutation.metadata["router_diagnostics"]["module_alpha"][m], c / (c + 1 + 1e-12)
            )


class ScopedConfigurationChecks(unittest.TestCase):
    def test_modes_conflicts_and_optional_cap_do_not_leak_into_baselines(self):
        self.assertFalse(OursRouter().capabilities.requires_job_resolution)
        self.assertTrue(
            OursRouter(
                {"rank_aggregation_mode": "overlap_switch"}
            ).capabilities.requires_job_resolution
        )
        for params in (
            {"rank_aggregation_mode": "bad"},
            {"selection_window_size": 0},
            {"server_step_size": 0},
            {"base_precision": 1},
            {"alpha_max": 0},
        ):
            with self.assertRaises(ValueError):
                OursRouter(params)
        for name in method_names():
            if name != "ours_router":
                method = create_method(name)
                self.assertFalse(method.capabilities.requires_job_resolution)
                self.assertEqual(method.on_job_resolved("unknown", 0, "cancelled"), {})

    def test_legacy_arrival_ast_and_common_training_source_are_preserved(self):
        path = "code/afl_vlm/methods/ours_router/method.py"
        previous = subprocess.check_output(["git", "show", "7c5638b:" + path]).decode("utf-8")
        before = ast.parse(previous)
        after = ast.parse((ROOT / path).read_text("utf-8"))

        def function(tree):
            return next(
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "on_arrival"
            )

        self.assertEqual(
            [ast.dump(n) for n in function(before).body],
            [ast.dump(n) for n in function(after).body[2:]],
        )
        for path in (
            "code/afl_vlm/models/llava15.py",
        ):
            self.assertEqual(
                subprocess.check_output(["git", "show", "7c5638b:" + path]),
                (ROOT / path).read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
