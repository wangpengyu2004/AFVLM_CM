"""Finished-task layer selection/state/aggregation algebra; never run a model."""

from __future__ import annotations

import copy
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from afl_vlm.federation.server import FederatedServer
from afl_vlm.federation.types import ClientSpec, ScheduledEvent, Update
from afl_vlm.methods.ours import Ours
from afl_vlm.methods.ours_layer_protection import layer_modules, max_extra, select_layers

from tools.export_ours_diagnostics import export_diagnostics

TASKS = ("cls", "caption", "vqa", "chart_vqa", "visual_reasoning", "grounding")
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def modules(layers=32):
    return [
        f"base_model.model.layers.{layer}.self_attn.{projection}"
        for layer in range(layers)
        for projection in PROJECTIONS
    ]


def sensitivity(layers=32, selected=range(8)):
    chosen = set(selected)
    return {
        module: 0.5 + (0.5 * layers / len(chosen) if layer in chosen else 0)
        for layer in range(layers)
        for module in modules(layers)[4 * layer : 4 * layer + 4]
    }


def model_state(layers=32):
    return {
        f"{module}.lora_{side}.default.weight": [[1.0]]
        for module in modules(layers)
        for side in ("A", "B")
    }


def configured(*, layers=32, rounds=None, **params):
    rounds = rounds or {task: (2 if task == "grounding" else 1) for task in TASKS}
    method = Ours(
        {
            "history_age_enabled": True,
            "history_age_mode": "selected_layers_max",
            "history_age_boost": 2.0,
            "history_age_time_scale": 12.0,
            **params,
        }
    )
    specs = [
        ClientSpec(f"{task}/client_{i}", task, task, 4, 1, 0, 0, 1)
        for task in TASKS
        for i in range(2)
    ]
    initial = model_state(layers)
    method.configure_server(initial, specs)
    events = [
        ScheduledEvent(
            index,
            cid,
            task,
            task,
            r,
            float(index),
            float(index + 1),
            1,
            1,
            0,
            1,
        )
        for index, (task, cid, r) in enumerate(
            (task, f"{task}/client_{i}", r)
            for task in TASKS
            for i in range(2)
            for r in range(rounds[task])
        )
    ]
    method.configure_schedule(events)
    return method, FederatedServer(None, method, len(specs), initial), events, specs


def incoming(server, task, client=0, local_round=0, scores=None, zero=False):
    base = server.state
    local = {name: [[row[0][0] + (0 if zero else 0.05)]] for name, row in base.items()}
    scores = scores or {name.split(".lora_")[0]: 1.0 for name in base}
    metadata = {
        module: {
            "module_name": module,
            "a_name": f"{module}.lora_A.default.weight",
            "b_name": f"{module}.lora_B.default.weight",
            "rank": 1,
            "in_features": 1,
            "out_features": 1,
            "scaling": 1.0,
        }
        for module in scores
    }
    cid = f"{task}/client_{client}"
    return Update(
        f"{cid}-r{local_round}",
        cid,
        task,
        task,
        4,
        local_round,
        server.version,
        1.0,
        base,
        local,
        {name: [[local[name][0][0] - base[name][0][0]]] for name in base},
        42,
        "hash",
        [],
        2,
        metadata={
            "module_sensitivity": scores,
            "sensitivity_observations": dict.fromkeys(scores, 2),
            "sensitivity_valid": True,
            "lora_metadata": metadata,
        },
    )


def send(server, task, client=0, local_round=0, **kwargs):
    return server.receive(incoming(server, task, client, local_round, **kwargs))[0]


class SelectedLayerProtectionTests(unittest.TestCase):
    def finish_grounding(self, method, server):
        for cid in range(2):
            for local_round in range(2):
                send(server, "grounding", cid, local_round, scores=sensitivity())

    def test_floor_removed_only_for_selection_and_actual_32_layer_threshold(self):
        history = sensitivity()
        before = copy.deepcopy(history)
        snapshot = select_layers(history, 0.5, 1e-12)
        self.assertEqual(snapshot["selected_layers"], list(range(8)))
        self.assertEqual(snapshot["threshold"], 1 / 32)
        self.assertEqual(snapshot["layer_score"]["0"], 2.0)
        self.assertEqual(snapshot["layer_score"]["31"], 0)
        self.assertEqual(snapshot["history"], before)
        self.assertEqual(history, before)
        for chosen in (range(1), range(11), range(16)):
            self.assertEqual(
                select_layers(sensitivity(selected=chosen), 0.5, 1e-12)["selected_layers"],
                list(chosen),
            )

    def test_no_forced_top_k_for_zero_uniform_or_indistinguishable_scores(self):
        for value in (0.5, 1.0):
            history = dict.fromkeys(modules(), value)
            self.assertEqual(select_layers(history, 0.5, 1e-12)["selected_layers"], [])
        history = dict.fromkeys(modules(), 1.0)
        history[modules()[0]] += 1e-10
        self.assertEqual(select_layers(history, 0.5, 1e-12)["selected_layers"], [])
        with self.assertRaisesRegex(ValueError, "four"):
            layer_modules(modules()[:-1])
        for value in (float("nan"), float("inf"), -1):
            history[modules()[0]] = value
            with self.assertRaisesRegex(ValueError, "finite"):
                select_layers(history, 0.5, 1e-12)

    def test_projection_strength_preserved_and_overlap_uses_max_not_sum(self):
        names = modules(2)
        left = dict.fromkeys(names, 0.5)
        right = dict(left)
        for projection, value in zip(PROJECTIONS, (4, 2, 3, 2), strict=True):
            left[f"base_model.model.layers.0.self_attn.{projection}"] = value
        for projection, value in zip(PROJECTIONS, (2, 4, 3, 2), strict=True):
            right[f"base_model.model.layers.0.self_attn.{projection}"] = value
        snapshots = {"a": select_layers(left, 0.5, 1e-12), "b": select_layers(right, 0.5, 1e-12)}
        extra, owner = max_extra(snapshots, {"a": 1 / 6, "b": 1 / 6}, {"a": 2, "b": 2}, names)
        self.assertAlmostEqual(extra[names[0]], 4 / 3)
        self.assertEqual((owner[names[0]], owner[names[1]]), ("a", "b"))
        self.assertEqual(extra[names[4]], 0)
        self.assertIsNone(owner[names[4]])
        duplicate = {**snapshots, "c": copy.deepcopy(snapshots["a"])}
        same, _ = max_extra(
            duplicate, dict.fromkeys(duplicate, 1 / 6), dict.fromkeys(duplicate, 2), names
        )
        self.assertEqual(same, extra)

    def test_both_clients_ten_successes_needed_rejection_and_retry_do_not_end_early(self):
        method, server, _, _ = configured(
            rounds={task: 10 if task == "grounding" else 1 for task in TASKS}
        )
        first = sensitivity(selected=range(4))
        last = sensitivity(selected=range(12))
        for cid in range(2):
            for r in range(10 if cid == 0 else 9):
                send(server, "grounding", cid, r, scores=first)
        self.assertEqual(method.history_accepted_updates, 19)
        self.assertEqual(method.task_pending_updates["grounding"], 1)
        rejected = send(server, "grounding", 1, 9, scores=last, zero=True)
        self.assertEqual(rejected.applied_weight, 0)
        self.assertEqual(method.task_pending_updates["grounding"], 1)
        self.assertNotIn("grounding", method.task_finished_at)
        self.assertNotIn(("grounding/client_1", 9), method.history_resolved_jobs)
        result = send(server, "grounding", 1, 9, scores=last)
        self.assertEqual(method.task_finished_at["grounding"], 20)
        snapshot = method.task_end_snapshots["grounding"]
        beta = 0.9
        expected = (
            beta * first[modules()[0]] * (1 - beta**19) + (1 - beta) * last[modules()[0]]
        ) / (1 - beta**20)
        self.assertAlmostEqual(snapshot["history"][modules()[0]], expected)
        self.assertNotEqual(snapshot["history"][modules()[0]], last[modules()[0]])
        self.assertEqual(
            snapshot["completion_by_client"],
            {
                "grounding/client_0": {"planned": 10, "accepted": 10, "pending": 0},
                "grounding/client_1": {"planned": 10, "accepted": 10, "pending": 0},
            },
        )
        self.assertEqual(
            result.metadata["ours_diagnostics"]["memory"]["finished_task_snapshot"], snapshot
        )
        counter = method.history_accepted_updates
        with self.assertRaisesRegex(ValueError, "already resolved"):
            send(server, "grounding", 1, 9, scores=last)
        self.assertEqual(method.history_accepted_updates, counter)

    def test_aggregation_pre_age_first_next_zero_then_positive_no_double_age(self):
        method, server, _, _ = configured()
        self.finish_grounding(method, server)
        frozen = copy.deepcopy(method.task_end_snapshots["grounding"])
        first = send(server, "cls").metadata["ours_diagnostics"]
        self.assertEqual(first["memory"]["history_age_before"]["age_by_task"]["grounding"], 0)
        self.assertEqual(first["memory"]["history_age_before"]["positive_extra_module_count"], 0)
        self.assertEqual(
            first["memory"]["history_age_before"]["selected_layer_union"], list(range(8))
        )
        self.assertEqual(
            first["memory"]["historical_precision_before"],
            first["memory"]["historical_precision_unweighted_before"],
        )
        before_state = server.state
        second = send(server, "caption").metadata["ours_diagnostics"]
        age = second["memory"]["history_age_before"]
        boost = 2 * (1 - math.exp(-1 / 12))
        for index, module in enumerate(modules()):
            extra = boost * frozen["history"][module] / 6 if index < 8 * 4 else 0
            self.assertAlmostEqual(age["extra_precision_by_module"][module], extra)
            base = second["memory"]["historical_precision_unweighted_before"][module]
            self.assertAlmostEqual(
                second["memory"]["historical_precision_before"][module], base + extra
            )
            alpha = second["aggregation"]["module_alpha_by_module"][module]
            weight = second["frequency"]["weight"]
            self.assertAlmostEqual(alpha, min(0.5, weight / (weight + base + extra + 1e-12)))
            for side in ("A", "B"):
                name = f"{module}.lora_{side}.default.weight"
                self.assertAlmostEqual(
                    server.state[name][0][0], before_state[name][0][0] + alpha * 0.05
                )
        self.assertEqual(method.task_end_snapshots["grounding"], frozen)

    def test_final_snapshot_not_used_retroactively_and_no_synthetic_update(self):
        method, server, _, _ = configured(layers=2)
        send(server, "grounding", 0, 0, scores=sensitivity(2, range(1)))
        send(server, "grounding", 0, 1, scores=sensitivity(2, range(1)))
        send(server, "grounding", 1, 0, scores=sensitivity(2, range(1)))
        result = send(server, "grounding", 1, 1, scores=sensitivity(2, range(1)))
        before = result.metadata["ours_diagnostics"]["memory"]["history_age_before"]
        after = result.metadata["ours_diagnostics"]["memory"]["history_age_after"]
        self.assertEqual(before["finished_at"], {})
        self.assertEqual(after["finished_at"], {"grounding": 4})
        self.assertEqual(after["positive_extra_module_count"], 0)
        self.assertEqual((server.version, server.accepted_updates), (4, 4))
        self.assertEqual(server.finish(), [])

    def test_prefix_matches_old_age_rule_and_later_task_never_resets_old_age(self):
        method, server, _, _ = configured()
        legacy, other, _, _ = configured(history_age_mode="all_modules_sum")
        prefix = (
            ("grounding", 0, 0),
            ("grounding", 0, 1),
            ("grounding", 1, 0),
            ("grounding", 1, 1),
            ("chart_vqa", 0, 0),
        )
        for task, cid, r in prefix:
            left = send(server, task, cid, r, scores=sensitivity())
            right = send(other, task, cid, r, scores=sensitivity())
            self.assertEqual(
                (server.state, left.applied_weight), (other.state, right.applied_weight)
            )
        send(server, "chart_vqa", 1, scores=sensitivity(selected=range(4, 12)))
        self.assertEqual(method.task_finished_at, {"grounding": 4, "chart_vqa": 6})
        send(server, "cls")
        event = send(server, "caption").metadata["ours_diagnostics"]
        age = event["memory"]["history_age_before"]
        self.assertEqual(age["age_by_task"]["grounding"], 3)
        self.assertEqual(age["age_by_task"]["chart_vqa"], 1)
        for module in modules():
            layer = age["layer_by_module"][module]
            candidates = {
                task: (
                    2 * (1 - math.exp(-age["age_by_task"][task] / 12)) * row["history"][module] / 6
                    if row["layer_mask"][layer]
                    else 0
                )
                for task, row in method.task_end_snapshots.items()
            }
            self.assertAlmostEqual(
                age["extra_precision_by_module"][module], max(candidates.values())
            )
        self.assertEqual(age["max_source_by_module"][modules()[4 * 4]], "grounding")

    def test_disabled_zero_boost_and_no_selected_layers_reduce_to_base(self):
        for scores, params in (
            (sensitivity(), {"history_age_enabled": False}),
            (sensitivity(), {"history_age_boost": 0}),
            (dict.fromkeys(modules(), 1.0), {}),
        ):
            method, server, _, _ = configured(**params)
            reference, other, _, _ = configured(history_age_enabled=False)
            for task, cid, r in (
                ("grounding", 0, 0),
                ("grounding", 0, 1),
                ("grounding", 1, 0),
                ("grounding", 1, 1),
                ("cls", 0, 0),
                ("caption", 0, 0),
            ):
                one = send(server, task, cid, r, scores=scores)
                two = send(other, task, cid, r, scores=scores)
                self.assertEqual(one.applied_weight, two.applied_weight)
                self.assertEqual(server.state, other.state)
                self.assertEqual(method.task_memory, reference.task_memory)
                self.assertEqual(method.task_frequency_window, reference.task_frequency_window)

    def test_zero_weight_and_terminal_snapshot_error_leave_accepted_state_unchanged(self):
        method, server, _, _ = configured(layers=2)
        row = incoming(server, "cls", scores=sensitivity(2, range(1)))
        method.prepare_upload(row, None)
        saved = method.state_dict()
        with patch("afl_vlm.methods.ours.staleness_reliability", return_value=0.0):
            result = method.on_arrival(row, server.context())[0]
        self.assertEqual(result.applied_weight, 0)
        self.assertEqual(method.state_dict(), saved)
        send(server, "cls", 0, scores=sensitivity(2, range(1)))
        terminal = incoming(server, "cls", 1, scores=sensitivity(2, range(1)))
        saved = method.state_dict()
        with patch.object(method, "_end_snapshot", side_effect=ValueError("bad snapshot")):
            with self.assertRaisesRegex(ValueError, "bad snapshot"):
                method.on_arrival(terminal, server.context())
        self.assertEqual(method.state_dict(), saved)

    def test_snapshot_roundtrip_preserves_next_update_and_rejects_corruption(self):
        method, server, events, specs = configured()
        self.finish_grounding(method, server)
        send(server, "cls")
        saved = json.loads(json.dumps(method.state_dict()))
        restored = Ours()
        restored.load_state_dict(saved)
        restored.configure_server(server.state, specs)
        restored.configure_schedule(events)
        self.assertEqual(restored.state_dict(), saved)
        row = incoming(server, "caption")
        left = method.on_arrival(copy.deepcopy(row), server.context())[0]
        right = restored.on_arrival(copy.deepcopy(row), server.context())[0]
        self.assertEqual((left.new_state, left.metadata), (right.new_state, right.metadata))
        for field, bad in (("history", {}), ("selected_layers", [31]), ("finished_at", 0)):
            corrupt = copy.deepcopy(saved)
            corrupt["history_age_state"]["finished_snapshots"]["grounding"][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                Ours().load_state_dict(corrupt)
        corrupt = copy.deepcopy(saved)
        del corrupt["history_age_state"]["finished_snapshots"]
        with self.assertRaisesRegex(ValueError, "snapshot"):
            Ours().load_state_dict(corrupt)

    def test_disabled_selected_mode_checkpoint_needs_no_fake_completion_plan(self):
        method, server, _, _ = configured(history_age_enabled=False)
        send(server, "cls")
        restored = Ours()
        restored.load_state_dict(method.state_dict())
        self.assertEqual(restored.state_dict(), method.state_dict())
        self.assertEqual(restored.task_end_snapshots, {})

    def test_missing_client_or_wrong_update_id_cannot_complete_task(self):
        method, server, events, specs = configured(layers=2)
        other = Ours(method.params)
        other.configure_server(server.state, specs)
        with self.assertRaisesRegex(ValueError, "every configured client"):
            other.configure_schedule([e for e in events if e.client_id != "cls/client_1"])
        bad = incoming(server, "cls")
        bad.update_id = "different-id-for-the-same-job"
        with self.assertRaisesRegex(ValueError, "unique Plan job"):
            method.on_arrival(bad, server.context())
        self.assertEqual(method.history_accepted_updates, 0)

    def test_export_records_end_history_masks_union_extra_source_and_alpha(self):
        method, server, _, _ = configured(layers=2)
        events = []
        for index, (task, cid, r) in enumerate(
            (
                ("grounding", 0, 0),
                ("grounding", 0, 1),
                ("grounding", 1, 0),
                ("grounding", 1, 1),
                ("cls", 0, 0),
                ("caption", 0, 0),
            )
        ):
            result = send(server, task, cid, r, scores=sensitivity(2, range(1)))
            events.append(
                {
                    "event_id": index,
                    "server_version_after": server.version,
                    "result_metadata": [result.metadata],
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "events.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
            exported = export_diagnostics(source, root / "export")
            self.assertEqual(exported["completed_tasks"], 1)
            with Path(exported["completion_csv"]).open(encoding="utf-8-sig", newline="") as handle:
                completed = list(csv.DictReader(handle))
            self.assertEqual(completed[0]["task"], "grounding")
            self.assertEqual(json.loads(completed[0]["selected_layers"]), [0])
            self.assertEqual(
                json.loads(completed[0]["history"]),
                method.task_end_snapshots["grounding"]["history"],
            )
            with Path(exported["update_csv"]).open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[-2]["history_age_mode"], "selected_layers_max")
            self.assertEqual(json.loads(rows[-2]["selected_layer_union_before"]), [0])
            self.assertEqual(rows[-2]["positive_extra_modules_before"], "0")
            self.assertEqual(rows[-1]["positive_extra_modules_before"], "4")
            with Path(exported["module_csv"]).open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))[-8:]
            self.assertEqual(rows[0]["history_age_max_source_before"], "grounding")
            self.assertGreater(float(rows[0]["history_age_extra_before"]), 0)
            self.assertEqual(rows[-1]["history_age_extra_before"], "0.0")
            self.assertEqual(rows[-1]["decoder_layer"], "1")


if __name__ == "__main__":
    unittest.main()
