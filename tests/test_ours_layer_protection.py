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
from afl_vlm.methods.ours import Ours, compute_module_alpha
from afl_vlm.methods.ours_layer_protection import layer_modules, max_extra, select_layers, sum_extra

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


class FirstObservationHistoryTests(unittest.TestCase):
    def configured(self, **params):
        return configured(
            layers=2, history_update_mode="first_observation_ema", base_precision=0.0, **params
        )

    def test_unobserved_unit_priors_mean_one_without_outer_base(self):
        method, _, _, _ = self.configured()
        for task in TASKS:
            self.assertEqual(method.task_memory_counts[task], 0)
            for module in modules(2):
                self.assertEqual(method.task_memory[task][module], 1)
                self.assertEqual(method._corrected_task_memory(task, module), 1)
        for module in modules(2):
            self.assertAlmostEqual(method._historical_precision(module), 1)
        self.assertEqual(method._base_precision(), 0)

    def test_first_replaces_prior_then_plain_ema_without_correction(self):
        method, server, _, _ = self.configured(rounds=dict.fromkeys(TASKS, 3))
        name = modules(2)[0]
        for (client, local_round), value, expected in zip(
            ((0, 0), (1, 0), (0, 1)), (1, 2, 3), (1, 1.1, 1.29), strict=True
        ):
            result = send(
                server, "cls", client, local_round, scores=dict.fromkeys(modules(2), value)
            )
            memory = result.metadata["ours_diagnostics"]["memory"]
            self.assertAlmostEqual(method.task_memory["cls"][name], expected)
            self.assertAlmostEqual(method._corrected_task_memory("cls", name), expected)
            self.assertAlmostEqual(memory["task_memory_after"][name], expected)
            self.assertEqual(memory["task_memory_raw_after"], memory["task_memory_after"])
            self.assertEqual(memory["history_update_mode"], "first_observation_ema")
            self.assertFalse(memory["history_bias_correction"])
            self.assertEqual(memory["base_precision"], 0)
            self.assertEqual(result.metadata["ours_diagnostics"]["schema_version"], 6)
            for task in set(TASKS) - {"cls"}:
                self.assertEqual(method.task_memory[task][name], 1)
            self.assertAlmostEqual(method._historical_precision(name), (5 + expected) / 6)

    def test_rejected_first_observation_does_not_replace_prior_or_advance_any_memory(self):
        method, server, _, _ = self.configured()
        row = incoming(server, "cls", zero=True, scores=sensitivity(2, [0]))
        # First annotation validation resolves fixed LoRA scaling metadata.
        method.prepare_upload(row, None)
        before = copy.deepcopy(method.state_dict())
        rejected = server.receive(row)[0]
        self.assertEqual(method.state_dict(), before)
        self.assertEqual(server.version, 0)
        self.assertEqual(rejected.applied_weight, 0)
        result = send(server, "cls", scores=sensitivity(2, [0]))
        self.assertEqual(method.task_memory["cls"], sensitivity(2, [0]))
        self.assertEqual(method.task_memory_counts["cls"], 1)
        memory = result.metadata["ours_diagnostics"]["memory"]
        self.assertEqual(memory["task_memory_count_before"], 0)

    def test_alpha_reads_prior_before_first_fusion_not_new_history(self):
        method, server, _, _ = self.configured()
        result = send(server, "cls", scores=sensitivity(2, [0]))
        diagnostics = result.metadata["ours_diagnostics"]
        for module, c in sensitivity(2, [0]).items():
            alpha = min(0.5, c / (c + 1 + 1e-12))
            self.assertAlmostEqual(
                diagnostics["aggregation"]["module_alpha_by_module"][module], alpha
            )
            self.assertAlmostEqual(diagnostics["memory"]["historical_precision_before"][module], 1)
            self.assertAlmostEqual(
                diagnostics["memory"]["historical_precision_after"][module], (5 + c) / 6
            )
            for side in ("A", "B"):
                key = f"{module}.lora_{side}.default.weight"
                self.assertAlmostEqual(server.state[key][0][0], 1 + 0.05 * alpha)

    def test_final_snapshot_uses_final_plain_ema_mask_and_unchanged_max_age(self):
        method, server, _, _ = self.configured()
        scores = [sensitivity(2, [0])] * 3 + [sensitivity(2, [1])]
        jobs = ((0, 0), (0, 1), (1, 0), (1, 1))
        for (client, r), values in zip(jobs, scores, strict=True):
            send(server, "grounding", client, r, scores=values)
        snapshot = method.task_end_snapshots["grounding"]
        self.assertEqual(snapshot["selected_layers"], [0])
        self.assertEqual(snapshot["history"], method.task_memory["grounding"])
        name = modules(2)[0]
        self.assertAlmostEqual(snapshot["history"][name], 1.4)
        self.assertEqual(snapshot["finished_at"], 4)
        frozen = copy.deepcopy(snapshot)
        first = send(server, "cls").metadata["ours_diagnostics"]["memory"]
        self.assertEqual(first["history_age_before"]["extra_precision_by_module"][name], 0)
        second = send(server, "caption").metadata["ours_diagnostics"]["memory"]
        expected = 2 * (1 - math.exp(-1 / 12)) * 1.4 / 6
        self.assertAlmostEqual(
            second["history_age_before"]["extra_precision_by_module"][name], expected
        )
        self.assertEqual(method.task_end_snapshots["grounding"], frozen)

    def test_state_roundtrip_restores_new_history_and_rejects_mixed_definitions(self):
        method, server, events, specs = self.configured()
        for cid in range(2):
            for r in range(2):
                send(server, "grounding", cid, r, scores=sensitivity(2, [0]))
        saved = json.loads(json.dumps(method.state_dict()))
        self.assertEqual(saved["state_schema_version"], 5)
        self.assertEqual(saved["history_update_mode"], "first_observation_ema")
        restored = Ours()
        restored.load_state_dict(saved)
        restored.configure_server(server.state, specs)
        restored.configure_schedule(events)
        self.assertEqual(restored.state_dict(), saved)
        row = incoming(server, "cls", scores=sensitivity(2, [1]))
        left = method.on_arrival(copy.deepcopy(row), server.context())[0]
        right = restored.on_arrival(copy.deepcopy(row), server.context())[0]
        self.assertEqual((left.new_state, left.metadata), (right.new_state, right.metadata))
        for corrupt in ("state_schema_version", "history_update_mode", "prior", "history"):
            bad = copy.deepcopy(saved)
            if corrupt == "prior":
                bad["task_memory"]["cls"][modules(2)[0]] = 0
            elif corrupt == "history":
                bad["history_age_state"]["finished_snapshots"]["grounding"]["history"][
                    modules(2)[0]
                ] *= 2
            else:
                bad[corrupt] = 4 if corrupt == "state_schema_version" else "wrong"
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                Ours().load_state_dict(bad)

    def test_missing_mode_preserves_legacy_zero_prior_corrected_ema_and_schema(self):
        method, server, _, _ = configured(layers=2)
        self.assertEqual(method.task_memory["cls"][modules(2)[0]], 0)
        self.assertEqual(method._base_precision(), 1)
        send(server, "cls", scores=sensitivity(2, [0]))
        self.assertAlmostEqual(method.task_memory["cls"][modules(2)[0]], 0.15)
        self.assertAlmostEqual(method._corrected_task_memory("cls", modules(2)[0]), 1.5)
        self.assertEqual(method.state_dict()["state_schema_version"], 4)
        restored = Ours({"history_update_mode": "first_observation_ema"})
        restored.load_state_dict(method.state_dict())
        self.assertFalse(restored._first_observation_history())
        self.assertEqual(restored.state_dict(), method.state_dict())

    def test_parameter_validation_no_hidden_outer_base_or_invalid_mode(self):
        for params in (
            {"history_update_mode": "wrong"},
            {"history_update_mode": "first_observation_ema", "base_precision": 1},
            {"history_update_mode": "first_observation_ema", "base_precision": float("nan")},
            {"base_precision": 0},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                Ours(params)
        new = Ours({"history_update_mode": "first_observation_ema"})
        self.assertEqual(new._base_precision(), 0)

    def test_beta_zero_and_disabled_age_have_same_history_definition(self):
        method, server, _, _ = self.configured(history_beta=0, history_age_enabled=False)
        send(server, "cls", scores=sensitivity(2, [0]))
        send(server, "cls", 1, scores=sensitivity(2, [1]))
        self.assertEqual(method.task_memory["cls"], sensitivity(2, [1]))
        restored = Ours()
        restored.load_state_dict(method.state_dict())
        self.assertEqual(restored.state_dict(), method.state_dict())
        self.assertEqual(restored.task_end_snapshots, {})

    def test_normalized_inputs_keep_history_module_mean_one(self):
        method, server, _, _ = self.configured(rounds=dict.fromkeys(TASKS, 2))
        for task in TASKS:
            for cid in range(2):
                for r in range(2):
                    send(server, task, cid, r, scores=sensitivity(2, [(cid + r) % 2]))
                    self.assertAlmostEqual(sum(method.task_memory[task].values()) / 8, 1)
                    base = [
                        method._historical_precision(m, apply_age_weight=False) for m in modules(2)
                    ]
                    self.assertAlmostEqual(sum(base) / 8, 1)

    def test_zero_alpha_rejection_cannot_initialize_new_history_in_either_age_mode(self):
        for mode in ("selected_layers_max", "all_modules_sum"):
            method, server, _, _ = self.configured(history_age_mode=mode)
            row = incoming(server, "cls", scores=sensitivity(2, [0]))
            method.prepare_upload(row, None)
            before = copy.deepcopy(method.state_dict())
            with patch("afl_vlm.methods.ours.staleness_reliability", return_value=0):
                result = server.receive(row)[0]
            self.assertEqual(result.applied_weight, 0)
            self.assertEqual(method.state_dict(), before)

    def test_new_diagnostics_export_and_all_three_inherited_configs(self):
        from afl_vlm.config import load_config, validate_config

        root = Path(__file__).resolve().parents[1]
        for setting in (2, 5, 10):
            config = load_config(
                root / f"configs/experiments/llava/afvlm_cm/{setting}clients/ours.yaml"
            )
            validate_config(config)
            self.assertEqual(config["method"]["params"]["base_precision"], 0)
            self.assertEqual(
                config["method"]["params"]["history_update_mode"], "first_observation_ema"
            )
        method, server, _, _ = self.configured()
        events = []
        for index, (cid, r) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            result = send(server, "grounding", cid, r, scores=sensitivity(2, [0]))
            events.append({
                "event_id": index, "server_version_after": server.version,
                "result_metadata": [result.metadata],
            })
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "events.jsonl"
            source.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
            result = export_diagnostics(source, folder / "export")
            for key in ("update_csv", "module_csv", "completion_csv"):
                with Path(result[key]).open(encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(rows[0]["history_update_mode"], "first_observation_ema")
                self.assertEqual(rows[0]["history_bias_correction"], "False")
                if key != "completion_csv":
                    self.assertEqual(float(rows[0]["base_precision"]), 0)
                    self.assertEqual(float(rows[0]["unobserved_task_sensitivity"]), 1)
                if key == "module_csv":
                    self.assertEqual(float(rows[0]["task_memory_before"]), 1)
                    self.assertEqual(float(rows[0]["task_memory_raw_before"]), 1)
                    self.assertEqual(float(rows[0]["task_memory_after"]), 1.5)
                    self.assertEqual(float(rows[0]["task_memory_raw_after"]), 1.5)


class UncappedNormalizedAlphaTests(unittest.TestCase):
    def configured(self, **params):
        return configured(
            layers=2, history_update_mode="first_observation_ema", base_precision=0.0,
            alpha_max=None, **params,
        )

    def test_null_cap_preserves_fraction_above_half_but_not_unbounded_ratio(self):
        alpha = compute_module_alpha(4, 0.8, 1, alpha_max=None)
        self.assertAlmostEqual(alpha, 3.2 / (3.2 + 1 + 1e-12))
        self.assertGreater(alpha, 0.5)
        self.assertLess(alpha, 1)
        self.assertNotAlmostEqual(alpha, 3.2)
        self.assertEqual(compute_module_alpha(4, 0.8, 1), 0.5)
        self.assertEqual(compute_module_alpha(4, 0.8, 1, alpha_max=0.3), 0.3)

    def test_fraction_bounds_zero_and_monotonicity(self):
        for c in (0, 0.5, 1, 2, 20):
            for r in (0, 0.1, 0.5, 1):
                for omega in (0, 0.5, 1, 10):
                    value = compute_module_alpha(c, r, omega, client_weight=1.5, alpha_max=None)
                    self.assertGreaterEqual(value, 0)
                    self.assertLessEqual(value, 1)
        low = compute_module_alpha(1, 1, 1, alpha_max=None)
        self.assertGreater(compute_module_alpha(2, 1, 1, alpha_max=None), low)
        self.assertLess(compute_module_alpha(1, 1, 2, alpha_max=None), low)
        self.assertLess(compute_module_alpha(1, 0.5, 1, alpha_max=None), low)

    def test_actual_arrival_applies_uncapped_shared_AB_increment_and_logs_precision(self):
        method, server, _, _ = self.configured()
        result = send(server, "cls", scores=sensitivity(2, [0]))
        diagnostic = result.metadata["ours_diagnostics"]
        agg = diagnostic["aggregation"]
        self.assertIsNone(agg["alpha_max"])
        self.assertEqual(agg["alpha_rule"], "normalized_precision_ratio")
        name = modules(2)[0]
        self.assertGreater(agg["module_alpha_by_module"][name], 0.5)
        for module, c in sensitivity(2, [0]).items():
            alpha = c / (c + 1 + 1e-12)
            self.assertAlmostEqual(agg["module_alpha_by_module"][module], alpha)
            self.assertEqual(agg["client_precision_by_module"][module], c)
            for side in ("A", "B"):
                self.assertAlmostEqual(
                    server.state[f"{module}.lora_{side}.default.weight"][0][0], 1 + 0.05 * alpha
                )
        self.assertEqual(method.task_memory["cls"], sensitivity(2, [0]))

    def test_null_cap_survives_state_roundtrip(self):
        method, server, events, specs = self.configured()
        send(server, "cls", scores=sensitivity(2, [0]))
        saved = json.loads(json.dumps(method.state_dict()))
        restored = Ours()
        restored.load_state_dict(saved)
        restored.configure_server(server.state, specs)
        restored.configure_schedule(events)
        self.assertIsNone(restored._alpha_max())
        self.assertEqual(restored.state_dict(), saved)
        row = incoming(server, "caption", scores=sensitivity(2, [1]))
        a = method.on_arrival(copy.deepcopy(row), server.context())[0]
        b = restored.on_arrival(copy.deepcopy(row), server.context())[0]
        self.assertEqual((a.new_state, a.metadata), (b.new_state, b.metadata))

    def test_template_inheritance_and_export_distinguish_cap_config_from_observed_max(self):
        from afl_vlm.config import load_config, validate_config

        root = Path(__file__).resolve().parents[1]
        for setting in (2, 5, 10):
            config = load_config(
                root / f"configs/experiments/llava/afvlm_cm/{setting}clients/ours.yaml"
            )
            validate_config(config)
            self.assertIsNone(config["method"]["params"]["alpha_max"])
        _, server, _, _ = self.configured()
        result = send(server, "cls", scores=sensitivity(2, [0]))
        event = {"event_id": 0, "server_version_after": 1, "result_metadata": [result.metadata]}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / "events.jsonl"
            path.write_text(json.dumps(event) + "\n", encoding="utf-8")
            exported = export_diagnostics(path, folder / "export")
            for key in ("update_csv", "module_csv"):
                with Path(exported[key]).open(encoding="utf-8-sig", newline="") as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(row["alpha_rule"], "normalized_precision_ratio")
                self.assertEqual(row["alpha_max_configured"], "none")
                if key == "update_csv":
                    self.assertGreater(float(row["alpha_max"]), 0.5)
                else:
                    self.assertEqual(float(row["client_precision"]), 1.5)

    def test_invalid_alpha_params_rejected_and_missing_legacy_cap_is_half(self):
        for cap in (0, -1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "alpha_max"):
                Ours({"alpha_max": cap})
        self.assertEqual(Ours()._alpha_max(), 0.5)
        self.assertIsNone(Ours({"alpha_max": None})._alpha_max())


class CumulativeLayerProtectionTests(unittest.TestCase):
    def configured(self, **params):
        return configured(
            layers=2, history_update_mode="first_observation_ema", base_precision=0.0,
            alpha_max=None, history_age_mode="selected_layers_sum",
            history_age_time_scale=18.0, **params,
        )

    def prefix(self, server):
        results = []
        for task, cid, r in (
            ("grounding", 0, 0), ("grounding", 0, 1),
            ("grounding", 1, 0), ("grounding", 1, 1),
            ("chart_vqa", 0, 0), ("chart_vqa", 1, 0), ("cls", 0, 0),
        ):
            results.append(send(server, task, cid, r, scores=sensitivity(2, [0])))
        return results

    def test_sum_retains_fixed_task_weights_masks_and_projection_strength(self):
        names = modules(2)
        left = dict.fromkeys(names, 0.5)
        right = dict(left)
        for i, (a, b) in enumerate(zip((4, 2, 3, 2), (2, 4, 3, 2), strict=True)):
            left[names[i]], right[names[i]] = a, b
        snapshots = {"a": select_layers(left, 0.5, 1e-12), "b": select_layers(right, 0.5, 1e-12)}
        extra, demands = sum_extra(
            snapshots, dict.fromkeys(snapshots, 1 / 6), {"a": 0.6, "b": 0.6}, names
        )
        self.assertAlmostEqual(extra[names[0]], 0.4 + 0.2)
        self.assertAlmostEqual(extra[names[1]], 0.2 + 0.4)
        self.assertAlmostEqual(demands["a"][names[0]], 0.4)
        self.assertEqual(extra[names[4]], 0)
        duplicate = {**snapshots, "c": copy.deepcopy(snapshots["a"])}
        more, _ = sum_extra(
            duplicate, dict.fromkeys(duplicate, 1 / 6), dict.fromkeys(duplicate, 0.6), names
        )
        self.assertAlmostEqual(more[names[0]], extra[names[0]] + 0.4)
        # Zero/one finished tasks agree with the preserved max mode, without L/K rescale.
        for subset in ({}, {"a": snapshots["a"]}):
            one, _ = sum_extra(subset, {"a": 1 / 6}, {"a": 0.6}, names)
            maximum, _ = max_extra(subset, {"a": 1 / 6}, {"a": 0.6}, names)
            self.assertEqual(one, maximum)
        for bad in (-1, float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "finite"):
                sum_extra(snapshots, dict.fromkeys(snapshots, 1 / 6), {"a": bad, "b": 1}, names)

    def test_actual_sum_omega_alpha_increment_and_task_contributions(self):
        method, server, _, _ = self.configured()
        self.prefix(server)
        before = copy.deepcopy(server.state)
        snapshots = copy.deepcopy(method.task_end_snapshots)
        result = send(server, "caption", scores=sensitivity(2, [0]))
        diagnostic = result.metadata["ours_diagnostics"]
        memory = diagnostic["memory"]
        age = memory["history_age_before"]
        self.assertEqual(age["mode"], "selected_layers_sum")
        self.assertNotIn("max_source_by_module", age)
        self.assertEqual(age["age_by_task"]["grounding"], 3)
        self.assertEqual(age["age_by_task"]["chart_vqa"], 1)
        for name, c in sensitivity(2, [0]).items():
            expected = {
                task: (2 * (1 - math.exp(-age["age_by_task"][task] / 18))
                       * row["history"][name] / 6 if row["layer_mask"][age["layer_by_module"][name]]
                       else 0)
                for task, row in snapshots.items()
            }
            for task, demand in expected.items():
                self.assertAlmostEqual(age["extra_precision_by_task"][task][name], demand)
            total = math.fsum(expected.values())
            self.assertAlmostEqual(age["extra_precision_by_module"][name], total)
            base = memory["historical_precision_unweighted_before"][name]
            omega = memory["historical_precision_before"][name]
            self.assertAlmostEqual(omega, base + total)
            n = diagnostic["aggregation"]["client_precision_by_module"][name]
            alpha = diagnostic["aggregation"]["module_alpha_by_module"][name]
            self.assertAlmostEqual(alpha, n / (n + omega + 1e-12))
            for side in ("A", "B"):
                key = f"{name}.lora_{side}.default.weight"
                self.assertAlmostEqual(server.state[key][0][0], before[key][0][0] + 0.05 * alpha)
            self.assertEqual(age["contributor_count_by_module"][name], 2 if c > 1 else 0)
        self.assertEqual(method.task_end_snapshots, snapshots)
        self.assertEqual(method.task_memory["caption"], sensitivity(2, [0]))

    def test_only_composition_changes_not_ema_selection_counts_or_single_task_prefix(self):
        method, server, _, _ = self.configured()
        maximum, other, _, _ = configured(
            layers=2, history_update_mode="first_observation_ema", base_precision=0.0,
            alpha_max=None, history_age_time_scale=18.0,
        )
        left, right = self.prefix(server), self.prefix(other)
        self.assertEqual(server.state, other.state)
        self.assertEqual([r.applied_weight for r in left], [r.applied_weight for r in right])
        one = send(server, "caption", scores=sensitivity(2, [0])).metadata["ours_diagnostics"]
        two = send(other, "caption", scores=sensitivity(2, [0])).metadata["ours_diagnostics"]
        self.assertEqual(one["functional_staleness"], two["functional_staleness"])
        self.assertEqual(one["frequency"], two["frequency"])
        for attr in ("task_memory", "task_memory_counts", "task_frequency_window",
                     "task_end_snapshots", "history_accepted_updates", "task_finished_at",
                     "task_pending_updates", "history_resolved_jobs"):
            self.assertEqual(getattr(method, attr), getattr(maximum, attr))
        for name in modules(2):
            a = one["aggregation"]["module_alpha_by_module"][name]
            b = two["aggregation"]["module_alpha_by_module"][name]
            self.assertLessEqual(a, b)
        self.assertNotEqual(server.state, other.state)

    def test_disabled_zero_boost_or_uniform_history_has_no_extra_and_rejections_do_not_count(self):
        for params, scores in (
            ({"history_age_enabled": False}, sensitivity(2, [0])),
            ({"history_age_boost": 0}, sensitivity(2, [0])),
            ({}, dict.fromkeys(modules(2), 1.0)),
        ):
            method, server, _, _ = self.configured(**params)
            for cid, r in ((0, 0), (0, 1), (1, 0), (1, 1)):
                send(server, "grounding", cid, r, scores=scores)
            send(server, "cls", scores=scores)
            age = method._history_age_diagnostics()
            self.assertTrue(all(value == 0 for value in age["extra_precision_by_module"].values()))
            row = incoming(server, "caption", zero=True)
            saved = copy.deepcopy(method.state_dict())
            result = server.receive(row)[0]
            self.assertEqual(result.applied_weight, 0)
            self.assertEqual(method.state_dict(), saved)

    def test_sum_state_roundtrip_preserves_mode_and_next_update(self):
        method, server, events, specs = self.configured()
        self.prefix(server)
        saved = json.loads(json.dumps(method.state_dict()))
        restored = Ours()
        restored.load_state_dict(saved)
        restored.configure_server(server.state, specs)
        restored.configure_schedule(events)
        self.assertEqual(restored.params["history_age_mode"], "selected_layers_sum")
        self.assertEqual(restored.state_dict(), saved)
        row = incoming(server, "caption", scores=sensitivity(2, [0]))
        left = method.on_arrival(copy.deepcopy(row), server.context())[0]
        right = restored.on_arrival(copy.deepcopy(row), server.context())[0]
        self.assertEqual((left.new_state, left.metadata), (right.new_state, right.metadata))

    def test_sum_export_lists_all_contributors_not_fictitious_max_owner(self):
        method, server, _, _ = self.configured()
        results = self.prefix(server)
        results.append(send(server, "caption", scores=sensitivity(2, [0])))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "events.jsonl"
            source.write_text("".join(json.dumps({"event_id": index,
                "result_metadata": [row.metadata]}) + "\n" for index, row in enumerate(results)),
                encoding="utf-8")
            exported = export_diagnostics(source, root / "export")
            with Path(exported["update_csv"]).open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[-1]["history_age_mode"], "selected_layers_sum")
            self.assertEqual(rows[-1]["max_sources_before"], "")
            demands = json.loads(rows[-1]["extra_precision_by_task_before"])
            with Path(exported["module_csv"]).open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))[-8:]
            for row in rows[:4]:
                self.assertEqual(row["history_age_max_source_before"], "")
                self.assertEqual(row["history_age_contributor_count_before"], "2")
                self.assertEqual(json.loads(row["history_age_contributors_before"]),
                                 ["chart_vqa", "grounding"])
                self.assertEqual(json.loads(row["history_age_task_demands_before"]),
                                 {task: values[row["module"]] for task, values in demands.items()})
            self.assertEqual(rows[-1]["history_age_contributors_before"], "[]")

    def test_current_three_configs_change_only_age_composition_defaults(self):
        from afl_vlm.config import load_config, validate_config

        root = Path(__file__).resolve().parents[1]
        for setting in (2, 5, 10):
            config = load_config(
                root / f"configs/experiments/llava/afvlm_cm/{setting}clients/ours.yaml"
            )
            validate_config(config)
            params = config["method"]["params"]
            self.assertEqual(params["history_age_mode"], "selected_layers_sum")
            for key, value in {"history_age_boost": 2.0, "history_age_time_scale": 18.0,
                               "history_beta": 0.9, "history_strength": 1.0, "history_lambda": 1.0,
                               "history_update_mode": "first_observation_ema",
                               "base_precision": 0.0,
                               "alpha_max": None, "gamma": 1.0, "frequency_window_size": 18,
                               "sensitivity_uniform_mix": 0.5}.items():
                self.assertEqual(params[key], value)


if __name__ == "__main__":
    unittest.main()
