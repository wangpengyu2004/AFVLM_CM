"""One-shot protocol/algebra/config checks only; no model or training is used."""

from __future__ import annotations

import copy
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from afl_vlm.config import load_config, validate_config
from afl_vlm.execution import resolve_runtime_backend
from afl_vlm.federation.server import FederatedServer
from afl_vlm.federation.types import ClientContext, ClientSpec, ServerContext, Update
from afl_vlm.methods.registry import create_method, method_names, profile_method_names
from afl_vlm.models.base import subtract
from afl_vlm.scheduling.train_plan import build_train_plan

A = "layers.0.q_proj.lora_A.default.weight"
B = "layers.0.q_proj.lora_B.default.weight"
ROOT = Path(__file__).resolve().parents[1]


def state(amount: float = 0.0):
    return {
        A: torch.eye(2, 4),
        B: torch.tensor(
            [
                [amount, 0.0],
                [0.0, amount / 2],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
    }


def spec(cid: str, task: str):
    return ClientSpec(cid, task, "dataset", 128, 1.0, 0.2, 0.0, 1.0)


def upload(cid: str, task: str, amount: float, arrival: float = 1):
    origin, local = state(), state(amount)
    return Update(
        f"{cid}-r0",
        cid,
        task,
        "dataset",
        128,
        0,
        0,
        arrival,
        origin,
        local,
        subtract(local, origin),
        42,
        "hash",
        [],
        20,
    )


class OneShotChecks(unittest.TestCase):
    def configured(self, params=None):
        method = create_method("opcm_oneshot", params)
        clients = [spec("cls/client_0", "cls"), spec("vqa/client_0", "vqa")]
        method.configure_server(state(), clients)
        return method, clients

    def test_common_origin_download_and_only_one_final_server_mutation(self):
        method, clients = self.configured()
        server = FederatedServer(None, method, 2, initial_state=state())
        for item in clients:
            ctx = ClientContext(item.id, item.task, item.dataset, 128, 0, 0, 1, 42)
            downloaded = method.prepare_download(state(99), ctx)
            for key in downloaded:
                torch.testing.assert_close(downloaded[key], state()[key])
        first = server.receive(upload(clients[0].id, "cls", 0.1))
        self.assertFalse(first[0].applied)
        self.assertEqual(server.version, 0)
        self.assertEqual(server.accepted_updates, 0)
        second = server.receive(upload(clients[1].id, "vqa", 0.2, 2))
        self.assertTrue(second[0].applied)
        self.assertEqual(server.version, 1)
        self.assertEqual(server.received_updates, 2)
        self.assertEqual(server.accepted_updates, 2)
        self.assertEqual(method.buffer_aggregations, 1)
        self.assertEqual(len(second[0].contributing_update_ids), 2)
        self.assertEqual(method.experts, {})
        self.assertEqual(server.finish(), [])

    def test_offline_merge_equals_existing_opcm_common_origin_algebra(self):
        method, _ = self.configured({"merge_order": "client_id", "merge_weight": 1.0})
        rows = [upload("cls/client_0", "cls", 0.1), upload("vqa/client_0", "vqa", 0.2)]
        saved = copy.deepcopy(rows)
        method.on_arrival(rows[1], ServerContext(state(), 0, 2))
        actual = method.on_arrival(rows[0], ServerContext(state(), 0, 2))[0]
        original = create_method(
            "opcm_lora",
            {
                "merge_weight": 1.0,
                "retain": 0.5,
                "eps": 1e-8,
                "staleness": {"type": "constant"},
            },
        )
        current = state()
        for row in rows:
            current = original.on_arrival(row, ServerContext(current, 0, 2))[0].new_state
        for key in current:
            torch.testing.assert_close(actual.new_state[key], current[key])
        for before, after in zip(saved, rows, strict=True):
            for key in before.local_state:
                torch.testing.assert_close(before.local_state[key], after.local_state[key])
        self.assertEqual(actual.metadata["merge_client_order"], [r.client_id for r in rows])
        self.assertEqual(actual.metadata["public_server_aggregations"], 1)

    def test_upload_order_does_not_change_seeded_merge_order_or_model(self):
        first, _ = self.configured({"merge_seed": 11})
        second, _ = self.configured({"merge_seed": 11})
        rows = [upload("cls/client_0", "cls", 0.1), upload("vqa/client_0", "vqa", 0.2)]
        initial_rng = random.getstate()
        first.on_arrival(rows[0], ServerContext(state(), 0, 2))
        left = first.on_arrival(rows[1], ServerContext(state(), 0, 2))[0]
        second.on_arrival(rows[1], ServerContext(state(), 0, 2))
        right = second.on_arrival(rows[0], ServerContext(state(), 0, 2))[0]
        self.assertEqual(left.metadata["merge_client_order"], right.metadata["merge_client_order"])
        for key in left.new_state:
            torch.testing.assert_close(left.new_state[key], right.new_state[key])
        self.assertEqual(random.getstate(), initial_rng)

    def test_arrival_merge_order_is_explicit_and_tie_broken_by_client_id(self):
        method, _ = self.configured({"merge_order": "arrival"})
        method.on_arrival(upload("cls/client_0", "cls", 0.1, 10), ServerContext(state(), 0, 2))
        final = method.on_arrival(
            upload("vqa/client_0", "vqa", 0.2, 1), ServerContext(state(), 0, 2)
        )[0]
        self.assertEqual(final.metadata["merge_client_order"], ["vqa/client_0", "cls/client_0"])

    def test_missing_client_duplicate_upload_and_extra_round_fail(self):
        method, _ = self.configured()
        row = upload("cls/client_0", "cls", 0.1)
        method.on_arrival(row, ServerContext(state(), 0, 2))
        with self.assertRaisesRegex(RuntimeError, "vqa/client_0"):
            method.on_finish(ServerContext(state(), 0, 2))
        with self.assertRaises(ValueError):
            method.on_arrival(row, ServerContext(state(), 0, 2))
        row = upload("vqa/client_0", "vqa", 0.2)
        row.local_round = 1
        with self.assertRaises(ValueError):
            method.on_arrival(row, ServerContext(state(), 0, 2))
        self.assertEqual(method.arrivals, 1)

    def test_unknown_client_wrong_origin_staleness_and_nonfinite_fail(self):
        for kind in ("unknown", "origin", "stale", "nan", "samples", "server_drift"):
            method, _ = self.configured()
            row = upload("cls/client_0", "cls", 0.1)
            current = state()
            if kind == "unknown":
                row.client_id = "unknown"
            elif kind == "origin":
                row.base_state[A] = row.base_state[A] + 0.5
            elif kind == "stale":
                row.base_version = 1
            elif kind == "nan":
                row.local_state[B][0, 0] = float("nan")
            elif kind == "samples":
                row.num_samples = 1
            else:
                current = state(0.5)
            with self.assertRaises(ValueError):
                method.on_arrival(row, ServerContext(current, 0, 2))
            self.assertEqual(method.experts, {})

    def test_pending_expert_is_detached_owned_copy_and_roundtrip_is_equivalent(self):
        method, _ = self.configured()
        row = upload("cls/client_0", "cls", 0.1)
        row.local_state[B].requires_grad_(True)
        method.on_arrival(row, ServerContext(state(), 0, 2))
        self.assertFalse(method.experts[row.client_id][B].requires_grad)
        with torch.no_grad():
            row.local_state[B].fill_(99)
        self.assertLess(float(method.experts[row.client_id][B].max()), 1)
        restored = create_method("opcm_oneshot")
        restored.load_state_dict(method.state_dict())
        next_row = upload("vqa/client_0", "vqa", 0.2)
        expected = method.on_arrival(next_row, ServerContext(state(), 0, 2))[0]
        actual = restored.on_arrival(next_row, ServerContext(state(), 0, 2))[0]
        for key in actual.new_state:
            torch.testing.assert_close(actual.new_state[key], expected.new_state[key])
        with self.assertRaises(ValueError):
            restored.on_arrival(next_row, ServerContext(state(), 0, 2))

    def test_plan_has_one_job_per_client_and_rejects_iterative_plan(self):
        method, clients = self.configured()
        plan = build_train_plan(clients, 1, 10, 4, 4, "synchronous")
        method.configure_schedule(plan)
        self.assertEqual(len(plan), len(clients))
        for bad in ([], plan + plan, build_train_plan(clients, 2, 1, 4, 4, "synchronous")):
            with self.assertRaises(ValueError):
                method.configure_schedule(bad)

    def test_parameters_configs_capabilities_and_standard_profiles_are_isolated(self):
        self.assertIn("opcm_oneshot", method_names())
        self.assertNotIn("opcm_oneshot", profile_method_names())
        self.assertEqual(set(profile_method_names()), set(method_names()) - {"opcm_oneshot"})
        for setting in (2, 5, 10):
            folder = ROOT / "configs/experiments/llava/afvlm_cm" / f"{setting}clients"
            cfg, ordinary = (
                load_config(folder / "opcm_oneshot.yaml"),
                load_config(folder / "opcm_lora.yaml"),
            )
            validate_config(cfg, check_paths=False)
            self.assertEqual(cfg["training"]["local_epochs"], 10)
            self.assertEqual(cfg["federation"]["rounds"], 1)
            self.assertEqual(ordinary["method"]["params"]["merge_weight"], 0.5)
            for section in ("model", "dataset", "runtime", "run"):
                self.assertEqual(cfg[section], ordinary[section])
            self.assertEqual(
                resolve_runtime_backend(cfg, create_method("opcm_oneshot")), "client_ddp"
            )
            for section, key, value in (
                ("federation", "rounds", 10),
                ("evaluation", "eval_every_client_updates", 10),
            ):
                bad = copy.deepcopy(cfg)
                bad[section][key] = value
                with self.assertRaises(ValueError):
                    validate_config(bad, check_paths=False)
        for params in (
            {"merge_order": "bad"},
            {"merge_seed": True},
            {"merge_weight": 0},
            {"retain": 2},
            {"eps": 0},
        ):
            with self.assertRaises(ValueError):
                create_method("opcm_oneshot", params)


class OneShotProfileChecks(unittest.TestCase):
    def test_reusing_system_metadata_never_reuses_multiround_plan(self):
        from tools.generate_oneshot_profiles import create_profile, validate_profile
        from tools.generate_system_profiles import TASK_DATASETS

        def metadata(root, setting, seed, costs):
            return {
                "seed": seed,
                "clients": [
                    {
                        "client_id": f"{task}/client_{i}",
                        "task": task,
                        "dataset": dataset,
                        "num_samples": 128,
                        "speed_factor": 1.3,
                        "network_delay": 0.7,
                        "initial_availability": 1.0,
                        "base_training_cost": 0.8,
                        "task_compute_factor": costs[task],
                    }
                    for task, dataset in TASK_DATASETS.items()
                    for i in range(setting)
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "configs", root / "configs")
            costs = {task: 1.0 for task in TASK_DATASETS}
            originals = {}
            for setting in (2, 5, 10):
                folder = root / "experiment_profiles/old/plans" / f"{setting}clients"
                folder.mkdir(parents=True)
                body = json.dumps(metadata(root, setting, 42, costs)).encode()
                (folder / "system_profile.json").write_bytes(body)
                # No input TrainPlan exists: only system attributes may be copied.
                originals[setting] = body
            with patch("tools.generate_oneshot_profiles.build", side_effect=metadata):
                path = create_profile(
                    root, "reuse", reference_world_size=8, system_profile_from="old"
                )
                validate_profile(root, "reuse")
                for setting, body in originals.items():
                    copied = path / f"plans/{setting}clients/system_profile.json"
                    self.assertEqual(copied.read_bytes(), body)
                old_system = root / "experiment_profiles/old/plans/2clients/system_profile.json"
                bad = json.loads(old_system.read_text())
                bad["clients"][0]["num_samples"] = 129
                old_system.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, "actual data"):
                    create_profile(
                        root, "mismatch", reference_world_size=8, system_profile_from="old"
                    )
                self.assertFalse((root / "one_shot_profiles/mismatch").exists())

    def test_standalone_generator_snapshot_validation_and_no_overwrite(self):
        from tools.generate_oneshot_profiles import create_profile, validate_profile

        tasks = ("cls", "caption", "vqa", "chart_vqa", "visual_reasoning", "grounding")

        def metadata(root, setting, seed, costs):
            return {
                "seed": seed,
                "clients": [
                    {
                        "client_id": f"{task}/client_{i}",
                        "task": task,
                        "dataset": "dataset",
                        "num_samples": 128,
                        "speed_factor": 1.0,
                        "network_delay": 0.2,
                        "initial_availability": 0,
                        "base_training_cost": 1.0,
                        "task_compute_factor": costs[task],
                    }
                    for task in tasks
                    for i in range(setting)
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "configs", root / "configs")
            with patch("tools.generate_oneshot_profiles.build", side_effect=metadata):
                path = create_profile(root, "e3", local_epochs=3, reference_world_size=8)
                result = validate_profile(root, "e3")
                self.assertEqual(result["configs"], 3)
                cfg = load_config(path / "configs/2clients/opcm_oneshot.yaml")
                self.assertEqual(cfg["training"]["local_epochs"], 3)
                self.assertEqual(cfg["training"]["reference_world_size"], 8)
                self.assertFalse((root / "experiment_profiles").exists())
                self.assertEqual(
                    len(json.loads((path / "plans/2clients/one_shot_train_plan.json").read_text())),
                    12,
                )
                with self.assertRaises(FileExistsError):
                    create_profile(root, "e3", reference_world_size=8)
                saved = path / "configs/2clients/opcm_oneshot.yaml"
                saved.write_bytes(saved.read_bytes() + b"\n# tampered\n")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    validate_profile(root, "e3")


if __name__ == "__main__":
    unittest.main()
