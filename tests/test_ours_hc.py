"""Equation, sparse protocol, isolation and config checks only; never train a model."""

from __future__ import annotations

import ast
import copy
import csv
import json
import math
import runpy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from afl_vlm.config import load_config, validate_config
from afl_vlm.federation.types import ClientContext, ClientSpec, ServerContext, Update
from afl_vlm.methods.ours_hc import payload
from afl_vlm.methods.ours_hc.local import phase_boundaries
from afl_vlm.methods.ours_hc.method import OursHC
from afl_vlm.methods.registry import create_method, method_names, profile_method_names

ROOT = Path(__file__).resolve().parents[1]
MODULE = "base_model.model.layers.0.self_attn.q_proj"
A, B = f"{MODULE}.lora_A.default.weight", f"{MODULE}.lora_B.default.weight"
BASE = {A: [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], B: [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]}
META = {
    MODULE: {
        "module_name": MODULE,
        "a_name": A,
        "b_name": B,
        "rank": 3,
        "in_features": 2,
        "out_features": 2,
        "scaling": 2.0,
    }
}
CLIENTS = [
    ClientSpec(f"{task}/client_0", task, task, 4, 1.0, 0.0, 0.0, 1.0) for task in ("vqa", "cls")
]


def context(round_number: int = 0, task: str = "vqa", version: int = 0) -> ClientContext:
    return ClientContext(f"{task}/client_0", task, task, 4, round_number, version, 1.0, 42)


def incoming(ctx: ClientContext, c: float = 2.0, gates: list | None = None) -> Update:
    gates = [0.5, 0.0, 0.75] if gates is None else gates
    local = {A: [[2.0, 1.0], [99.0, 99.0], [3.0, 2.0]], B: [[2.0, 99.0, 2.0], [1.0, 99.0, 3.0]]}
    return Update(
        f"{ctx.client_id}-r{ctx.local_round}",
        ctx.client_id,
        ctx.task,
        ctx.dataset,
        4,
        ctx.local_round,
        ctx.base_version,
        1.0,
        copy.deepcopy(BASE),
        local,
        {A: [], B: []},
        42,
        "ids",
        [],
        3,
        metadata={
            "lora_metadata": copy.deepcopy(META),
            "module_sensitivity": {MODULE: c},
            "sensitivity_transform": {},
            "hc_selection": {MODULE: [j for j, value in enumerate(gates) if value > 0]},
            "hc_gate_values": {MODULE: gates},
            "hc_local": {},
        },
    )


def prepared(
    method: OursHC, ctx: ClientContext, c: float = 2.0, gates: list | None = None
) -> Update:
    method.prepare_download(copy.deepcopy(BASE), ctx)
    update = method.pack_local_update(incoming(ctx, c, gates), ctx)
    return method.prepare_upload(update, ctx)


class SparseAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.method = OursHC()
        self.method.configure_server(copy.deepcopy(BASE), CLIENTS)

    def test_payload_is_selected_base_relative_deltas_with_gate_absorbed_only_in_b(self):
        item = prepared(self.method, context())
        self.assertEqual(item.base_state, {})
        self.assertEqual(item.local_state, {})
        self.assertNotIn("hc_gate_values", item.metadata)
        self.assertEqual(item.metadata["hc_selection"][MODULE], [0, 2])
        self.assertEqual(item.delta[A], [[1.0, 1.0], [2.0, 1.0]])
        self.assertEqual(item.delta[B], [[0.0, 0.5], [0.5, 1.25]])
        self.assertEqual(item.metadata["hc_communication"]["factor_elements"], 8)
        self.assertEqual(item.metadata["hc_communication"]["full_factor_elements"], 12)
        self.assertFalse(item.metadata["hc_communication"]["includes_gate_parameters"])

    def test_unselected_server_drift_does_not_affect_selected_functional_reliability(self):
        item = prepared(self.method, context())
        current = copy.deepcopy(BASE)
        current[A][1] = [500.0, 900.0]
        current[B][0][1], current[B][1][1] = 400.0, 700.0
        mutation = self.method.on_arrival(item, ServerContext(current, 10, 2))[0]
        diagnostic = mutation.metadata["hc_diagnostics"]
        self.assertEqual(diagnostic["functional_drift"], 0.0)
        self.assertEqual(diagnostic["reliability"], 1.0)
        alpha = 2 / (2 + 1 + 1e-12)
        self.assertAlmostEqual(diagnostic["module_alpha"][MODULE], alpha)
        self.assertEqual(mutation.new_state[A][1], current[A][1])
        self.assertEqual(mutation.new_state[B][0][1], current[B][0][1])
        self.assertEqual(mutation.new_state[B][1][1], current[B][1][1])
        self.assertAlmostEqual(mutation.new_state[A][0][0], 1 + alpha)
        self.assertEqual(BASE[A][0][0], 1.0)

    def test_functional_distance_matches_selected_dense_products(self):
        item = prepared(self.method, context())
        now = copy.deepcopy(BASE)
        now[B][0][0] += 1
        drift, energy = payload.functional_energies(BASE, now, item)
        self.assertAlmostEqual(drift, 8.0)  # C=2, scaling=2, one unit at selected (0,0).

        def effective(a, b):
            return [
                [
                    2 * sum(x * y for x, y in zip(row, column, strict=True))
                    for column in zip(*a, strict=True)
                ]
                for row in b
            ]

        indices = [0, 2]
        a0, b0 = payload.select(BASE[A], indices, "A"), payload.select(BASE[B], indices, "B")
        ak, bk = payload.add(a0, item.delta[A]), payload.add(b0, item.delta[B])
        f0, fk = effective(a0, b0), effective(ak, bk)
        expected = 2 * sum(
            (x - y) ** 2
            for row0, rowk in zip(f0, fk, strict=True)
            for x, y in zip(row0, rowk, strict=True)
        )
        self.assertAlmostEqual(energy, expected)

    def test_history_first_replacement_then_plain_ema_and_frequency_use_previous_arrivals(self):
        first = prepared(self.method, context(), c=2.0)
        mutation = self.method.on_arrival(first, ServerContext(copy.deepcopy(BASE), 0, 2))[0]
        d = mutation.metadata["hc_diagnostics"]
        self.assertEqual(d["omega_before"][MODULE], 1.0)
        self.assertEqual(self.method.history["vqa"][MODULE], 2.0)
        self.assertEqual(self.method.history["cls"][MODULE], 1.0)
        second = prepared(self.method, context(1, version=1), c=4.0)
        second_mutation = self.method.on_arrival(second, ServerContext(copy.deepcopy(BASE), 1, 2))[
            0
        ]
        d = second_mutation.metadata["hc_diagnostics"]
        self.assertAlmostEqual(d["frequency_weight"], math.sqrt(1.5 / 2))
        self.assertEqual(d["omega_before"][MODULE], 1.5)
        self.assertAlmostEqual(self.method.history["vqa"][MODULE], 2.2)
        self.assertEqual(self.method.history_counts["vqa"], 2)

    def test_empty_mask_rejects_without_server_version_history_or_frequency_changes(self):
        item = prepared(self.method, context(), gates=[0.0, 0.0, 0.0])
        before = copy.deepcopy(self.method.history)
        mutation = self.method.on_arrival(item, ServerContext(copy.deepcopy(BASE), 3, 2))[0]
        self.assertFalse(mutation.increment_version)
        self.assertEqual(mutation.applied_weight, 0.0)
        self.assertEqual(mutation.new_state, BASE)
        self.assertEqual(self.method.history, before)
        self.assertEqual(self.method.window, [])
        self.assertFalse(self.method.downloads)

    def test_inflight_snapshot_and_memory_roundtrip(self):
        item = prepared(self.method, context())
        saved = self.method.state_dict()
        restored = OursHC()
        restored.load_state_dict(saved)
        original = self.method.on_arrival(item, ServerContext(copy.deepcopy(BASE), 1, 2))[0]
        replayed = restored.on_arrival(item, ServerContext(copy.deepcopy(BASE), 1, 2))[0]
        self.assertEqual(original.new_state, replayed.new_state)
        self.assertEqual(self.method.history, restored.history)

    def test_invalid_indices_duplicate_job_and_wrong_base_version_rejected(self):
        item = prepared(self.method, context())
        bad = copy.deepcopy(item)
        bad.metadata["hc_selection"][MODULE] = [2, 0]
        with self.assertRaises(ValueError):
            self.method.on_arrival(bad, ServerContext(copy.deepcopy(BASE), 0, 2))
        bad = copy.deepcopy(item)
        bad.base_version = 1
        with self.assertRaises(ValueError):
            self.method.on_arrival(bad, ServerContext(copy.deepcopy(BASE), 1, 2))
        self.method.on_arrival(item, ServerContext(copy.deepcopy(BASE), 1, 2))
        with self.assertRaises(ValueError):
            self.method.on_arrival(item, ServerContext(copy.deepcopy(BASE), 1, 2))

    def test_non_finite_payload_does_not_modify_history_or_frequency(self):
        item = prepared(self.method, context())
        item.delta[A][0][0] = math.inf
        before = copy.deepcopy(self.method.history)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.method.on_arrival(item, ServerContext(copy.deepcopy(BASE), 0, 2))
        self.assertEqual(self.method.history, before)
        self.assertEqual(self.method.window, [])

    def test_diagnostic_export_uses_only_this_methods_numeric_records(self):
        export = runpy.run_path(str(ROOT / "tools/export_ours_hc_diagnostics.py"))["export"]
        item = prepared(self.method, context())
        mutation = self.method.on_arrival(item, ServerContext(copy.deepcopy(BASE), 0, 2))[0]
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "events.jsonl").write_text(
                json.dumps({"result_metadata": [mutation.metadata]}) + "\n", encoding="utf-8"
            )
            report = export(run, run / "analysis")
            self.assertEqual(report["updates"], 1)
            self.assertEqual(report["modules"], 1)
            with (run / "analysis/modules.csv").open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["selected_indices"], "[0, 2]")
            self.assertEqual(float(rows[0]["sensitivity"]), 2)
            self.assertEqual(rows[0]["client_id"], "vqa/client_0")


class ConfigurationAndIsolationTests(unittest.TestCase):
    def test_three_configs_keep_training_data_runtime_and_separate_method_output(self):
        for setting in (2, 5, 10):
            folder = ROOT / f"configs/experiments/llava/afvlm_cm/{setting}clients"
            cfg, previous = load_config(folder / "ours_hc.yaml"), load_config(folder / "ours.yaml")
            validate_config(cfg, check_paths=False)
            for section in ("training", "dataset", "runtime", "evaluation", "federation"):
                self.assertEqual(cfg[section], previous[section])
            self.assertEqual(cfg["model"]["lora"]["r"], 32)
            self.assertEqual(cfg["model"]["lora"]["alpha"], 64)
            self.assertEqual(cfg["method"]["name"], "ours_hc")
            self.assertNotEqual(cfg["output"]["directory"], previous["output"]["directory"])
        self.assertIn("ours_hc", profile_method_names())
        self.assertTrue(create_method("ours_hc").capabilities.supports_client_ddp)

    def test_pretransport_hook_is_identity_for_every_existing_method(self):
        item, ctx = incoming(context()), context()
        for name in method_names():
            if name != "ours_hc":
                self.assertIs(create_method(name).pack_local_update(item, ctx), item, name)

    def test_phase_partition_uses_existing_budget_and_keeps_joint_and_fixed_stages(self):
        for steps in (3, 7, 15, 48):
            warmup, boundary = phase_boundaries(steps, 1, 0.2)
            self.assertEqual(warmup + (boundary - warmup) + (steps - boundary), steps)
            self.assertGreater(boundary - warmup, 0)
            self.assertGreater(steps - boundary, 0)
        self.assertEqual(phase_boundaries(15, 1, 0.2), (1, 12))
        with self.assertRaises(ValueError):
            phase_boundaries(2, 1, 0.2)

    def test_only_instance_wrappers_use_nonreentrant_checkpoint_and_private_gate_filter(self):
        path = ROOT / "code/afl_vlm/methods/ours_hc/local.py"
        source = path.read_text("utf-8")
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "checkpoint"
        ]
        self.assertEqual(len(calls), 1)
        self.assertIn("use_reentrant", {kw.arg for kw in calls[0].keywords})
        self.assertIn('if "._hc_gate." not in name', source)
        self.assertIn("gate.reset()", source)
        self.assertIn('gate.phase = "off"', source)

    def test_standard_generator_includes_hc_and_preserves_reusable_snapshots(self):
        from tools.generate_system_profiles import TASK_DATASETS, create_profile

        # Configuration/Plan metadata only; no model, image or training is involved.
        def client_metadata(root, setting, seed, task_compute_factors):
            return {"clients": [
                {"client_id": f"{task}/client_{index}", "task": task,
                 "dataset": dataset, "num_samples": 1024, "speed_factor": 1.0,
                 "network_delay": 0.1, "initial_availability": 0.0,
                 "base_training_cost": 1.0, "task_compute_factor": task_compute_factors[task]}
                for task, dataset in TASK_DATASETS.items() for index in range(setting)
            ]}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(ROOT / "configs", root / "configs")
            with patch("tools.generate_system_profiles.build", side_effect=client_metadata), \
                    patch("tools.generate_system_profiles._source_commit", return_value=None):
                first = create_profile(root, "shared", 42, None, reference_world_size=8)
                before = {p.relative_to(first): p.read_bytes() for p in first.rglob("*")
                          if p.is_file()}
                second = create_profile(root, "reuse", 42, "shared", reference_world_size=8)
                with self.assertRaises(FileExistsError):
                    create_profile(root, "shared", 42, None, reference_world_size=8)
            self.assertEqual(before, {p.relative_to(first): p.read_bytes()
                                      for p in first.rglob("*") if p.is_file()})
            for directory in (first, second):
                self.assertEqual(len(list((directory / "configs").rglob("*.yaml"))),
                                 3 * len(profile_method_names()))
                for setting in (2, 5, 10):
                    configs = directory / "configs" / f"{setting}clients"
                    cfg = load_config(configs / "ours_hc.yaml")
                    old = load_config(configs / "ours.yaml")
                    for section in ("training", "dataset", "runtime", "evaluation", "federation"):
                        self.assertEqual(cfg[section], old[section])
                    self.assertEqual(cfg["training"]["reference_world_size"], 8)
                    self.assertEqual(cfg["model"]["lora"]["r"], 32)
                    self.assertEqual(old["model"]["lora"]["r"], 8)
                    self.assertTrue((root / cfg["federation"]["train_plan"]).is_file())
                    self.assertEqual(cfg["output"]["directory"],
                                     f"runs/llava/afvlm_cm/profiles/{directory.name}/"
                                     f"{setting}clients/ours_hc/seed42")
                    for filename in ("system_profile.json", "async_train_plan.json"):
                        relative = Path("plans") / f"{setting}clients" / filename
                        self.assertEqual((first / relative).read_bytes(),
                                         (second / relative).read_bytes())


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Tensor equation checks require torch, never a model")
class HardConcreteEquationTests(unittest.TestCase):
    def test_interior_gate_and_l0_probability_derivatives(self):
        from afl_vlm.methods.ours_hc.gates import active_probability, sample_gate

        theta = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        z = sample_gate(theta, torch.tensor([0.5]), 2 / 3, -0.1, 1.1)
        derivative = torch.autograd.grad(z.sum(), theta)[0]
        self.assertAlmostEqual(float(derivative[0]), 1.2 * 0.25 / (2 / 3))
        p = active_probability(theta, 2 / 3, -0.1, 1.1)
        derivative = torch.autograd.grad(p.sum(), theta)[0]
        self.assertTrue(torch.allclose(derivative, p.detach() * (1 - p.detach())))

    def test_private_gate_is_fp32_and_reset_drops_previous_client_gate_values(self):
        from afl_vlm.methods.ours_hc.gates import make_gate

        gate = make_gate(3, torch.empty(1, dtype=torch.float16), {})
        self.assertEqual(gate.log_alpha.dtype, torch.float32)
        with torch.no_grad():
            gate.log_alpha.fill_(-20)
        gate.freeze()
        self.assertEqual(gate.fixed_z.tolist(), [0.0, 0.0, 0.0])
        gate.reset()
        self.assertEqual(gate.phase, "warmup")
        self.assertEqual(gate.fixed_z.tolist(), [1.0, 1.0, 1.0])
        self.assertTrue(torch.allclose(gate.probability(), torch.full((3,), 0.9)))

    def test_probability_and_temperature_free_deterministic_equations(self):
        from afl_vlm.methods.ours_hc.gates import (
            active_probability,
            deterministic_gate,
            sample_gate,
        )

        theta = torch.tensor([-10.0, 0.0, 10.0], dtype=torch.float64)
        p = active_probability(theta, 2 / 3, -0.1, 1.1)
        self.assertTrue(torch.allclose(p, torch.sigmoid(theta - 2 / 3 * math.log(0.1 / 1.1))))
        z = deterministic_gate(theta, -0.1, 1.1)
        self.assertTrue(torch.allclose(z, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)))
        sampled = sample_gate(theta, torch.tensor([0.5, 0.5, 0.5]), 2 / 3, -0.1, 1.1)
        self.assertTrue(torch.allclose(sampled, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)))

    def test_b_absorption_preserves_deterministic_selected_function(self):
        a, b = torch.tensor(BASE[A]), torch.tensor(BASE[B])
        z = torch.tensor([0.5, 0.0, 0.75])
        indices = [0, 2]
        reduced_b = payload.fold_b(payload.select(b, indices, "B"), [0.5, 0.75])
        self.assertTrue(
            torch.allclose(reduced_b @ payload.select(a, indices, "A"), b @ torch.diag(z) @ a)
        )


if __name__ == "__main__":
    unittest.main()
