"""Non-VLM checks: spectra, detached router equations, sparse protocol and config isolation.

No checkpoint, image, LLaVA forward/backward or distributed process is loaded.
"""

from __future__ import annotations

import copy
import csv
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from afl_vlm.config import load_config, validate_config
from afl_vlm.execution import resolve_runtime_backend
from afl_vlm.federation.types import ClientContext, ClientSpec, ServerContext, Update
from afl_vlm.methods.ours_router.capacity import spectral_capacity
from afl_vlm.methods.ours_router.local import LocalRouter
from afl_vlm.methods.ours_router.method import OursRouter
from afl_vlm.methods.ours_router.router import PrivateRouter, component_features, component_targets
from afl_vlm.methods.registry import create_method, method_names, profile_method_names
from afl_vlm.models.base import clone_state

ROOT = Path(__file__).resolve().parents[1]


class SpectrumAndRouterEquations(unittest.TestCase):
    def test_qr_core_matches_dense_singular_spectrum_including_cross_components(self):
        generator = torch.Generator().manual_seed(7)
        a = torch.randn(4, 9, generator=generator, dtype=torch.float64)
        b = torch.randn(7, 4, generator=generator, dtype=torch.float64)
        result = spectral_capacity(a, b, 2, 0.95)
        expected = torch.linalg.svdvals(2 * b @ a).square()[:4]
        self.assertTrue(
            torch.allclose(
                torch.tensor(result["squared_singular_values"], dtype=torch.float64),
                expected,
                rtol=1e-6,
            )
        )
        self.assertGreaterEqual(result["retained_energy_fraction"], 0.95)
        self.assertEqual(
            result["rank"], int((expected.cumsum(0) < 0.95 * expected.sum()).sum()) + 1
        )

    def test_zero_spectrum_and_capacity_bound_are_explicit_not_fake_estimates(self):
        a, b = torch.eye(3), torch.zeros(3, 3)
        result = spectral_capacity(a, b, 2, 0.95)
        self.assertEqual(result["rank"], 1)
        self.assertFalse(result["energy_valid"])
        self.assertFalse(result["threshold_met"])
        bounded = spectral_capacity(a, torch.diag(torch.tensor([3.0, 2.0, 1.0])), 1, 1.0, maximum=1)
        self.assertEqual(bounded["rank"], 1)
        self.assertTrue(bounded["bounded_by_maximum"])
        self.assertFalse(bounded["threshold_met"])

    def test_features_and_signed_targets_use_same_pre_step_state_without_model_graph(self):
        a = torch.nn.Parameter(torch.tensor([[3.0, 4.0], [0.0, 1.0]]))
        b = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        features = component_features(a, b, 1e-12)
        labels = component_targets(b, torch.tensor([[1.0, -1.0], [2.0, -2.0]]))
        self.assertEqual(labels.tolist(), [-7.0, 10.0])
        self.assertEqual(tuple(features.shape), (2, 6))
        self.assertFalse(features.requires_grad)
        self.assertFalse(labels.requires_grad)
        before = features.clone()
        with torch.no_grad():
            a.add_(99)
            b.mul_(10)
        self.assertTrue(torch.equal(features, before))

    def test_private_initialization_preserves_rng_and_zero_labels_skip_learning(self):
        saved = torch.get_rng_state().clone()
        router = PrivateRouter([(2, 2)], torch.device("cpu"), 42, {})
        self.assertTrue(torch.equal(saved, torch.get_rng_state()))
        before = clone_state(router.heads.state_dict())
        report = router.learn({"m": ("d2_2", torch.ones(2, 6), torch.zeros(2))})
        self.assertEqual(report["router_valid_modules"], 0)
        self.assertEqual(router.steps, 0)
        for name, value in before.items():
            self.assertTrue(torch.equal(router.heads.state_dict()[name], value))

    def test_private_regression_updates_only_router_and_restores_optimizer_history(self):
        router = PrivateRouter([(2, 2)], torch.device("cpu"), 42, {})
        a, b = torch.nn.Parameter(torch.ones(2, 2)), torch.nn.Parameter(torch.ones(2, 2))
        a.grad, b.grad = torch.ones_like(a), torch.full_like(b, 2)
        saved_a, saved_b = a.grad.clone(), b.grad.clone()
        features = component_features(a, b, 1e-12)
        labels = component_targets(b, b.grad)
        router.learn({"m": ("d2_2", features, labels)})
        self.assertEqual(router.steps, 1)
        self.assertTrue(torch.equal(a.grad, saved_a))
        self.assertTrue(torch.equal(b.grad, saved_b))
        restored = PrivateRouter([(2, 2)], torch.device("cpu"), 99, {})
        restored.load_state_dict(copy.deepcopy(router.state_dict()))
        self.assertTrue(torch.equal(router.predict(a, b), restored.predict(a, b)))
        self.assertEqual(restored.steps, 1)


class FactorRegistry:
    """Parameter-name/shape fixture only; deliberately has no VLM forward/train method."""

    def __init__(self):
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
        for layer in self.model.layers:
            layer.self_attn = torch.nn.Module()
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                module = torch.nn.Module()
                module.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 3, bias=False)})
                module.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 2, bias=False)})
                module.scaling, module.r = {"default": 2.0}, {"default": 3}
                with torch.no_grad():
                    module.lora_A["default"].weight.copy_(
                        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
                    )
                    module.lora_B["default"].weight.zero_()
                setattr(layer.self_attn, projection, module)
        self._ddp_enabled = False

    def _network(self):
        return self.model

    def named_federated_parameters(self):
        return list(self.model.named_parameters())

    def snapshot_trainable(self):
        return {name: value.detach().clone() for name, value in self.named_federated_parameters()}

    def load_trainable(self, state):
        with torch.no_grad():
            for name, value in self.named_federated_parameters():
                value.copy_(state[name])


def context(round_number=0, task="vqa", version=0):
    return ClientContext(f"{task}/client_0", task, task, 128, round_number, version, 1.0, 42)


class CompactFactorAndProtocol(unittest.TestCase):
    def setUp(self):
        self.model = FactorRegistry()
        self.controller = LocalRouter(self.model, {})
        self.base = self.model.snapshot_trainable()
        self.metadata = self.controller.full_metadata
        self.clients = [
            ClientSpec(f"{task}/client_0", task, task, 128, 1, 0, 0, 1) for task in ("vqa", "cls")
        ]

    def packet(self, method, ctx, changed=True):
        local = clone_state(self.base)
        selection, c = {}, {}
        for name, item in self.metadata.items():
            selection[name] = [0, 2]
            c[name] = 1.5 if ".0." in name else 0.5
            if changed:
                local[item["b_name"]][:, [0, 2]] += 1
        method.downloads[ctx.client_id, ctx.local_round] = (
            ctx.base_version,
            clone_state(self.base),
        )
        update = Update(
            f"{ctx.client_id}-r{ctx.local_round}",
            ctx.client_id,
            ctx.task,
            ctx.dataset,
            128,
            ctx.local_round,
            ctx.base_version,
            ctx.arrival_time,
            clone_state(self.base),
            local,
            {},
            42,
            "ids",
            [],
            4,
            metadata={
                "lora_metadata": copy.deepcopy(self.metadata),
                "hc_selection": selection,
                "module_sensitivity": c,
                "sensitivity_valid": True,
                "sensitivity_transform": {},
                "sensitivity_observations": {},
                "hc_local": {
                    "trajectory": [],
                    "selection_source": {name: "uniform_cold_start" for name in self.metadata},
                },
            },
        )
        return method.pack_local_update(update, ctx)

    def test_compact_parameters_and_snapshots_preserve_scaling_and_unselected_positions(self):
        choice = {name: [0, 2] for name in self.metadata}
        self.controller.prepare(context(), {"indices": choice, "source": {}})
        self.assertTrue(
            all(
                value.shape[0 if ".lora_A." in name else 1] == 2
                for name, value in self.model.named_federated_parameters()
            )
        )
        with torch.no_grad():
            for _, b, _ in self.controller.tracker.modules.values():
                b.add_(2)
        full = self.model.snapshot_trainable()
        for item in self.metadata.values():
            self.assertTrue(
                torch.equal(full[item["b_name"]][:, 1], self.base[item["b_name"]][:, 1])
            )
            self.assertEqual(full[item["a_name"]].shape[0], 3)
        self.controller.restore()
        for module, adapter, _, _ in self.controller.slots.values():
            self.assertEqual(module.scaling[adapter], 2)
            self.assertEqual(module.r[adapter], 3)
        for name, value in self.model.snapshot_trainable().items():
            self.assertTrue(torch.equal(value, self.base[name]))

    def test_capacity_and_router_cache_persist_but_each_job_selection_is_frozen(self):
        self.controller.capacities[context().client_id] = {
            name: {"rank": 2} for name in self.metadata
        }
        selection = self.controller.select(self.base, context())
        self.assertTrue(all(len(indices) == 2 for indices in selection["indices"].values()))
        router = self.controller.router(context().client_id, 999)
        self.assertIs(router, self.controller.router(context().client_id, 1000))
        self.controller.prepare(context(), selection)
        with torch.no_grad():
            for value in router.heads.parameters():
                value.add_(99)
        self.assertEqual(self.controller.choice, selection["indices"])
        self.controller.restore()

    def test_different_module_capacities_produce_different_physical_parameter_shapes(self):
        choice = {name: ([0] if ".0." in name else [0, 2]) for name in self.metadata}
        self.controller.prepare(context(), {"indices": choice, "source": {}})
        for name, (a, b, item) in self.controller.tracker.modules.items():
            self.assertEqual(a.shape[0], len(choice[name]))
            self.assertEqual(b.shape[1], len(choice[name]))
            self.assertEqual(item.scaling, 2.0)
        self.controller.restore()

    def test_virtual_download_selection_survives_later_global_and_router_changes(self):
        ctx = context()
        self.controller.capacities[ctx.client_id] = {m: {"rank": 2} for m in self.metadata}
        method = OursRouter({"history_age_enabled": False})
        method.controller = self.controller
        method.configure_server(self.base, self.clients)
        downloaded = method.prepare_download(clone_state(self.base), ctx)
        state = method.client_runtime_state(ctx)
        saved = copy.deepcopy(state)
        # Physical work can be delayed. Later server weights and private scores
        # cannot retroactively replace the already-dispatched support or base.
        with torch.no_grad():
            for value in self.model.model.parameters():
                value.add_(100)
            for value in self.controller.router(ctx.client_id, 42).heads.parameters():
                value.mul_(0)
        local = OursRouter({"history_age_enabled": False})
        local.load_client_runtime_state(state, ctx)
        self.assertEqual(local.local_selection, saved["selection"])
        for name, value in downloaded.items():
            self.assertTrue(torch.equal(value, self.base[name]))

    def test_sparse_upload_and_selected_drift_ignore_unselected_current_server_changes(self):
        method = OursRouter({"history_age_enabled": False})
        method.configure_server(self.base, self.clients)
        update = self.packet(method, context())
        self.assertFalse(update.base_state or update.local_state)
        self.assertNotIn("hc_gate_values", update.metadata)
        now = clone_state(self.base)
        for item in self.metadata.values():
            now[item["b_name"]][:, 1] += 100
        result = method.on_arrival(update, ServerContext(now, 7, 2))[0]
        report = result.metadata["router_diagnostics"]
        self.assertEqual(report["reliability"], 1)
        self.assertEqual(report["version_staleness"], 7)
        for item in self.metadata.values():
            self.assertTrue(
                torch.equal(result.new_state[item["b_name"]][:, 1], now[item["b_name"]][:, 1])
            )
        self.assertNotIn("heads", str(update.metadata))

    def test_task_end_protection_uses_last_ema_and_previous_success_count(self):
        method = OursRouter()
        method.configure_server(self.base, self.clients)
        method.configure_schedule(
            [
                SimpleNamespace(client_id=f"{task}/client_0", task=task, local_round=r)
                for task, r in (("vqa", 0), ("cls", 0), ("cls", 1), ("cls", 2))
            ]
        )
        first = method.on_arrival(self.packet(method, context()), ServerContext(self.base, 0, 2))[0]
        snapshot = first.metadata["router_diagnostics"]["finished_task_snapshot"]
        self.assertEqual(snapshot["selected_layers"], [0])
        self.assertEqual(snapshot["finished_at"], 1)
        second = method.on_arrival(
            self.packet(method, context(task="cls")), ServerContext(self.base, 1, 2)
        )[0]
        self.assertTrue(
            all(x == 0 for x in second.metadata["router_diagnostics"]["omega_extra"].values())
        )
        third = method.on_arrival(
            self.packet(method, context(1, "cls")), ServerContext(self.base, 2, 2)
        )[0]
        extra = third.metadata["router_diagnostics"]["omega_extra"]
        self.assertTrue(
            all(value > 0 if ".0." in name else value == 0 for name, value in extra.items())
        )
        restored = OursRouter()
        restored.load_state_dict(method.state_dict())
        self.assertEqual(restored.finished, method.finished)
        self.assertEqual(restored.accepted_count, 3)

    def test_rejected_zero_update_does_not_change_history_window_or_task_completion(self):
        method = OursRouter({"history_age_enabled": False})
        method.configure_server(self.base, self.clients)
        before = copy.deepcopy(method.history)
        result = method.on_arrival(
            self.packet(method, context(), changed=False), ServerContext(self.base, 0, 2)
        )[0]
        self.assertFalse(result.increment_version)
        self.assertEqual(method.history, before)
        self.assertEqual(method.window, [])
        self.assertFalse(method.finished or method.resolved)

    def test_numeric_export_preserves_per_module_ranks_and_history_without_weights(self):
        method = OursRouter({"history_age_enabled": False})
        method.configure_server(self.base, self.clients)
        result = method.on_arrival(self.packet(method, context()), ServerContext(self.base, 0, 2))[
            0
        ]
        export = runpy.run_path(str(ROOT / "tools/export_ours_router_diagnostics.py"))["export"]
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "events.jsonl").write_text(
                json.dumps({"result_metadata": [result.metadata]}) + "\n", encoding="utf-8"
            )
            count = export(run, run / "diagnostics")
            self.assertEqual(count["updates"], 1)
            self.assertEqual(count["modules"], len(self.metadata))
            with (run / "diagnostics/modules.csv").open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertTrue(all(row["selected_rank"] == "2" for row in rows))
            self.assertTrue(all(row["selected_indices"] == "[0, 2]" for row in rows))
            self.assertTrue(all(row["client_id"] == context().client_id for row in rows))
            with self.assertRaises(FileExistsError):
                export(run, run / "diagnostics")


class WorkflowIsolation(unittest.TestCase):
    def test_all_three_templates_use_standard_generation_and_ddp_selection(self):
        self.assertIn("ours_router", profile_method_names())
        for setting in (2, 5, 10):
            config = load_config(
                ROOT / f"configs/experiments/llava/afvlm_cm/{setting}clients/ours_router.yaml"
            )
            old = load_config(
                ROOT / f"configs/experiments/llava/afvlm_cm/{setting}clients/ours.yaml"
            )
            validate_config(config, check_paths=False)
            for section in ("training", "dataset", "evaluation", "federation", "runtime"):
                self.assertEqual(config[section], old[section])
            self.assertEqual(config["model"]["lora"]["r"], 32)
            self.assertEqual(resolve_runtime_backend(config, OursRouter()), "client_ddp")

    def test_calibration_is_opt_in_and_existing_methods_keep_noop_lifecycle(self):
        for name in method_names():
            method = create_method(name)
            if name != "ours_router":
                self.assertFalse(method.capabilities.requires_client_calibration)
                self.assertEqual(method.calibrate_clients(None, {}, {}, None, None), {})


if __name__ == "__main__":
    unittest.main()
