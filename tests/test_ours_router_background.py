"""Full-context forward and absolute drift regression checks, no VLM training."""

import math
import unittest
from unittest.mock import patch

import test_ours_router_conflict as conflict_tests
import torch
from afl_vlm.federation.types import ServerContext
from afl_vlm.methods.ours_router.background import attach_background
from afl_vlm.methods.ours_router.conflict_server import full_functional_drift
from afl_vlm.methods.ours_router.local import LocalRouter
from test_ours_router import FactorRegistry, context


class BackgroundChecks(unittest.TestCase):
    def test_background_keeps_correct_gradients_under_reentrant_checkpoint(self):
        from torch.utils.checkpoint import checkpoint

        a = torch.nn.Linear(4, 2, bias=False).double()
        b = torch.nn.Linear(2, 3, bias=False).double()
        fa, fb = torch.ones(1, 4, dtype=torch.float64), torch.ones(3, 1, dtype=torch.float64)
        original_a, original_b = a.weight.detach().clone(), b.weight.detach().clone()
        restore = attach_background(a, b, fa, fb)
        x = torch.ones(2, 4, dtype=torch.float64, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        ar, br = original_a.clone().requires_grad_(), original_b.clone().requires_grad_()
        actual = checkpoint(lambda value: b(a(value)), x, use_reentrant=True)
        expected = xr @ ar.T @ br.T + xr @ fa.T @ fb.T
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x.grad, xr.grad)
        torch.testing.assert_close(a.weight.grad, ar.grad)
        torch.testing.assert_close(b.weight.grad, br.grad)
        self.assertIsNone(fa.grad)
        self.assertIsNone(fb.grad)
        restore()

    def test_partial_background_setup_failure_restores_every_module(self):
        fixture = FactorRegistry()
        ctl = LocalRouter(fixture, {"local_frozen_background": True})
        initial = fixture.snapshot_trainable()
        chosen = {m: [0] for m in ctl.full_metadata}
        calls = 0

        def failing_attach(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("installation failed")
            return attach_background(*args)

        with patch("afl_vlm.methods.ours_router.background.attach_background", failing_attach):
            with self.assertRaisesRegex(RuntimeError, "installation failed"):
                ctl.prepare(context(), {"indices": chosen, "source": {}})
        self.assertFalse(ctl.active)
        self.assertEqual(ctl.background_restorers, [])
        for module, adapter, _, _ in ctl.slots.values():
            self.assertEqual(module.lora_A[adapter](torch.ones(1, 2)).shape[-1], 3)
        for name, value in fixture.snapshot_trainable().items():
            torch.testing.assert_close(value, initial[name])

    def test_tracker_setup_failure_also_restores_background(self):
        ctl = LocalRouter(FactorRegistry(), {"local_frozen_background": True})
        chosen = {m: [0] for m in ctl.full_metadata}
        with patch("afl_vlm.methods.ours_router.local.ModuleSensitivityTracker",
                   side_effect=RuntimeError("tracker failed")):
            with self.assertRaisesRegex(RuntimeError, "tracker failed"):
                ctl.prepare(context(), {"indices": chosen, "source": {}})
        self.assertFalse(ctl.active)
        self.assertEqual(ctl.background_restorers, [])
        for module, adapter, _, _ in ctl.slots.values():
            self.assertEqual(module.lora_A[adapter](torch.ones(1, 2)).shape[-1], 3)

    def test_matches_full_forward_and_input_and_selected_gradients_with_dropout(self):
        torch.manual_seed(18)
        full_a = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        full_b = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
        selected, rest = [0, 3], [1, 2, 4]
        a, b = (
            torch.nn.Linear(4, 2, bias=False).double(),
            torch.nn.Linear(2, 3, bias=False).double(),
        )
        a.weight.data.copy_(full_a.detach()[selected])
        b.weight.data.copy_(full_b.detach()[:, selected])
        fa, fb = full_a.detach()[rest].clone(), full_b.detach()[:, rest].clone()
        restore = attach_background(a, b, fa, fb)
        # Mutating caller-owned sources must not change the download background.
        fa.add_(100)
        fb.add_(100)
        x = torch.randn(2, 7, 4, dtype=torch.float64, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        dropout = torch.nn.Dropout(0.2)
        rng = torch.get_rng_state()
        actual = 2 * b(a(dropout(x)))
        torch.set_rng_state(rng)
        expected = 2 * (dropout(xr) @ full_a.T @ full_b.T)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, xr.grad)
        torch.testing.assert_close(a.weight.grad, full_a.grad[selected])
        torch.testing.assert_close(b.weight.grad, full_b.grad[:, selected])
        self.assertEqual(sum(p.numel() for p in list(a.parameters()) + list(b.parameters())), 14)
        restore()
        self.assertEqual(a(x.detach()).shape[-1], 2)

    def test_controller_snapshot_upload_support_and_restore_for_repeated_jobs(self):
        fixture = FactorRegistry()
        with torch.no_grad():
            for n, p in fixture.named_federated_parameters():
                if "lora_B" in n:
                    p.fill_(0.3)
        ctl = LocalRouter(fixture, {"local_frozen_background": True})
        full = fixture.snapshot_trainable()
        for chosen in ([0, 2], [0, 1, 2], [1]):
            selection = {"indices": {m: chosen for m in ctl.full_metadata}, "source": {}}
            ctl.prepare(context(), selection)
            for name, (module, adapter, _, _) in ctl.slots.items():
                item = ctl.full_metadata[name]
                x = torch.tensor([[0.5, -0.2]], requires_grad=True)
                a, b = module.lora_A[adapter], module.lora_B[adapter]
                torch.testing.assert_close(
                    b(a(x)), x @ full[item["a_name"]].T @ full[item["b_name"]].T
                )
                self.assertEqual(a.weight.shape[0], len(chosen))
                self.assertEqual(b.weight.shape[1], len(chosen))
            snapshot = fixture.snapshot_trainable()
            for name in full:
                torch.testing.assert_close(snapshot[name], full[name])
            ctl.restore()
            self.assertEqual(ctl.background_restorers, [])
            for module, adapter, _, _ in ctl.slots.values():
                self.assertEqual(module.lora_A[adapter](torch.ones(1, 2)).shape[-1], 3)


class AbsoluteDriftChecks(unittest.TestCase):
    def test_absolute_diagnostics_export_keeps_relative_field_empty(self):
        import csv
        import json
        import runpy
        import tempfile
        from pathlib import Path

        from test_ours_router import ROOT

        fixture = conflict_tests.ConflictServerChecks()
        fixture.setUp()
        method, server = fixture.setup_method()
        method.params["staleness_mode"] = "full_functional_absolute"
        packet = fixture.packet(method)
        packet.metadata["hc_local"]["local_frozen_background"] = True
        metadata = server.receive(packet)[0].metadata
        export = runpy.run_path(str(ROOT / "tools/export_ours_router_diagnostics.py"))["export"]
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "events.jsonl").write_text(
                json.dumps({"result_metadata": [metadata]}) + "\n", encoding="utf-8"
            )
            export(run, run / "csv")
            with (run / "csv/updates.csv").open(encoding="utf-8-sig", newline="") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["staleness_mode"], "full_functional_absolute")
            self.assertEqual(row["relative_staleness"], "")
            self.assertEqual(float(row["absolute_staleness"]), 0.0)
            self.assertEqual(row["local_frozen_background"], "True")

    def test_new_defaults_and_legacy_missing_options_are_explicit(self):
        from afl_vlm.config import load_config
        from afl_vlm.methods.ours_router.method import OursRouter
        from test_ours_router import ROOT

        for setting in (2, 5, 10):
            cfg = load_config(
                ROOT / f"configs/experiments/llava/afvlm_cm/{setting}clients/ours_router.yaml"
            )
            self.assertTrue(cfg["method"]["params"]["local_frozen_background"])
            self.assertEqual(cfg["method"]["params"]["staleness_mode"], "full_functional_absolute")
            OursRouter(cfg["method"]["params"]).validate_runtime()
        OursRouter({"rank_aggregation_mode": "conflict_router"}).validate_runtime()
        for params in (
            {"local_frozen_background": "false"},
            {"staleness_mode": "unknown"},
            {"local_frozen_background": True},
            {"staleness_mode": "full_functional_absolute"},
        ):
            with self.assertRaises(ValueError):
                OursRouter(params)

    def test_full_drift_includes_unselected_changes_and_matches_dense(self):
        torch.manual_seed(6)
        base = {
            "a": torch.randn(3, 4, dtype=torch.float64),
            "b": torch.randn(5, 3, dtype=torch.float64),
        }
        now = {k: v.clone() for k, v in base.items()}
        now["b"][:, 1] += 0.7
        modules = {"m": dict(a_name="a", b_name="b", scaling=2.0)}
        expected = (2 * (now["b"] @ now["a"] - base["b"] @ base["a"])).square().sum()
        self.assertAlmostEqual(
            full_functional_drift(base, now, modules), expected.item(), places=10
        )

    def test_rho_independent_of_upload_energy_and_sensitivity(self):
        fixture = conflict_tests.ConflictServerChecks()
        fixture.setUp()
        results = []
        for magnitude, c in ((0.1, 1.0), (2.0, 100.0)):
            method, _ = fixture.setup_method()
            method.params["staleness_mode"] = "full_functional_absolute"
            update = fixture.packet(method, delta=magnitude, selected=[0, 2])
            for name in method.modules:
                update.metadata["module_sensitivity"][name] = c
            now = {k: v.clone() for k, v in fixture.base.items()}
            for item in method.modules.values():
                now[item["b_name"]][:, 1] += 0.2
            result = method.on_arrival(update, ServerContext(now, 1, 2))[0]
            row = result.metadata["router_diagnostics"]
            self.assertGreater(row["functional_drift"], 0)
            self.assertEqual(row["selected_functional_drift"], 0)
            self.assertIsNone(row["relative_staleness"])
            self.assertAlmostEqual(
                row["reliability"], math.exp(-math.sqrt(row["functional_drift"]))
            )
            for item in method.modules.values():
                torch.testing.assert_close(
                    result.new_state[item["b_name"]][:, 1], now[item["b_name"]][:, 1]
                )
            results.append(row["reliability"])
        self.assertEqual(results[0], results[1])
