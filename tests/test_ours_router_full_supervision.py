"""Gate algebra/lifecycle/collective checks; no VLM or distributed training job."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from afl_vlm.methods.ours_router.background import attach_background
from afl_vlm.methods.ours_router.local import LocalRouter
from afl_vlm.methods.ours_router.router import component_features
from afl_vlm.methods.ours_router.supervision import gate_labels
from test_ours_router import FactorRegistry, context


class FullSupervisionChecks(unittest.TestCase):
    def test_fixed_gates_preserve_forward_and_main_gradients_and_label_every_rank(self):
        from torch.utils.checkpoint import checkpoint

        for reentrant in (False, True):
            torch.manual_seed(24)
            aa = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
            bb = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
            selected, rest = [0, 3], [1, 2, 4]
            a = torch.nn.Linear(4, 2, bias=False).double()
            b = torch.nn.Linear(2, 3, bias=False).double()
            with torch.no_grad():
                a.weight.copy_(aa[selected])
                b.weight.copy_(bb[:, selected])
            h = torch.ones(5, dtype=torch.float64, requires_grad=True)
            order = torch.tensor(selected + rest)
            restore = attach_background(a, b, aa[rest], bb[:, rest], h, order)
            parameters = list(a.parameters()) + list(b.parameters())
            self.assertFalse(any(p is h for p in parameters))
            x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
            xr = x.detach().clone().requires_grad_()
            dropout = torch.nn.Dropout(0.2)
            rng = torch.get_rng_state()

            def forward(value, a=a, b=b, dropout=dropout):
                return 2 * b(a(dropout(value)))

            actual = checkpoint(forward, x, use_reentrant=True) if reentrant else forward(x)
            torch.set_rng_state(rng)
            expected = 2 * (dropout(xr) @ aa.T @ bb.T)
            actual.square().sum().backward()
            expected.square().sum().backward()
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(x.grad, xr.grad)
            torch.testing.assert_close(a.weight.grad, aa.grad[selected])
            torch.testing.assert_close(b.weight.grad, bb.grad[:, selected])
            torch.testing.assert_close(h.grad, (bb * bb.grad).sum(0))
            torch.testing.assert_close(h.grad, (aa * aa.grad).sum(1))
            self.assertTrue(bool((h.grad[rest] != 0).all()))
            self.assertTrue(torch.equal(h, torch.ones_like(h)))
            restore()

    def test_gate_gradients_accumulate_then_mean_across_replicas_and_clear(self):
        h = torch.ones(3, requires_grad=True)
        for part in (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 4.0, 5.0])):
            (h * part / 2).sum().backward()
        local = h.grad.clone()
        remote = torch.tensor([6.0, 8.0, 10.0])

        def reduce(packed):
            packed[:3].add_(remote)
            packed[3:].add_(1)

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "all_reduce", side_effect=reduce) as collective,
        ):
            labels = gate_labels({"m": h})
        torch.testing.assert_close(labels["m"], -(local + remote) / 2)
        self.assertEqual(collective.call_count, 1)
        self.assertIsNone(h.grad)
        self.assertFalse(labels["m"].requires_grad)
        self.assertTrue(torch.equal(h, torch.ones_like(h)))

    def test_missing_or_nonfinite_gate_gradients_fail_and_clear(self):
        h = torch.ones(2, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "missing task gradients"):
            gate_labels({"m": h})
        h.grad = torch.full_like(h, float("nan"))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            gate_labels({"m": h})
        self.assertIsNone(h.grad)

    def test_controller_uses_full_pre_step_features_global_indices_and_cleans_up(self):
        fixture = FactorRegistry()
        with torch.no_grad():
            for name, p in fixture.named_federated_parameters():
                if "lora_B" in name:
                    p.fill_(0.3)
        ctl = LocalRouter(
            fixture, {"local_frozen_background": True, "router_supervision": "full_candidates"}
        )
        for chosen in ([0, 2], [0, 1, 2]):
            ctl.prepare(
                context(), {"indices": {m: chosen for m in ctl.full_metadata}, "source": {}}
            )
            self.assertEqual(set(ctl.observation_gates), set(ctl.full_metadata))
            for module, adapter, _, _ in ctl.slots.values():
                a, b = module.lora_A[adapter], module.lora_B[adapter]
                with torch.no_grad():
                    a.weight.add_(0.1)
                b(a(torch.ones(2, 2))).square().mean().backward()
            before = fixture.snapshot_trainable()
            grads = {n: p.grad.clone() for n, p in fixture.named_federated_parameters()}
            ctl.observe(1)
            for name, item in ctl.full_metadata.items():
                features = ctl.pending[name][1]
                torch.testing.assert_close(
                    features,
                    component_features(before[item["a_name"]], before[item["b_name"]], 1e-12),
                )
                self.assertEqual(ctl.pending[name][2].numel(), item["rank"])
                self.assertFalse(features.requires_grad)
                self.assertIsNone(ctl.observation_gates[name].grad)
            saved = {m: v[1].clone() for m, v in ctl.pending.items()}
            with torch.no_grad():
                for _, p in fixture.named_federated_parameters():
                    p.add_(0.2)
            for name, p in fixture.named_federated_parameters():
                torch.testing.assert_close(p.grad, grads[name])
            for m in saved:
                self.assertTrue(torch.equal(saved[m], ctl.pending[m][1]))
            ctl.loss_sum, ctl.loss_count = torch.tensor(1.0), 1
            ctl.step(1)
            ranking = ctl.trajectory[-1]["router_ranking"]
            self.assertEqual(ranking["phase"], "formal")
            self.assertEqual(ranking["scope"], "full_candidate_components")
            for detail in ranking["modules"].values():
                self.assertEqual(detail["component_indices"], [0, 1, 2])
                self.assertEqual(detail["observed_components"], 3)
            self.assertEqual(ctl.pending, {})
            ctl.restore()
            self.assertEqual(ctl.observation_gates, {})
            self.assertEqual(ctl.background_restorers, [])

    def test_new_option_requires_full_context_and_old_configs_remain_selected(self):
        from afl_vlm.config import load_config
        from afl_vlm.methods.ours_router.method import OursRouter
        from test_ours_router import ROOT

        for setting in (2, 5, 10):
            cfg = load_config(
                ROOT / f"configs/experiments/llava/afvlm_cm/{setting}clients/ours_router.yaml"
            )
            self.assertEqual(cfg["method"]["params"]["router_supervision"], "full_candidates")
            OursRouter(cfg["method"]["params"]).validate_runtime()
        OursRouter({"rank_aggregation_mode": "conflict_router"}).validate_runtime()
        for params in (
            {"router_supervision": "unknown"},
            {"router_supervision": "full_candidates"},
            {"rank_aggregation_mode": "conflict_router", "router_supervision": "full_candidates"},
        ):
            with self.assertRaises(ValueError):
                OursRouter(params)
        OursRouter(
            {
                "rank_aggregation_mode": "conflict_router",
                "local_frozen_background": True,
                "router_supervision": "full_candidates",
            }
        ).validate_runtime()
