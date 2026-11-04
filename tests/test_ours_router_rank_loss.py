"""Private ranking objective equations and gradient isolation; no VLM training."""

from __future__ import annotations

import copy
import math
import unittest
from unittest.mock import patch

import torch
from afl_vlm.methods.ours_router.losses import pairwise_rank_loss
from afl_vlm.methods.ours_router.method import OursRouter
from afl_vlm.methods.ours_router.router import PrivateRouter


class RankLossChecks(unittest.TestCase):
    def test_formula_pair_orientation_gap_threshold_and_detached_labels(self):
        scores = torch.tensor([0.2, -0.4, 0.8], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([3.0, 1.0, 2.0], dtype=torch.float64, requires_grad=True)
        loss, count = pairwise_rank_loss(scores, labels, 1.0)
        # Strictly >1: only label3 vs label1, never reverse/double-count or self-pair.
        self.assertEqual(int(count), 1)
        self.assertAlmostEqual(float(loss.detach()), math.log1p(math.exp(-0.6)))
        loss.backward()
        self.assertLess(float(scores.grad[0]), 0)
        self.assertGreater(float(scores.grad[1]), 0)
        self.assertEqual(float(scores.grad[2]), 0)
        self.assertIsNone(labels.grad)

    def test_mean_over_pairs_and_ordered_predictions_have_lower_loss(self):
        labels = torch.tensor([3.0, 2.0, 0.0])
        ordered = torch.tensor([2.0, 1.0, -1.0])
        wrong = -ordered
        good, count = pairwise_rank_loss(ordered, labels, 0.1)
        bad, _ = pairwise_rank_loss(wrong, labels, 0.1)
        expected = sum(math.log1p(math.exp(-v)) for v in (1.0, 3.0, 2.0)) / 3
        self.assertEqual(int(count), 3)
        self.assertAlmostEqual(float(good), expected, places=6)
        self.assertLess(float(good), float(bad))

    def test_ties_singletons_invalid_and_near_ties_produce_zero_gradients(self):
        for labels, valid in (
            ([1.0], True),
            ([1.0, 1.0], True),
            ([1.0, 1.05], True),
            ([1.0, 4.0], False),
        ):
            scores = torch.ones(len(labels), requires_grad=True)
            loss, count = pairwise_rank_loss(scores, torch.tensor(labels), 0.1, valid)
            self.assertEqual(int(count), 0)
            self.assertEqual(float(loss.detach()), 0)
            loss.backward()
            self.assertTrue(torch.equal(scores.grad, torch.zeros_like(scores)))

    def test_zero_lambda_matches_missing_lambda_exactly_and_skips_pair_construction(self):
        a = PrivateRouter([(2, 2)], torch.device("cpu"), 42, {})
        b = PrivateRouter([(2, 2)], torch.device("cpu"), 42, {"router_rank_lambda": 0.0})
        observations = {
            "m": ("d2_2", torch.arange(18.0).reshape(3, 6) / 18, torch.tensor([-2.0, 1.0, 3.0]))
        }
        with patch(
            "afl_vlm.methods.ours_router.losses.pairwise_rank_loss",
            side_effect=AssertionError("disabled loss was called"),
        ):
            for _ in range(3):
                ra, rb = a.learn(observations), b.learn(observations)
                self.assertEqual(ra, rb)
                self.assertEqual(ra["router_loss"], ra["router_regression_loss"])
                self.assertEqual(ra["router_rank_pairs"], 0)
        for name, value in a.heads.state_dict().items():
            self.assertTrue(torch.equal(value, b.heads.state_dict()[name]))
        for key, state in a.optimizer.state_dict()["state"].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value, b.optimizer.state_dict()["state"][key][name]))

    def test_combined_objective_normalization_and_model_gradient_isolation(self):
        router = PrivateRouter(
            [(2, 2)],
            torch.device("cpu"),
            42,
            {"router_rank_lambda": 0.2, "router_rank_min_gap": 0.1},
        )
        features = torch.arange(18.0).reshape(3, 6).requires_grad_()
        labels = torch.tensor([-1.0, 2.0, 3.0], requires_grad=True)
        # Equal-label module has no rank pairs; it must not dilute the rank loss.
        observations = {"m": ("d2_2", features, labels), "tie": ("d2_2", features, torch.ones(3))}
        with torch.no_grad():
            predicted = router.heads["d2_2"](features).flatten()
            target = labels.detach() / (labels.detach().abs().mean() + 1e-12)
            expected, count = pairwise_rank_loss(predicted, target, 0.1)
        rng = torch.get_rng_state().clone()
        row = router.learn(observations)
        self.assertEqual(row["router_rank_pairs"], int(count))
        self.assertEqual(row["router_rank_valid_modules"], 1)
        self.assertAlmostEqual(row["router_rank_loss"], float(expected), places=6)
        self.assertAlmostEqual(
            row["router_loss"],
            row["router_regression_loss"] + 0.2 * row["router_rank_loss"],
            places=6,
        )
        self.assertIsNone(features.grad)
        self.assertIsNone(labels.grad)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(all(p.grad is None for p in router.heads.parameters()))
        restored = PrivateRouter([(2, 2)], torch.device("cpu"), 7, router.params)
        restored.load_state_dict(copy.deepcopy(router.state_dict()))
        self.assertEqual(router.learn(observations), restored.learn(observations))

    def test_configs_validate_new_loss_parameters_and_defaults(self):
        from afl_vlm.config import load_config
        from test_ours_router import ROOT

        for setting in (2, 5, 10):
            cfg = load_config(
                ROOT / f"configs/experiments/llava/afvlm_cm/{setting}clients/ours_router.yaml"
            )
            params = cfg["method"]["params"]
            self.assertEqual(params["router_rank_lambda"], 0.0)
            self.assertEqual(params["router_rank_min_gap"], 0.1)
            self.assertEqual(params["router_supervision"], "full_candidates")
            self.assertTrue(params["local_frozen_background"])
            self.assertEqual(params["gamma"], 0.75)
            OursRouter(params).validate_runtime()
        for key in ("router_rank_lambda", "router_rank_min_gap"):
            for value in (-1, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    OursRouter({key: value})

    def test_separate_losses_export_for_probe_and_formal_and_old_rows_stay_blank(self):
        import csv
        import json
        import runpy
        import tempfile
        from pathlib import Path

        from test_ours_router import ROOT
        from test_ours_router_conflict import ConflictServerChecks

        fixture = ConflictServerChecks()
        fixture.setUp()
        method, server = fixture.setup_method()
        packet = fixture.packet(method)
        router = PrivateRouter([(2, 2)], torch.device("cpu"), 42, {"router_rank_lambda": 0.1})
        name = next(iter(method.modules))
        row = router.learn(
            {name: ("d2_2", torch.arange(18.0).reshape(3, 6) / 18, torch.tensor([-2.0, 1.0, 3.0]))}
        )
        row["optimizer_step"] = 1
        packet.metadata["hc_local"]["trajectory"] = [row, {"optimizer_step": 2, "router_loss": 0.5}]
        result = server.receive(packet)[0].metadata
        export = runpy.run_path(str(ROOT / "tools/export_ours_router_diagnostics.py"))["export"]
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "events.jsonl").write_text(
                json.dumps({"result_metadata": [result]}) + "\n", encoding="utf-8"
            )
            (run / "client_calibration.json").write_text(
                json.dumps(
                    {
                        "clients": [
                            {
                                "client_id": "a/client_0",
                                "task": "a",
                                "modules": {},
                                "trajectory": [row],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            export(run, run / "csv")
            for filename in ("local_steps.csv", "probe_steps.csv"):
                with (run / "csv" / filename).open(encoding="utf-8-sig", newline="") as file:
                    rows = list(csv.DictReader(file))
                self.assertAlmostEqual(float(rows[0]["router_rank_loss"]), row["router_rank_loss"])
                self.assertEqual(int(rows[0]["router_rank_pairs"]), row["router_rank_pairs"])
                self.assertAlmostEqual(
                    float(rows[0]["router_loss"]),
                    float(rows[0]["router_regression_loss"])
                    + 0.1 * float(rows[0]["router_rank_loss"]),
                    places=6,
                )
                if len(rows) > 1:
                    self.assertEqual(rows[1]["router_rank_loss"], "")
