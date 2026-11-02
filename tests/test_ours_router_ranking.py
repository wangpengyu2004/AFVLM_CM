"""Read-only diagnostic math, optimizer equivalence and log export; no VLM runs."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from afl_vlm.methods.ours_router.local import LocalRouter
from afl_vlm.methods.ours_router.ranking import average_ranks, ranking_report, ranking_tensor
from afl_vlm.methods.ours_router.router import PrivateRouter

ROOT = Path(__file__).resolve().parents[1]


def measure(predictions, labels, valid=True):
    values = ranking_tensor(torch.tensor(predictions), torch.tensor(labels), valid)
    return ranking_report(["m"], [values.tolist()])


class RankingMathChecks(unittest.TestCase):
    def test_perfect_reversed_and_signed_rank_order(self):
        perfect = measure([-3., -1., 2., 4.], [-30., -10., 20., 40.])
        row = perfect["modules"]["m"]
        self.assertAlmostEqual(row["spearman"], 1.)
        self.assertEqual(row["topk_overlap"], 1.)
        self.assertEqual(row["top1_overlap"], 1.)
        self.assertEqual((row["topk_k"], row["observed_components"]), (1, 4))
        self.assertEqual(row["topk_chance_overlap"], .25)
        reverse = measure([4., 3., 2., 1.], [1., 2., 3., 4.])["modules"]["m"]
        self.assertAlmostEqual(reverse["spearman"], -1.)
        self.assertEqual(reverse["topk_overlap"], 0.)

    def test_spearman_uses_average_tied_ranks_and_cutoff_ties_are_unavailable(self):
        ranks = average_ranks(torch.tensor([3., 1., 1., 2.]))
        self.assertEqual(ranks.tolist(), [3., .5, .5, 2.])
        row = measure([1., 1., 2., 3.], [4., 3., 2., 1.])["modules"]["m"]
        self.assertAlmostEqual(row["spearman"], -4.5 / math.sqrt(22.5), places=6)
        self.assertTrue(row["spearman_valid"])
        self.assertTrue(row["prediction_has_ties"])
        tied = measure([4., 4., 2., 1.], [4., 3., 2., 1.])["modules"]["m"]
        self.assertTrue(tied["spearman_valid"])
        self.assertIsNone(tied["topk_overlap"])
        self.assertIsNone(tied["top1_overlap"])
        self.assertTrue(tied["prediction_cutoff_tie"])

    def test_topk_is_intersection_over_k_and_does_not_use_the_entire_pool(self):
        row = measure([8., 7., 6., 5., 4., 3., 2., 1.],
                      [8., 6., 7., 5., 4., 3., 2., 1.])["modules"]["m"]
        self.assertEqual(row["topk_k"], 2)
        self.assertEqual(row["topk_overlap"], .5)
        self.assertEqual(row["top1_overlap"], 1.)
        self.assertEqual(row["topk_chance_overlap"], .25)
        row = measure([5., 4., 3., 2., 1.], [5., 4., 3., 2., 1.])["modules"]["m"]
        self.assertEqual(row["topk_k"], 2)

    def test_constant_singleton_zero_label_and_nonfinite_vectors_never_fake_full_scores(self):
        cases = [([0., 0., 0.], [1., 2., 3.], True),
                 ([1., 2., 3.], [0., 0., 0.], True),
                 ([1.], [2.], True), ([1., 2.], [1., 2.], False),
                 ([float("nan"), 1.], [1., 2.], True)]
        for p, y, valid in cases:
            report = measure(p, y, valid)
            for metric in ("spearman", "topk_overlap", "top1_overlap"):
                self.assertIsNone(report["modules"]["m"][metric])
                self.assertIsNone(report[f"{metric}_mean"])
                self.assertEqual(report[f"{metric}_valid_modules"], 0)
            json.dumps(report, allow_nan=False)

    def test_step_summary_excludes_invalid_modules_and_does_not_average_with_zero(self):
        values = [ranking_tensor(torch.tensor([1., 2., 3.]), torch.tensor([1., 2., 3.]), True),
                  ranking_tensor(torch.tensor([1.]), torch.tensor([2.]), True)]
        report = ranking_report(["valid", "rank_one"], torch.stack(values).tolist())
        self.assertAlmostEqual(report["spearman_mean"], 1.)
        self.assertEqual(report["spearman_valid_modules"], 1)

    def test_diagnostics_are_detached_leave_gradients_values_and_rng_unchanged(self):
        p = torch.tensor([3., 2., 1.], requires_grad=True)
        y = torch.tensor([1., 2., 3.], requires_grad=True)
        p.grad, y.grad = torch.ones_like(p), torch.full_like(y, 2.)
        values, grads = (p.clone(), y.clone()), (p.grad.clone(), y.grad.clone())
        rng = torch.get_rng_state()
        diagnostic = ranking_tensor(p, y, True)
        self.assertFalse(diagnostic.requires_grad)
        self.assertTrue(torch.equal(p, values[0]) and torch.equal(y, values[1]))
        self.assertTrue(torch.equal(p.grad, grads[0]) and torch.equal(y.grad, grads[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))


class TrainingAndExportIsolationChecks(unittest.TestCase):
    def test_enabled_and_disabled_diagnostics_have_identical_router_updates_and_optimizer(self):
        a = PrivateRouter([(2, 2)], torch.device("cpu"), 42, {})
        b = PrivateRouter([(2, 2)], torch.device("cpu"), 42, {})
        observations = {"m": ("d2_2", torch.arange(24.).reshape(4, 6) / 24,
                              torch.tensor([-2., 1., 3., 4.]))}
        for _ in range(3):
            with patch.dict(os.environ, {"AFVLM_PRIMARY_PROCESS": "1"}):
                primary = a.learn(observations)
            with patch.dict(os.environ, {"AFVLM_PRIMARY_PROCESS": "0"}):
                other = b.learn(observations)
            for field in (
                "router_loss", "router_valid_modules", "label_abs_mean", "router_lifetime_steps"
            ):
                self.assertEqual(primary[field], other[field])
            self.assertIn("router_ranking", primary)
            self.assertNotIn("router_ranking", other)
        for name, value in a.heads.state_dict().items():
            self.assertTrue(torch.equal(value, b.heads.state_dict()[name]))
        for key, moment in a.optimizer.state_dict()["state"].items():
            for name, value in moment.items():
                self.assertTrue(torch.equal(value, b.optimizer.state_dict()["state"][key][name]))
        self.assertEqual(a.state_dict().keys(), b.state_dict().keys())

    def test_phase_scope_and_global_indices_added_without_changing_the_selected_set(self):
        from test_ours_router import FactorRegistry

        controller = LocalRouter(FactorRegistry(), {})
        name = next(iter(controller.full_metadata))
        for probing, indices in ((True, [0, 1, 2]), (False, [0, 2])):
            report = measure(list(range(len(indices))), list(range(len(indices))))
            controller.client_router = SimpleNamespace(
                learn=lambda _observations, saved=report: {"router_ranking": copy.deepcopy(saved)}
            )
            controller.probing, controller.pending = probing, {"observation": 1}
            report["modules"][name] = report["modules"].pop("m")
            controller.choice = {name: list(indices)}
            controller.loss_sum, controller.loss_count = torch.tensor(2.), 1
            controller.step(1)
            ranking = controller.trajectory[-1]["router_ranking"]
            self.assertEqual(ranking["phase"], "probe" if probing else "formal")
            self.assertEqual(ranking["scope"], "full_candidate_components" if probing
                             else "selected_components_only")
            self.assertEqual(ranking["modules"][name]["component_indices"], indices)
            self.assertEqual(controller.choice[name], indices)

    def test_new_and_old_log_formats_export_without_inventing_missing_rank_measurements(self):
        from test_ours_router_conflict import ConflictServerChecks

        fixture = ConflictServerChecks()
        fixture.setUp()
        method, server = fixture.setup_method()
        packet = fixture.packet(method)
        name = next(iter(method.modules))
        ranking = measure([1., 2.], [1., 2.])
        ranking["phase"], ranking["scope"] = "formal", "selected_components_only"
        ranking["modules"][name] = ranking["modules"].pop("m")
        ranking["modules"][name].update(component_indices=[0, 2], global_candidate_rank=3)
        trajectory = [{"optimizer_step": 1, "router_ranking": ranking,
                       "router_spearman_mean": 1.}, {"optimizer_step": 2, "router_loss": .1}]
        packet.metadata["hc_local"]["trajectory"] = trajectory
        result = server.receive(packet)[0].metadata
        exporter = runpy.run_path(str(ROOT / "tools/export_ours_router_diagnostics.py"))["export"]
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "events.jsonl").write_text(json.dumps({"result_metadata": [result]}) + "\n",
                                             encoding="utf-8")
            probe_ranking = copy.deepcopy(ranking)
            probe_ranking.update(phase="probe", scope="full_candidate_components")
            (run / "client_calibration.json").write_text(json.dumps({"clients": [{
                "client_id": "a/client_0", "task": "a", "modules": {},
                "trajectory": [{"optimizer_step": 1, "router_ranking": probe_ranking}],
            }]}), encoding="utf-8")
            counts = exporter(run, run / "export")
            self.assertEqual(counts["router_ranking"], 2)
            with (run / "export/router_ranking.csv").open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({r["phase"] for r in rows}, {"formal", "probe"})
            self.assertTrue(all(r["spearman"] == "1.0" for r in rows))
            with (run / "export/local_steps.csv").open(encoding="utf-8-sig", newline="") as f:
                steps = list(csv.DictReader(f))
            self.assertEqual(steps[1]["router_spearman_mean"], "")


if __name__ == "__main__":
    unittest.main()
