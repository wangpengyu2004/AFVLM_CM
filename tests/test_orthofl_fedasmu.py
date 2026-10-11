"""Non-training equation/controller/timeline checks, no VLM forward or GPU load."""

from __future__ import annotations

import copy
import math
import unittest
from types import SimpleNamespace

import torch
from afl_vlm.federation.types import ClientContext, ServerContext, Update
from afl_vlm.methods.fedasmu import FedASMU
from afl_vlm.methods.fedasmu_controller import RequestController
from afl_vlm.methods.registry import create_method
from afl_vlm.models.base import clone_state, state_dot, subtract
from afl_vlm.scheduling.timeline import EventTimeline


def context(round_index=0, version=0):
    return ClientContext("vqa/client_0", "vqa", "AOKVQA", 128, round_index, version, 10, 42)


def update(base, local, version=0, round_index=0):
    return Update(
        "vqa/client_0-r0",
        "vqa/client_0",
        "vqa",
        "AOKVQA",
        128,
        round_index,
        version,
        10,
        clone_state(base),
        clone_state(local),
        subtract(local, base),
        42,
        "hash",
        [],
        10,
    )


class OrthoEquations(unittest.TestCase):
    def test_projection_and_raw_global_aggregation_are_distinct(self):
        method = create_method("orthofl", {"beta": 0.6, "a": 0.5})
        key = "base.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        base = {key: torch.tensor([0.0, 0.0])}
        current = {key: torch.tensor([2.0, 3.0])}
        local = {key: torch.tensor([1.0, 0.0])}
        method.prepare_download(base, context())
        row = method.prepare_upload(update(base, local), context())
        result = method.on_arrival(row, ServerContext(current, 2, 2))[0]
        weight = 0.6 / math.sqrt(3)
        torch.testing.assert_close(
            result.new_state[key], current[key] * (1 - weight) + local[key] * weight
        )
        # Parallel [2,0] removed, perpendicular [0,3] is sent only to client.
        torch.testing.assert_close(
            method.client_models[row.client_id][key], torch.tensor([1.0, 3.0])
        )
        self.assertAlmostEqual(result.metadata["orthogonality_residuals"]["layer0"], 0)
        start = method.prepare_download(result.new_state, context(1, 3))
        torch.testing.assert_close(start[key], torch.tensor([1.0, 3.0]))
        self.assertFalse(torch.equal(start[key], result.new_state[key]))

    def test_joint_layer_projection_does_not_project_each_factor_independently(self):
        method = create_method("orthofl")
        keys = [f"layers.1.q_proj.lora_{factor}.weight" for factor in ("A", "B")]
        base = {k: torch.tensor([0.0]) for k in keys}
        local = {keys[0]: torch.tensor([1.0]), keys[1]: torch.tensor([1.0])}
        current = {keys[0]: torch.tensor([3.0]), keys[1]: torch.tensor([1.0])}
        method.prepare_download(base, context())
        row = method.prepare_upload(update(base, local), context())
        method.on_arrival(row, ServerContext(current, 1, 2))
        perpendicular = subtract(method.client_models[row.client_id], local)
        self.assertAlmostEqual(state_dot(perpendicular, row.delta), 0)
        torch.testing.assert_close(perpendicular[keys[0]], torch.tensor([1.0]))
        torch.testing.assert_close(perpendicular[keys[1]], torch.tensor([-1.0]))

    def test_checkpoint_and_reference_are_previous_post_aggregation_not_next_download(self):
        method = create_method("orthofl")
        base = {"x": torch.tensor([0.0, 0.0])}
        method.prepare_download(base, context())
        row = method.prepare_upload(update(base, {"x": torch.tensor([1.0, 0.0])}), context())
        result = method.on_arrival(row, ServerContext(base, 0, 2))[0]
        restored = create_method("orthofl")
        restored.load_state_dict(copy.deepcopy(method.state_dict()))
        # More other-client updates before the next job starts do not erase drift.
        restored.prepare_download({"x": torch.tensor([3.0, 3.0])}, context(1, 5))
        torch.testing.assert_close(
            restored.global_references[row.client_id]["x"], result.new_state["x"]
        )
        self.assertEqual(restored.reference_versions[row.client_id], 1)

    def test_zero_delta_has_no_division_by_zero(self):
        method = create_method("orthofl")
        base = {"x": torch.tensor([0.0, 0.0])}
        method.prepare_download(base, context())
        row = method.prepare_upload(update(base, base), context())
        method.on_arrival(row, ServerContext({"x": torch.tensor([2.0, 3.0])}, 1, 2))
        torch.testing.assert_close(
            method.client_models[row.client_id]["x"], torch.tensor([2.0, 3.0])
        )


class ASMUEquations(unittest.TestCase):
    def finite_difference(self, fn, controls):
        weight, derivatives = fn(controls)
        for key in controls:
            plus, minus = dict(controls), dict(controls)
            plus[key] += 1e-6
            minus[key] -= 1e-6
            numerical = (fn(plus)[0] - fn(minus)[0]) / 2e-6
            self.assertAlmostEqual(derivatives[key], numerical, places=7)
        self.assertTrue(0 <= weight <= 1)

    def test_exact_server_coefficient_derivatives(self):
        controls = {"lambda": 1.0, "sigma": 0.5, "iota": 0.1}
        self.finite_difference(lambda c: FedASMU.server_weight(c, 4, 2, 1), controls)
        xi = 1 / (2 * math.sqrt(3)) + 0.1
        self.assertAlmostEqual(FedASMU.server_weight(controls, 4, 2, 1)[0], xi / (1 + xi))

    def test_exact_local_coefficient_derivatives_and_no_fresh_model(self):
        controls = {"gamma": 1.0, "upsilon": 0.5}
        self.finite_difference(lambda c: FedASMU.local_weight(c, 4, 1, 1), controls)
        self.assertEqual(FedASMU.local_weight(controls, 1, 1, 1)[0], 0)

    def test_learned_server_controls_used_before_model_mix(self):
        method = create_method("fedasmu", {"server_meta_lr": 0.1})
        base = {"x": torch.tensor([0.0])}
        row = update(base, {"x": torch.tensor([1.0])})
        request = method.arrival_feedback_request(row, ServerContext(base, 2, 2))
        row.metadata["arrival_objective_feedback"] = {
            "server_version": 2,
            "directional_derivative": -1.0,
        }
        result = method.on_arrival(row, ServerContext(base, 2, 2))[0]
        before, after = (
            result.metadata["server_controls_before"],
            result.metadata["server_controls_after"],
        )
        self.assertGreater(after["lambda"], before["lambda"])
        self.assertGreater(after["iota"], before["iota"])
        self.assertLess(after["sigma"], before["sigma"])
        self.assertGreater(result.applied_weight, request["weight"])
        torch.testing.assert_close(result.new_state["x"], torch.tensor([result.applied_weight]))

    def test_bound_rejection_requires_no_objective_probe(self):
        method = create_method("fedasmu", {"tau": 1})
        base = {"x": torch.tensor([0.0])}
        row = update(base, {"x": torch.tensor([1.0])})
        self.assertIsNone(method.arrival_feedback_request(row, ServerContext(base, 2, 1)))
        self.assertEqual(method.on_arrival(row, ServerContext(base, 2, 1)), [])
        self.assertEqual(method.server_controls, {})

    def test_controller_policy_q_refinement_and_state_are_reproducible(self):
        controller = RequestController({}, "cpu")
        decision = controller.select("client", 0, 42, 10, 0)
        self.assertEqual(decision["kind"], "lstm")
        before = [p.detach().clone() for p in controller.head.parameters()]
        controller.observe("client", 0, decision, 1.0)
        self.assertTrue(
            any(
                not torch.equal(a, b)
                for a, b in zip(before, controller.head.parameters(), strict=True)
            )
        )
        decision2 = controller.select("client", 1, 43, 10, 5)
        self.assertEqual(decision2["kind"], "q_learning")
        controller.observe("client", 1, decision2, 0.5)
        row = controller.clients["client"]
        self.assertNotEqual(row["q"][decision2["previous_slot"]][decision2["action"]], 0)
        other = RequestController({}, "cpu")
        other.load_state_dict(controller.state_dict())
        self.assertEqual(
            other.select("client", 2, 44, 10, 8), controller.select("client", 2, 44, 10, 8)
        )

    def test_dispatch_and_checkpoint_without_server_examples(self):
        method = create_method("fedasmu")
        initial = {"x": torch.tensor([0.0])}
        method.configure_server(initial, [])
        step = method.select_refresh_step(context(), 10)
        self.assertTrue(0 < step < 10)
        runtime = method.client_runtime_state(context())
        self.assertEqual(runtime["decision"]["step"], step)
        self.assertNotIn("dataset", runtime)
        restored = create_method("fedasmu")
        restored.configure_server(initial, [])
        restored.load_state_dict(method.state_dict())
        self.assertEqual(restored.client_runtime_state(context()), runtime)
        method.prepare_upload(update(initial, initial), context())
        self.assertEqual(method.decisions, {})

    def test_short_job_and_non_finite_feedback(self):
        method = create_method("fedasmu")
        method.configure_server({"x": torch.tensor([0.0])}, [])
        self.assertIsNone(method.select_refresh_step(context(), 1))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            method._learn({"gamma": 1.0}, {"gamma": 1.0}, float("nan"), "local")


class VirtualTimeline(unittest.TestCase):
    def test_other_methods_retain_original_sort_order(self):
        plan = [
            SimpleNamespace(
                event_id=i,
                client_id=f"c{i % 3}",
                local_round=i // 3,
                start_time=i * 0.1,
                arrival_time=1 + i * 0.1,
            )
            for i in range(12)
        ]
        old = [
            (time, priority, kind, e)
            for e in plan
            for time, priority, kind in ((e.start_time, 2, "start"), (e.arrival_time, 0, "arrival"))
        ]
        old.sort(key=lambda item: (item[0], item[1], item[3].client_id, item[3].local_round))
        self.assertEqual(list(EventTimeline(plan)), old)

    def test_refresh_is_inserted_causally_and_before_arrival(self):
        e = SimpleNamespace(
            event_id=0,
            client_id="c",
            local_round=0,
            start_time=0,
            arrival_time=11,
            estimated_train_time=10,
        )
        timeline = EventTimeline([e])
        iterator = iter(timeline)
        self.assertEqual(next(iterator)[2], "start")
        self.assertEqual(timeline.refresh(e, 3, 10), 3)
        self.assertEqual([x[2] for x in iterator], ["refresh", "arrival"])


if __name__ == "__main__":
    unittest.main()
