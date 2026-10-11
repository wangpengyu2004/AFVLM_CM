"""Non-training equation/state checks. No dataset job or LLaVA model is loaded."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from afl_vlm.federation.types import ClientContext, ClientSpec, ServerContext, Update
from afl_vlm.methods.base import staleness_weight
from afl_vlm.methods.continual_merge.algebra import opcm_filter, spectrum
from afl_vlm.methods.pilot.adapters import build_pilot_connector
from afl_vlm.methods.pilot.aggregation import pilot_aggregate
from afl_vlm.methods.registry import create_method
from afl_vlm.methods.unifed_lora.hypernetwork import LoRAHypernetwork
from afl_vlm.models.base import TrainConfig, clone_state, load_evaluation_state, subtract
from afl_vlm.scheduling.train_plan import build_train_plan, event_records, load_train_plan


def upload(
    value: float,
    *,
    cid: str = "vqa/client_0",
    base: float = 0,
    version: int = 0,
    n: int = 1,
    local_round: int = 0,
) -> Update:
    state, anchor = {"x": torch.tensor([value])}, {"x": torch.tensor([base])}
    return Update(
        f"{cid}-r{local_round}",
        cid,
        cid.split("/")[0],
        "dataset",
        n,
        local_round,
        version,
        1,
        anchor,
        state,
        subtract(state, anchor),
        42,
        "hash",
        [],
        2,
    )


class ClassicalEquations(unittest.TestCase):
    def test_fedavg_sample_weighted_and_duplicate_rejected(self) -> None:
        method = create_method("fedavg", {})
        ctx = ServerContext({"x": torch.tensor([0.0])}, 0, 2)
        first = upload(2, n=1)
        self.assertEqual(method.on_arrival(first, ctx), [])
        with self.assertRaises(ValueError):
            method.on_arrival(first, ctx)
        result = method.on_arrival(upload(6, cid="cls/client_0", n=3), ctx)[0]
        torch.testing.assert_close(result.new_state["x"], torch.tensor([5.0]))

    def test_proximal_loss_is_download_anchored_and_differentiable(self) -> None:
        for name, params, mu in (
            ("fedprox", {"mu": 0.01}, 0.01),
            ("fedasync", {"proximal_mode": "paper_loss", "proximal_mu": 0.1}, 0.1),
        ):
            p = torch.nn.Parameter(torch.tensor([3.0]))
            model = SimpleNamespace(named_federated_parameters=lambda p=p: [("x", p)])
            loss = create_method(name, params).local_loss(
                p.sum() * 0, model, None, {"base_state": {"x": torch.tensor([1.0])}}
            )
            self.assertAlmostEqual(float(loss.detach()), 2 * mu)
            loss.backward()
            torch.testing.assert_close(p.grad, torch.tensor([2 * mu]))

    def test_fedadam_variants_match_independent_scalar_recursions(self) -> None:
        for variant in ("paper", "keras", "legacy"):
            method = create_method(
                "fedadam",
                {
                    "server_lr": 0.2,
                    "beta1": 0.8,
                    "beta2": 0.7,
                    "epsilon": 0.03,
                    "variant": variant,
                },
            )
            m = position = 0.0
            v = 0.03**2 if variant == "paper" else 0.0
            for t, delta in enumerate((2.0, -1.0, 0.4), 1):
                ctx = ServerContext({"x": torch.tensor([position])}, t - 1, 1)
                row = upload(position + delta, base=position, local_round=t)
                result = method.on_arrival(row, ctx)[0]
                m, v = 0.8 * m + 0.2 * delta, 0.7 * v + 0.3 * delta**2
                mh, vh, lr = m, v, 0.2
                if variant == "keras":
                    lr *= (1 - 0.7**t) ** 0.5 / (1 - 0.8**t)
                elif variant == "legacy":
                    mh, vh = m / (1 - 0.8**t), v / (1 - 0.7**t)
                position += lr * mh / (vh**0.5 + 0.03)
                self.assertAlmostEqual(float(result.new_state["x"]), position, places=6)

    def test_local_only_keeps_server_unchanged_and_clients_separate(self) -> None:
        method = create_method("local", {})
        ctx = ServerContext({"x": torch.tensor([10.0])}, 7, 2)
        for cid, val in (("vqa/client_0", 2), ("vqa/client_1", 5)):
            mutation = method.on_arrival(upload(val, cid=cid), ctx)[0]
            self.assertFalse(mutation.increment_version)
            torch.testing.assert_close(mutation.new_state["x"], ctx.global_state["x"])
        states = method.evaluation_states(ctx.global_state)
        self.assertEqual(float(states["vqa/client_0"]["x"]), 2)
        self.assertEqual(float(states["vqa/client_1"]["x"]), 5)


class AsynchronousEquations(unittest.TestCase):
    def test_staleness_functions(self) -> None:
        self.assertEqual(staleness_weight(3, {"type": "constant"}), 1)
        self.assertEqual(staleness_weight(3, {"type": "polynomial", "a": 0.5}), 0.5)
        self.assertEqual(staleness_weight(2, {"type": "hinge", "a": 1, "b": 2}), 1)
        self.assertEqual(staleness_weight(4, {"type": "hinge", "a": 1, "b": 2}), 1 / 3)
        with self.assertRaises(ValueError):
            staleness_weight(-1, {})

    def test_fedasync_interpolates_client_model_not_download_delta(self) -> None:
        method = create_method(
            "fedasync", {"alpha": 0.5, "staleness": {"type": "polynomial", "a": 0.5}}
        )
        row = upload(2, base=0, version=0)
        ctx = ServerContext({"x": torch.tensor([10.0])}, 3, 1)
        mutation = method.on_arrival(row, ctx)[0]
        self.assertEqual(mutation.applied_weight, 0.25)
        self.assertEqual(float(mutation.new_state["x"]), 8)

    def test_author_contraction_does_not_overwrite_existing_gradient(self) -> None:
        method = create_method("fedasync", {"proximal_mode": "official_contraction", "rho": 0.2})
        p = torch.nn.Parameter(torch.tensor([3.0]))
        p.grad = torch.tensor([7.0])
        model = SimpleNamespace(named_federated_parameters=lambda: [("x", p)])
        method.transform_gradients(model, {"base_state": {"x": torch.tensor([1.0])}})
        torch.testing.assert_close(p.detach(), torch.tensor([2.6]))
        torch.testing.assert_close(p.grad, torch.tensor([7.0]))

    def test_fedasync_decay_milestone(self) -> None:
        method = create_method(
            "fedasync", {"alpha": 0.8, "alpha_decay": 0.5, "alpha_decay_milestones": [2]}
        )
        result = method.on_arrival(upload(2), ServerContext({"x": torch.tensor([0.0])}, 2, 1))[0]
        self.assertEqual(result.applied_weight, 0.4)

    def test_fedbuff_single_item_staleness_does_not_cancel(self) -> None:
        method = create_method(
            "fedbuff",
            {
                "buffer_size": 1,
                "staleness_weighting": True,
                "staleness": {"type": "polynomial", "a": 0.5},
                "weighting": "uniform",
                "server_lr": 2,
            },
        )
        result = method.on_arrival(upload(2), ServerContext({"x": torch.tensor([10.0])}, 3, 1))[0]
        self.assertEqual(float(result.new_state["x"]), 12)

    def test_fedbuff_uniform_vs_samples_and_residual_flush(self) -> None:
        for scheme, expected in (("uniform", 4), ("sample_size", 5)):
            method = create_method("fedbuff", {"buffer_size": 3, "weighting": scheme})
            ctx = ServerContext({"x": torch.tensor([0.0])}, 0, 2)
            method.on_arrival(upload(2, n=1), ctx)
            method.on_arrival(upload(6, n=3, cid="vqa/client_1"), ctx)
            result = method.on_finish(ctx)[0]
            self.assertEqual(float(result.new_state["x"]), expected)
            self.assertEqual(len(result.contributing_update_ids), 2)
            self.assertEqual(method.on_finish(ctx), [])

    def test_fedasmu_refuses_missing_real_objective_feedback(self) -> None:
        method = create_method("fedasmu", {})
        method.validate_runtime()
        with self.assertRaisesRegex(RuntimeError, "client-training-objective feedback"):
            method.on_arrival(upload(1), ServerContext({"x": torch.tensor([0.0])}, 0, 1))


class MasEquations(unittest.TestCase):
    def initialized(self, name: str):
        method = create_method(
            name, {"beta": 0.9, "client_lr": 0.1, "server_lr": 1 if name == "adamasfl" else 0.01}
        )
        method.configure_server(
            {}, [SimpleNamespace(id="vqa/client_0"), SimpleNamespace(id="vqa/client_1")]
        )
        for cid, gradient in (("vqa/client_0", 1.0), ("vqa/client_1", 3.0)):
            row = upload(0, cid=cid)
            row.mean_gradient = {"x": torch.tensor([gradient])}
            method.initialize_client_gradient(row)
        method.finalize_initial_gradients()
        return method

    def test_initial_controls_and_optimizer_requirements(self) -> None:
        for name in ("masfl", "adamasfl"):
            method = self.initialized(name)
            self.assertEqual(float(method.global_control["x"]), 2)
            self.assertEqual(float(method.momentum["x"]), 2)
            config = method.configure_local_training(TrainConfig(1, 4, 4, 2e-5, 2048, 42))
            self.assertEqual(config.optimizer, "sgd")
            self.assertEqual(config.learning_rate, 0.1)
            self.assertEqual(config.batch_size, 4)

    def test_local_corrected_direction_and_normalization(self) -> None:
        for name, expected in (("masfl", 4.7), ("adamasfl", 1.0)):
            method = self.initialized(name)
            p = torch.nn.Parameter(torch.tensor([0.0]))
            p.grad = torch.tensor([4.0])
            model = SimpleNamespace(named_federated_parameters=lambda p=p: [("x", p)])
            method.transform_gradients(model, {"client": SimpleNamespace(client_id="vqa/client_0")})
            self.assertAlmostEqual(float(p.grad), expected, places=6)

    def test_server_recursions_use_actual_eta_k_and_population(self) -> None:
        for name, expected_position, expected_momentum in (
            ("masfl", 9.962, 3.8),
            ("adamasfl", 9.8, 3.62),
        ):
            method = self.initialized(name)
            row = upload(9.96, base=10)
            row.mean_gradient = {"x": torch.tensor([3.0])}
            row.metadata["local_learning_rate"] = 0.1
            method.prepare_upload(row, None)
            result = method.on_arrival(row, ServerContext({"x": torch.tensor([10.0])}, 0, 2))[0]
            self.assertAlmostEqual(float(result.new_state["x"]), expected_position, places=5)
            self.assertAlmostEqual(float(method.momentum["x"]), expected_momentum, places=5)
            self.assertEqual(float(method.global_control["x"]), 3)
            restored = create_method(name, {})
            restored.load_state_dict(method.state_dict())
            torch.testing.assert_close(restored.momentum["x"], method.momentum["x"])


class CompassEquations(unittest.TestCase):
    def test_dynamic_plan_preserves_jobs_and_client_sequential_work(self) -> None:
        specs = [
            ClientSpec(f"vqa/client_{i}", "vqa", "AOKVQA", 100, speed, delay, 0, 1)
            for i, (speed, delay) in enumerate(((0.5, 0), (1, 5), (4, 30)))
        ]
        params = {"min_local_steps": 2, "max_local_steps": 10}
        events = build_train_plan(specs, 5, 1, 4, 4, "semi_asynchronous", params)
        self.assertEqual(len(events), 15)
        self.assertEqual(len({(e.client_id, e.local_round) for e in events}), 15)
        self.assertTrue(all(2 <= e.local_steps <= 10 for e in events))
        self.assertGreater(len({e.local_steps for e in events}), 1)
        for spec in specs:
            jobs = sorted(
                (e for e in events if e.client_id == spec.id), key=lambda e: e.local_round
            )
            for previous, next_job in zip(jobs, jobs[1:], strict=False):
                self.assertGreaterEqual(
                    next_job.start_time, previous.metadata["communication_arrival_time"]
                )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "plan.json"
            path.write_text(json.dumps(event_records(events)), encoding="utf-8")
            self.assertEqual(load_train_plan(path), events)

    def test_aggregation_uses_global_population_and_stale_delta(self) -> None:
        method = create_method(
            "fedcompass", {"alpha": 0.9, "staleness": {"type": "polynomial", "a": -0.5}}
        )
        row = upload(2)
        row.group_id = 1
        row.metadata["group_size"] = 1
        result = method.on_arrival(row, ServerContext({"x": torch.tensor([10.0])}, 3, 2))[0]
        self.assertAlmostEqual(float(result.new_state["x"]), 10.45, places=5)


class TaskHeterogeneityEquations(unittest.TestCase):
    def test_pilot_stage_freezing_token_router_and_difference(self) -> None:
        base = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.GELU(), torch.nn.Linear(3, 3))
        wrapper = build_pilot_connector(base, ["vqa", "caption"], normalize_difference=False)
        value = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2) / 8
        wrapper.configure_local("vqa", "vqa/client_0", 1, None)
        self.assertTrue(all(p.requires_grad for p in wrapper.private_adapter.parameters()))
        self.assertTrue(
            all(not p.requires_grad for p in wrapper.task_adapters["caption"].parameters())
        )
        expected = wrapper.task_adapters["vqa"](value) + wrapper.private_adapter(value)
        torch.testing.assert_close(wrapper(value), expected)
        x, y = wrapper.task_adapters["vqa"](value), wrapper.private_adapter(value)
        diff = (x.transpose(-1, -2) @ y).square().sum(dim=(-2, -1)).mean()
        torch.testing.assert_close(wrapper.auxiliary_losses["difference"], diff)
        private = wrapper.export_private()
        wrapper.configure_local("vqa", "vqa/client_0", 2, private)
        self.assertTrue(all(not p.requires_grad for p in wrapper.private_adapter.parameters()))
        self.assertTrue(all(p.requires_grad for p in wrapper.router.parameters()))
        out = wrapper(value)
        self.assertEqual(out.shape, (2, 2, 3))
        self.assertNotIn("difference", wrapper.auxiliary_losses)
        self.assertAlmostEqual(float(wrapper.auxiliary_losses["load_balance"].detach()), 1)
        self.assertAlmostEqual(
            float(wrapper.auxiliary_losses["router_z"].detach()),
            float(torch.tensor(2.0).log() ** 2),
        )
        wrapper.set_context("vqa", None)
        torch.testing.assert_close(wrapper(value), wrapper.task_adapters["vqa"](value))

    def test_pilot_text_adaptation_retains_size_and_distance(self) -> None:
        rows = [upload(2, n=1), upload(6, cid="caption/client_0", n=3)]
        for row in rows:
            row.local_state = {"p.lora_A.weight": row.local_state["x"]}
        global_view, personalized = pilot_aggregate(
            rows, {"p.lora_A.weight": torch.tensor([0.0])}, 1
        )
        self.assertEqual(float(global_view["p.lora_A.weight"]), 5)
        self.assertEqual(float(personalized[rows[0].client_id]["p.lora_A.weight"]), 5)

    def factors(self):
        a = "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        b = a.replace("lora_A", "lora_B")
        return {a: torch.tensor([[1.0, 0.0]]), b: torch.zeros(2, 1)}

    def test_unifed_generates_factors_vjp_sign_and_does_not_reset_rng(self) -> None:
        initial = self.factors()
        before = torch.random.get_rng_state().clone()
        hyper = LoRAHypernetwork(
            initial,
            {"vqa/client_0": "vqa", "caption/client_0": "caption"},
            {"vqa": [1, 0], "caption": [0, 1]},
            4,
            2,
            0.01,
            42,
            "cpu",
        )
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        downloaded = clone_state(hyper.generate("vqa/client_0"))
        for key in initial:
            torch.testing.assert_close(downloaded[key], initial[key])
        trained = {key: value + 0.1 for key, value in downloaded.items()}
        hyper.begin_meta_step()
        hyper.accumulate_meta_gradient("vqa/client_0", trained, downloaded, 1)
        head = hyper.module["heads"]["client1_module0"]
        torch.testing.assert_close(head.bias.grad, torch.full_like(head.bias, -0.1))
        hyper.finish_meta_step()
        generated = hyper.generate("vqa/client_0")
        self.assertTrue(all(bool((generated[k] > downloaded[k]).all()) for k in generated))
        untouched = hyper.generate("caption/client_0")
        for key in initial:
            torch.testing.assert_close(untouched[key], initial[key])

    def test_unifed_download_anchor_checkpoint_and_secondary_states(self) -> None:
        method = create_method(
            "unifed_lora", {"hidden_size": 4, "embedding_dim": 2, "hyper_device": "cpu"}
        )
        initial = self.factors()
        method.configure_server(initial, [SimpleNamespace(id="vqa/client_0", task="vqa")])
        context = ClientContext("vqa/client_0", "vqa", "AOKVQA", 3, 0, 0, 1, 42)
        start = method.prepare_download(initial, context)
        row = upload(1)
        row.local_state = {key: value + 0.01 for key, value in start.items()}
        row = method.prepare_upload(row, context)
        for key in initial:
            torch.testing.assert_close(row.base_state[key], start[key])
        method.on_arrival(row, ServerContext(initial, 0, 1))
        states = method.secondary_evaluation_states(initial)
        restored = create_method("unifed_lora", {})
        restored.load_state_dict(copy.deepcopy(method.state_dict()))
        for key in initial:
            torch.testing.assert_close(
                restored.secondary_evaluation_states(initial)[context.client_id][key],
                states[context.client_id][key],
            )

    def test_personalized_payload_loads_private_separately(self) -> None:
        calls = []
        model = SimpleNamespace(
            load_trainable=lambda state: calls.append(("shared", state)),
            load_method_private_state=lambda state: calls.append(("private", state)),
        )
        load_evaluation_state(
            model, {"__federated_state__": {"a": 1}, "__private_state__": {"b": 2}}
        )
        self.assertEqual(calls, [("shared", {"a": 1}), ("private", {"b": 2})])

    def test_singular_mass_cutoff_boundaries(self) -> None:
        old = torch.diag(torch.tensor([4.0, 1.0]))
        self.assertEqual(len(spectrum(old, 0.5)[1]), 1)
        self.assertEqual(len(spectrum(old, 1)[1]), 2)
        self.assertEqual(opcm_filter(torch.ones(2, 2), old, 1, 1e-8)[1], 2)


if __name__ == "__main__":
    unittest.main()
