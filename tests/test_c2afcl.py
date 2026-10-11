"""Non-training C2-AFCL equation/state/config checks; no model or forward/backward."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from afl_vlm.config import load_config, validate_config
from afl_vlm.execution import resolve_runtime_backend
from afl_vlm.federation.server import FederatedServer
from afl_vlm.federation.types import ServerContext, Update
from afl_vlm.methods.registry import create_method
from afl_vlm.models.base import TrainConfig, clone_state, subtract

ROOT = Path(__file__).resolve().parents[1]


def upload(
    values: list[float],
    task: str = "cls",
    client: str = "0",
    round_index: int = 0,
    base_version: int = 0,
    base: list[float] | None = None,
    samples: int = 1,
) -> Update:
    anchor = {"x": torch.tensor(base or [0.0] * len(values))}
    raw = {"x": torch.tensor(values)}
    local = {"x": anchor["x"] + raw["x"]}
    client_id = f"{task}/client_{client}"
    return Update(
        f"{client_id}-r{round_index}",
        client_id,
        task,
        "dataset",
        samples,
        round_index,
        base_version,
        float(round_index + 1),
        anchor,
        local,
        subtract(local, anchor),
        42,
        "hash",
        [],
        2,
    )


class C2AFCLChecks(unittest.TestCase):
    def context(self, values: list[float], version: int = 0) -> ServerContext:
        return ServerContext({"x": torch.tensor(values)}, version, 12)

    def test_first_arrival_is_base_relative_increment_not_stale_model_interpolation(self):
        method = create_method("c2afcl")
        row = upload([2.0, 4.0], base=[1.0, 1.0])
        result = method.on_arrival(row, self.context([10.0, 20.0], 7))[0]
        torch.testing.assert_close(result.new_state["x"], torch.tensor([11.0, 22.0]))
        diag = result.metadata["c2afcl_diagnostics"][row.update_id]
        self.assertEqual(diag["staleness"], 7)
        self.assertFalse(diag["cache_hit"])
        self.assertEqual(diag["subspace_rank"], 0)

    def test_cache_is_previous_raw_same_client_task_not_previous_calibrated_update(self):
        method = create_method("c2afcl", {"server_lr": 1.0})
        server = FederatedServer(None, method, 12, {"x": torch.zeros(2)})
        for r, raw in enumerate(([1.0, 2.0], [3.0, 4.0], [6.0, 8.0])):
            server.receive(upload(list(raw), round_index=r))
        # u0 + (u1-u0) + (u2-u1) = u2 for this single-task reference sequence.
        torch.testing.assert_close(server.state["x"], torch.tensor([6.0, 8.0]))
        torch.testing.assert_close(method.client_cache[("cls/client_0", "cls")], server.state["x"])
        torch.testing.assert_close(method.task_means["cls"], torch.tensor([10 / 3, 14 / 3]))

    def test_other_clients_have_independent_caches(self):
        method = create_method("c2afcl", {"server_lr": 1.0})
        method.on_arrival(upload([3.0, 0.0]), self.context([0.0, 0.0]))
        row = upload([2.0, 0.0], client="1")
        result = method.on_arrival(row, self.context([3.0, 0.0], 1))[0]
        torch.testing.assert_close(result.new_state["x"], torch.tensor([5.0, 0.0]))
        self.assertFalse(result.metadata["c2afcl_diagnostics"][row.update_id]["cache_hit"])

    def test_projection_excludes_current_task_and_uses_only_prior_other_history(self):
        method = create_method("c2afcl", {"server_lr": 1.0})
        method.on_arrival(upload([1.0, 0.0]), self.context([0.0, 0.0]))
        row = upload([2.0, 3.0], task="vqa")
        result = method.on_arrival(row, self.context([1.0, 0.0], 1))[0]
        torch.testing.assert_close(result.new_state["x"], torch.tensor([1.0, 3.0]))
        diag = result.metadata["c2afcl_diagnostics"][row.update_id]
        self.assertEqual(diag["protected_tasks"], ["cls"])
        self.assertLess(diag["projection_residual_norm"], 1e-6)
        # The recorded VQA prototype is RAW [2,3], not the safe [0,3].
        torch.testing.assert_close(method.task_means["vqa"], torch.tensor([2.0, 3.0]))

    def test_joint_parameter_projection_matches_dense_reference(self):
        method = create_method("c2afcl", {"subspace_rank": 1})
        method.task_means = {"cls": torch.tensor([1.0, 1.0])}
        keys = ("layers.0.q_proj.lora_A.weight", "layers.0.q_proj.lora_B.weight")
        row = upload([1.0, 0.0], task="vqa")
        row.base_state = {k: torch.zeros(1, 1) for k in keys}
        row.local_state = {keys[0]: torch.ones(1, 1), keys[1]: torch.zeros(1, 1)}
        row.delta = clone_state(row.local_state)
        result = method.on_arrival(row, ServerContext(row.base_state, 0, 12))[0]
        # Joint projection [1,0] -> [.5,-.5], NOT independent A/B projections.
        torch.testing.assert_close(result.new_state[keys[0]], torch.tensor([[0.25]]))
        torch.testing.assert_close(result.new_state[keys[1]], torch.tensor([[-0.25]]))

    def test_top_d_projection_and_dependent_or_zero_columns(self):
        method = create_method("c2afcl", {"subspace_rank": 1})
        method.task_means = {
            "cls": torch.tensor([3.0, 0.0, 0.0]),
            "caption": torch.tensor([0.0, 1.0, 0.0]),
        }
        safe, info = method._project(torch.tensor([1.0, 2.0, 3.0]), "vqa")
        torch.testing.assert_close(safe, torch.tensor([0.0, 2.0, 3.0]))
        self.assertEqual(info["subspace_rank"], 1)
        method.task_means = {"cls": torch.tensor([1.0, 0.0]), "caption": torch.tensor([2.0, 0.0])}
        safe, info = method._project(torch.tensor([1.0, 2.0]), "vqa")
        torch.testing.assert_close(safe, torch.tensor([0.0, 2.0]), atol=1e-6, rtol=1e-6)
        self.assertEqual(info["subspace_rank"], 1)
        method.task_means = {"cls": torch.zeros(2)}
        safe, info = method._project(torch.ones(2), "vqa")
        torch.testing.assert_close(safe, torch.ones(2))
        self.assertEqual(info["subspace_rank"], 0)

    def test_thin_projection_matches_explicit_dense_projector(self):
        method = create_method("c2afcl", {"subspace_rank": 2})
        columns = torch.tensor(
            [
                [2.0, 0.0, 1.0],
                [0.0, 3.0, 0.0],
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 1.0],
            ]
        )
        method.task_means = {
            q: columns[:, i] for i, q in enumerate(("cls", "caption", "grounding"))
        }
        delta = torch.tensor([1.0, -2.0, 3.0, 4.0])
        safe, info = method._project(delta, "vqa")
        basis, _, _ = torch.linalg.svd(columns, full_matrices=False)
        dense_reference = (torch.eye(4) - basis[:, :2] @ basis[:, :2].T) @ delta
        torch.testing.assert_close(safe, dense_reference, atol=2e-6, rtol=2e-6)
        self.assertEqual(info["subspace_rank"], 2)

    def test_buffer_is_task_specific_equal_weight_and_flushes_all_residuals(self):
        method = create_method("c2afcl", {"server_lr": 1.0, "buffer_size": 2})
        server = FederatedServer(None, method, 12, {"x": torch.zeros(2)})
        server.receive(upload([2.0, 0.0], samples=1))
        server.receive(upload([0.0, 3.0], task="vqa"))
        self.assertEqual(server.version, 0)
        self.assertEqual(method.task_counts, {})
        result = server.receive(upload([6.0, 0.0], client="1", samples=99))[0]
        torch.testing.assert_close(server.state["x"], torch.tensor([4.0, 0.0]))
        self.assertEqual(server.accepted_updates, 2)
        self.assertEqual(result.metadata["buffer_occupancy"], 2)
        server.receive(upload([0.0, 2.0], task="grounding"))
        finished = server.finish()
        self.assertEqual(len(finished), 2)
        self.assertEqual(server.accepted_updates, 4)
        torch.testing.assert_close(server.state["x"], torch.tensor([4.0, 5.0]))
        self.assertEqual(method.buffers, {})
        self.assertEqual(method.buffer_aggregations, 3)
        self.assertEqual(server.finish(), [])

    def test_repeated_raw_update_is_zero_calibrated_no_invented_fallback(self):
        method = create_method("c2afcl")
        method.on_arrival(upload([1.0, 2.0]), self.context([0.0, 0.0]))
        row = upload([1.0, 2.0], round_index=1)
        result = method.on_arrival(row, self.context([8.0, 9.0], 1))[0]
        torch.testing.assert_close(result.new_state["x"], torch.tensor([8.0, 9.0]))
        self.assertEqual(result.metadata["c2afcl_diagnostics"][row.update_id]["safe_delta_norm"], 0)

    def test_state_roundtrip_preserves_pending_buffers_parameters_and_future_results(self):
        method = create_method("c2afcl", {"server_lr": 0.2, "buffer_size": 2})
        method.on_arrival(upload([2.0, 0.0]), self.context([0.0, 0.0]))
        restored = create_method("c2afcl")
        restored.load_state_dict(copy.deepcopy(method.state_dict()))
        row = upload([4.0, 0.0], client="1")
        expected = method.on_arrival(row, self.context([0.0, 0.0]))[0]
        actual = restored.on_arrival(row, self.context([0.0, 0.0]))[0]
        torch.testing.assert_close(actual.new_state["x"], expected.new_state["x"])
        self.assertEqual(actual.metadata, expected.metadata)
        self.assertEqual(restored.server_lr, 0.2)
        restored.client_cache[("cls/client_0", "cls")].zero_()
        self.assertTrue(bool(method.client_cache[("cls/client_0", "cls")].any()))

    def test_input_state_is_not_mutated_or_attached_to_autograd(self):
        method = create_method("c2afcl")
        row = upload([1.0, 2.0])
        row.local_state["x"].requires_grad_(True)
        before = clone_state(row.local_state)
        context = self.context([3.0, 4.0])
        result = method.on_arrival(row, context)[0]
        torch.testing.assert_close(row.local_state["x"], before["x"])
        torch.testing.assert_close(context.global_state["x"], torch.tensor([3.0, 4.0]))
        self.assertFalse(result.new_state["x"].requires_grad)
        self.assertFalse(method.client_cache[("cls/client_0", "cls")].requires_grad)

    def test_invalid_update_leaves_cache_counts_and_buffers_unchanged(self):
        for field, value in (
            ("base_version", 2),
            ("num_samples", 0),
            ("arrival_time", float("nan")),
        ):
            method = create_method("c2afcl")
            row = upload([1.0, 2.0])
            setattr(row, field, value)
            with self.assertRaises(ValueError):
                method.on_arrival(row, self.context([0.0, 0.0]))
            self.assertEqual(method.client_cache, {})
            self.assertEqual(method.buffers, {})
        method = create_method("c2afcl")
        with self.assertRaises(ValueError):
            method.on_arrival(upload([float("nan"), 1.0]), self.context([0.0, 0.0]))
        self.assertEqual(method.client_cache, {})

    def test_duplicate_and_task_change_rejected_without_double_count(self):
        method = create_method("c2afcl")
        row = upload([1.0, 2.0])
        method.on_arrival(row, self.context([0.0, 0.0]))
        with self.assertRaises(ValueError):
            method.on_arrival(row, self.context([0.0, 0.0]))
        other = upload([1.0, 2.0], task="vqa", round_index=1)
        other.client_id = row.client_id
        with self.assertRaises(ValueError):
            method.on_arrival(other, self.context([0.0, 0.0]))
        self.assertEqual(method.arrivals, 1)

    def test_destination_dtype_shapes_and_device_preserved(self):
        method = create_method("c2afcl")
        current = {"x": torch.zeros(2, dtype=torch.float16)}
        result = method.on_arrival(upload([1.0, 2.0]), ServerContext(current, 0, 12))[0]
        self.assertEqual(result.new_state["x"].dtype, torch.float16)
        self.assertEqual(result.new_state["x"].device, current["x"].device)
        with self.assertRaises(ValueError):
            method.on_arrival(upload([1.0], client="1"), ServerContext(current, 1, 12))

    def test_configuration_metadata_and_parameters(self):
        for setting in (2, 5, 10):
            folder = ROOT / "configs/experiments/llava/afvlm_cm" / f"{setting}clients"
            cfg = load_config(folder / "c2afcl.yaml")
            ref = load_config(folder / "fedavg.yaml")
            validate_config(cfg, check_paths=False)
            method = create_method("c2afcl", cfg["method"]["params"])
            for section in ("training", "runtime", "dataset", "model", "evaluation"):
                self.assertEqual(cfg[section], ref[section])
            self.assertEqual(cfg["federation"]["train_plan"], ref["federation"]["train_plan"])
            self.assertEqual(resolve_runtime_backend(cfg, method), "client_ddp")
        method.configure_server({"x": torch.zeros(2)}, [SimpleNamespace(id="fixed", task="cls")])
        self.assertEqual(method.client_tasks["fixed"], "cls")
        for params in (
            {"server_lr": 0},
            {"server_lr": float("nan")},
            {"subspace_rank": 0},
            {"buffer_size": 1.5},
            {"buffer_size": True},
            {"svd_rtol": 1},
            {"eps": 0},
            {"invented_weight": 1},
        ):
            with self.assertRaises(ValueError):
                create_method("c2afcl", params)
        training = TrainConfig(1, 4, 4, 2e-5, 2048, 42)
        self.assertIs(method.configure_local_training(training), training)


if __name__ == "__main__":
    unittest.main()
