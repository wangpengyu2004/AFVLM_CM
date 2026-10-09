"""Non-training matrix/state contract checks; no model, forward or dataset job."""

from __future__ import annotations

import copy
import unittest

from afl_vlm.execution import resolve_runtime_backend
from afl_vlm.federation.types import ServerContext, Update
from afl_vlm.methods.continual_merge.algebra import (
    compact_coordinates,
    lora_pairs,
    opcm_filter,
    projection_loss,
    refactor,
    spectrum,
)
from afl_vlm.methods.registry import create_method

try:
    import torch
except ImportError:
    torch = None

A = "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
B = "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
NAMES = ("opcm_lora", "dop_lora", "nufilt_lora")


class RegistryContractTests(unittest.TestCase):
    def test_capabilities_and_parameter_validation(self) -> None:
        for name in NAMES:
            method = create_method(name, {})
            self.assertEqual(method.evaluation_scope, "server_global")
            self.assertEqual(
                resolve_runtime_backend(
                    {"runtime": {"backend": "client_parallel", "executor_policy": "capability"}},
                    method,
                ),
                "client_ddp",
            )
            self.assertFalse(method.capabilities.requires_custom_adapter)
            self.assertFalse(method.capabilities.requires_local_step_control)
            with self.assertRaises(ValueError):
                create_method(name, {"merge_weight": 2})
            with self.assertRaises(ValueError):
                create_method(name, {"unexpected": 1})
        for name in NAMES[1:]:
            with self.assertRaises(ValueError):
                create_method(name, {"inner_steps": 0})

    def test_no_frozen_or_incomplete_keys_accepted(self) -> None:
        self.assertEqual(
            lora_pairs({A: None, B: None}),
            {"base_model.model.layers.0.self_attn.q_proj:default": (A, B)},
        )
        for invalid in ({A: None}, {"backbone.weight": None}, {}):
            with self.assertRaises(ValueError):
                lora_pairs(invalid)


@unittest.skipIf(torch is None, "PyTorch required for matrix algebra checks")
class FunctionalAlgebraTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = torch.Generator().manual_seed(31)

    def matrix(self, *shape: int):
        return torch.randn(*shape, generator=self.generator)

    def test_compact_products_and_download_relative_increment(self) -> None:
        factors = [(self.matrix(2, 7), self.matrix(8, 2)) for _ in range(3)]
        qo, qi, cores = compact_coordinates(factors, "cpu")
        for (a, b), core in zip(factors, cores, strict=True):
            torch.testing.assert_close(qo @ core @ qi.T, b @ a, rtol=2e-5, atol=2e-5)
        expected = factors[1][1] @ factors[1][0] - factors[2][1] @ factors[2][0]
        torch.testing.assert_close(
            qo @ (cores[1] - cores[2]) @ qi.T, expected, rtol=2e-5, atol=2e-5
        )

    def test_refactor_matches_dense_optimal_rank_approximation(self) -> None:
        factors = [(self.matrix(2, 7), self.matrix(8, 2)) for _ in range(3)]
        qo, qi, cores = compact_coordinates(factors, "cpu")
        core = cores[0] + cores[1] - cores[2]
        a, b, info = refactor(core, qo, qi, 2, *factors[0], 1e-8)
        dense = qo @ core @ qi.T
        u, s, vh = torch.linalg.svd(dense, full_matrices=False)
        expected = (u[:, :2] * s[:2]) @ vh[:2]
        torch.testing.assert_close(b @ a, expected, rtol=3e-5, atol=3e-5)
        self.assertAlmostEqual(
            info["compression_relative_error"],
            float((dense - b @ a).norm() / dense.norm()),
            places=5,
        )
        az, bz, _ = refactor(torch.zeros_like(core), qo, qi, 2, *factors[0], 1e-8)
        torch.testing.assert_close(az, factors[0][0])
        self.assertEqual(int(torch.count_nonzero(bz)), 0)

    def test_dop_projection_loss_is_coordinate_exact(self) -> None:
        qo = torch.linalg.qr(self.matrix(8, 4))[0]
        qi = torch.linalg.qr(self.matrix(7, 4))[0]
        old, new = self.matrix(4, 4), self.matrix(4, 4)
        u, s, vh = torch.linalg.svd(old, full_matrices=False)
        compact = projection_loss(new, old, u, s, vh.T)
        difference = qo @ (new - old) @ qi.T
        dense = (s[:, None] * ((qo @ u).T @ difference)).square().sum() + (
            (difference @ (qi @ vh.T)) * s
        ).square().sum()
        torch.testing.assert_close(compact, dense, rtol=2e-5, atol=2e-5)

    def test_spectrum_ignores_fp32_null_space_roundoff(self) -> None:
        qo = torch.linalg.qr(self.matrix(24, 24))[0]
        qi = torch.linalg.qr(self.matrix(24, 24))[0]
        values = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05] + [0.0] * 16)
        core = (qo * values) @ qi.T
        u, s, v = spectrum(core)
        self.assertEqual(len(s), 8)
        torch.testing.assert_close((u * s) @ v.T, core, atol=1e-6, rtol=1e-5)

    def test_opcm_removes_block_and_diagonal_not_all_right_overlap(self) -> None:
        old = torch.diag(torch.tensor([4.0, 2.0, 1.0, 0.0]))
        incoming = torch.ones(4, 4)
        clean, rank = opcm_filter(incoming, old, 0.5, 1e-8)
        self.assertEqual(rank, 1)
        for i in range(3):
            self.assertEqual(float(clean[i, i]), 0)
        self.assertEqual(float(clean[0, 1]), 1)
        self.assertEqual(float(clean[3, 3]), 1)  # null completion not spuriously filtered

    def state(self, amount: float = 0):
        return {
            A: torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            B: torch.tensor([[amount, 0.0], [0.0, amount], [0.0, 0.0], [0.0, 0.0]]),
        }

    def update(self, base, local, seed: int = 42):
        return Update(
            "u",
            "vqa/client_0",
            "vqa",
            "AOKVQA",
            500,
            0,
            0,
            9.0,
            base,
            local,
            {key: local[key] - base[key] for key in base},
            seed,
            "unchanged",
            [],
            17,
        )

    def test_arrivals_do_not_mutate_input_or_change_state_contract(self) -> None:
        for name in NAMES:
            params = {} if name == NAMES[0] else {"inner_steps": 3}
            method = create_method(name, params)
            base, local, current = self.state(), self.state(0.1), self.state(0.2)
            saved = copy.deepcopy((base, local, current))
            method.configure_server(base, [])
            result = method.on_arrival(self.update(base, local), ServerContext(current, 5, 12))[0]
            self.assertEqual(set(result.new_state), {A, B})
            self.assertEqual(result.metadata["staleness"], 5)
            self.assertEqual(result.contributing_update_ids, ["u"])
            for before, after in zip(saved, (base, local, current), strict=True):
                for key in before:
                    torch.testing.assert_close(before[key], after[key])
            for key, value in result.new_state.items():
                self.assertEqual(value.shape, current[key].shape)
                self.assertFalse(value.requires_grad)
                self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertEqual(result.metadata["merge_count"], 1)
            self.assertEqual(self.update(base, local).optimizer_steps, 17)

    def test_zero_increment_and_state_roundtrip(self) -> None:
        for name in NAMES:
            params = {} if name == NAMES[0] else {"inner_steps": 2}
            method = create_method(name, params)
            base, local = self.state(), self.state(0.1)
            method.configure_server(base, [])
            first = method.on_arrival(self.update(base, local), ServerContext(base, 0, 12))[0]
            restored = create_method(name, {})
            restored.load_state_dict(method.state_dict())
            context = ServerContext(first.new_state, 1, 12)
            next_update = self.update(base, local)
            expected = method.on_arrival(next_update, context)[0]
            actual = restored.on_arrival(next_update, context)[0]
            for key in expected.new_state:
                torch.testing.assert_close(expected.new_state[key], actual.new_state[key])
            zero = restored.on_arrival(self.update(base, base), context)[0]
            for key in context.global_state:
                torch.testing.assert_close(zero.new_state[key], context.global_state[key])

    def test_partial_zero_increment_preserves_other_module_coordinates(self) -> None:
        a_other, b_other = A.replace("q_proj", "k_proj"), B.replace("q_proj", "k_proj")
        for name in NAMES[1:]:
            method = create_method(name, {"inner_steps": 2})
            current, base, local = self.state(0.2), self.state(), self.state(0.1)
            for state in (current, base, local):
                state[a_other] = self.state()[A] * 3
                state[b_other] = self.state(0.1)[B] / 3
            result = method.on_arrival(self.update(base, local), ServerContext(current, 5, 12))[0]
            for key in (a_other, b_other):
                self.assertTrue(torch.equal(result.new_state[key], current[key]))

    def test_opcm_partial_zero_module_uses_previous_global_scale(self) -> None:
        method = create_method("opcm_lora", {})
        method.opcm_scale, method.arrivals, method.norm_sum = 2.0, 1, 1.0
        a_other, b_other = A.replace("q_proj", "k_proj"), B.replace("q_proj", "k_proj")
        current, base, local = self.state(0.2), self.state(), self.state(0.1)
        for state in (current, base, local):
            state[a_other], state[b_other] = self.state()[A], self.state(0.3)[B]
        old_product = current[b_other] @ current[a_other]
        result = method.on_arrival(self.update(base, local), ServerContext(current, 5, 12))[0]
        scale = result.metadata["opcm_scale"]
        expected = old_product * (0.5 + 0.5 * 2.0 / scale)
        actual = result.new_state[b_other] @ result.new_state[a_other]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_server_lifecycle_counts_versions_and_acceptances(self) -> None:
        from afl_vlm.federation.server import FederatedServer

        for name in NAMES:
            params = {} if name == NAMES[0] else {"inner_steps": 2}
            method = create_method(name, params)
            base, local = self.state(), self.state(0.1)
            method.configure_server(base, [])
            server = FederatedServer(None, method, 12, initial_state=base)
            for number in range(3):
                update = self.update(base, local)
                update.update_id = f"u{number}"
                result = server.receive(update)[0]
                self.assertTrue(result.applied)
                self.assertEqual(result.metadata["staleness"], number)
                self.assertEqual(result.contributing_update_ids, [f"u{number}"])
            self.assertEqual(server.version, 3)
            self.assertEqual(server.accepted_updates, 3)
            self.assertEqual(server.received_updates, 3)
            self.assertEqual(server.finish(), [])

    def test_nufilt_surrogate_keeps_filter_and_residual_components(self) -> None:
        method = create_method("nufilt_lora", {"inner_steps": 8, "inner_lr": 0.01})
        old = torch.diag(torch.tensor([1.0, 0.0, 0.0]))
        incoming = torch.diag(torch.tensor([0.0, 0.2, 0.0]))
        merged, details = method.merge_core(old, incoming, 9)
        self.assertEqual(details["protected_rank"], 1)
        self.assertEqual(details["new_rank"], 2)
        self.assertTrue(bool(torch.isfinite(merged).all()))
        torch.testing.assert_close(merged, old + incoming, atol=1e-6, rtol=1e-6)

    def test_reject_wrong_origin_shape_and_nonfinite(self) -> None:
        method = create_method("opcm_lora", {})
        with self.assertRaises(ValueError):
            method.configure_server(self.state(0.1), [])
        with self.assertRaises(ValueError):
            compact_coordinates([(torch.zeros(2, 4), torch.zeros(3, 3))], "cpu")
        bad = self.state()
        bad[A][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            method.configure_server(bad, [])


if __name__ == "__main__":
    unittest.main()
