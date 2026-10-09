from __future__ import annotations

import copy
import math
import unittest

from afl_vlm.federation.types import ClientSpec, ServerContext, Update
from afl_vlm.methods.ours import (
    Ours,
    _add_local_increment,
    _gate_direction,
    _normalize_module_sensitivity,
    compute_functional_staleness,
    lora_functional_distance_sq,
    staleness_reliability,
)

MODULE = "base_model.model.layers.0.self_attn.q_proj"
A = f"{MODULE}.lora_A.default.weight"
B = f"{MODULE}.lora_B.default.weight"


def state(a: float, b: float | None = None) -> dict[str, list[list[float]]]:
    b = a if b is None else b
    return {
        A: [[a, a], [a, a]],
        B: [[b, b], [b, b]],
    }


def subtract(
    left: dict[str, list[list[float]]],
    right: dict[str, list[list[float]]],
) -> dict[str, list[list[float]]]:
    return {
        key: [
            [value - right[key][row][column] for column, value in enumerate(values)]
            for row, values in enumerate(matrix)
        ]
        for key, matrix in left.items()
    }


def metadata(scaling: float = 1.0) -> dict[str, dict[str, object]]:
    return {
        MODULE: {
            "module_name": MODULE,
            "a_name": A,
            "b_name": B,
            "rank": 2,
            "in_features": 2,
            "out_features": 2,
            "scaling": scaling,
        }
    }


def client(client_id: str, task: str) -> ClientSpec:
    return ClientSpec(client_id, task, task, 4, 1.0, 0.0, 0.0, 1.0)


def update(
    local_a: float = 1.0,
    local_b: float | None = None,
    base_a: float = 0.0,
    base_b: float | None = None,
    sensitivity: tuple[float, float] = (1.0, 1.0),
    task: str = "vqa",
    client_id: str | None = None,
    update_id: str | None = None,
) -> Update:
    base = state(base_a, base_b)
    local = state(local_a, local_b)
    client_id = client_id or f"{task}/client_0"
    update_id = update_id or f"{client_id}-r0"
    return Update(
        update_id=update_id,
        client_id=client_id,
        task=task,
        dataset=task,
        num_samples=4,
        local_round=0,
        base_version=0,
        arrival_time=1.0,
        base_state=base,
        local_state=local,
        delta=subtract(local, base),
        seed=42,
        sample_ids_hash="hash",
        losses=[1.0],
        optimizer_steps=2,
        metadata={
            "module_sensitivity": {MODULE: sum(sensitivity) / len(sensitivity)},
            "sensitivity_valid": True,
            "sensitivity_observations": {MODULE: 2},
            "lora_metadata": metadata(),
        },
    )


class ModuleFunctionalDistanceTests(unittest.TestCase):
    def test_gate_direction_uses_signed_over_absolute_sum(self) -> None:
        self.assertAlmostEqual(_gate_direction(-2.0, 4.0, 1e-12), 0.5)
        self.assertAlmostEqual(_gate_direction(2.0, 4.0, 1e-12), -0.5)

    def test_module_sensitivity_normalization_preserves_mean_and_floor(self) -> None:
        values, valid = _normalize_module_sensitivity(
            {"q": 0.01, "v": 1.99}, uniform_mix=0.5
        )

        self.assertTrue(valid)
        self.assertAlmostEqual(sum(values.values()) / len(values), 1.0)
        self.assertGreaterEqual(min(values.values()), 0.5)

    def test_zero_module_sensitivity_falls_back_to_uniform(self) -> None:
        values, valid = _normalize_module_sensitivity({"q": 0.0, "v": 0.0})

        self.assertFalse(valid)
        self.assertEqual(values, {"q": 1.0, "v": 1.0})

    def test_reliability_uses_exponential_relative_staleness(self) -> None:
        self.assertAlmostEqual(staleness_reliability(1.0, gamma=1.0), math.exp(-1.0))

    def test_module_distance_includes_cross_rank_terms(self) -> None:
        distance = lora_functional_distance_sq(
            [[1.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            1.0,
        )
        self.assertAlmostEqual(distance, 16.0)

    def test_functional_staleness_is_module_weighted_and_size_normalized(self) -> None:
        drift, local, relative = compute_functional_staleness(
            state(0.0),
            state(2.0),
            state(1.0),
            {MODULE: 1.0},
            metadata(),
        )
        self.assertAlmostEqual(drift, 64.0)
        self.assertAlmostEqual(local, 4.0)
        self.assertAlmostEqual(relative, 4.0)


class OursMethodTests(unittest.TestCase):
    def configured(self, **params: object) -> Ours:
        method = Ours(params)
        method.configure_server(
            state(0.0),
            [client("vqa/client_0", "vqa"), client("cls/client_0", "cls")],
        )
        return method

    def test_fresh_update_uses_one_module_alpha_for_a_and_b(self) -> None:
        method = self.configured()
        mutation = method.on_arrival(update(), ServerContext(state(0.0), 0, 2))[0]

        self.assertAlmostEqual(mutation.applied_weight, 0.5)
        for parameter in (A, B):
            for row in mutation.new_state[parameter]:
                for value in row:
                    self.assertAlmostEqual(value, 0.5)
        self.assertEqual(mutation.metadata["relative_staleness"], 0.0)
        self.assertAlmostEqual(mutation.metadata["module_alphas"][MODULE], 0.5)
        self.assertAlmostEqual(method.task_memory["vqa"][MODULE], 0.1)
        self.assertEqual(method.task_memory["cls"], {MODULE: 0.0})
        diagnostics = mutation.metadata["ours_diagnostics"]
        self.assertEqual(diagnostics["schema_version"], 4)
        self.assertEqual(diagnostics["method_variant"], "module_gate")
        self.assertTrue(diagnostics["aggregation"]["accepted"])
        self.assertEqual(diagnostics["aggregation"]["rule"], "base_relative_delta")
        self.assertEqual(diagnostics["sensitivity"]["module_by_module"][MODULE], 1.0)
        self.assertEqual(
            diagnostics["sensitivity"]["observations_by_module"][MODULE], 2
        )
        self.assertAlmostEqual(
            diagnostics["memory"]["task_memory_before"][MODULE], 0.0
        )
        self.assertAlmostEqual(
            diagnostics["memory"]["task_memory_raw_after"][MODULE], 0.1
        )
        self.assertAlmostEqual(diagnostics["memory"]["task_memory_after"][MODULE], 1.0)
        self.assertEqual(diagnostics["memory"]["task_memory_count_after"], 1)
        self.assertAlmostEqual(
            diagnostics["memory"]["historical_precision_after"][MODULE], 1.5
        )
        self.assertEqual(diagnostics["frequency"]["window_length_before"], 0)
        self.assertEqual(diagnostics["frequency"]["window_length_after"], 1)
        self.assertAlmostEqual(diagnostics["frequency"]["weight"], 1.0)

    def test_functional_staleness_reduces_reliability_and_module_alpha(self) -> None:
        fresh = self.configured()
        stale = self.configured()
        fresh_mutation = fresh.on_arrival(update(), ServerContext(state(0.0), 0, 2))[0]
        stale_mutation = stale.on_arrival(update(), ServerContext(state(2.0), 3, 2))[0]

        self.assertAlmostEqual(stale_mutation.metadata["relative_staleness"], 4.0)
        self.assertAlmostEqual(stale_mutation.metadata["reliability"], math.exp(-4.0))
        self.assertLess(stale_mutation.applied_weight, fresh_mutation.applied_weight)

    def test_stale_update_adds_base_relative_increment_without_server_rollback(self) -> None:
        method = self.configured()
        incoming = update(local_a=1.2, local_b=0.8, base_a=1.0, base_b=1.0)
        current = state(2.0, 0.55)
        before = copy.deepcopy((current, incoming.base_state, incoming.local_state, incoming.delta))

        mutation = method.on_arrival(incoming, ServerContext(current, 3, 2))[0]
        alpha = mutation.metadata["module_alphas"][MODULE]

        self.assertGreater(alpha, 0.0)
        self.assertEqual(mutation.metadata["aggregation_rule"], "base_relative_delta")
        for name, initial, increment in ((A, 2.0, 0.2), (B, 0.55, -0.2)):
            for row in mutation.new_state[name]:
                for value in row:
                    self.assertAlmostEqual(value, initial + alpha * increment)
        self.assertGreater(mutation.new_state[A][0][0], current[A][0][0])
        self.assertNotAlmostEqual(mutation.new_state[A][0][0], 2.0 + alpha * (1.2 - 2.0))
        self.assertEqual(
            (current, incoming.base_state, incoming.local_state, incoming.delta), before
        )

    def test_fresh_nonzero_base_is_equivalent_to_endpoint_interpolation(self) -> None:
        method = self.configured()
        incoming = update(local_a=1.2, local_b=0.8, base_a=1.0, base_b=1.0)
        current = state(1.0)
        mutation = method.on_arrival(incoming, ServerContext(current, 0, 2))[0]
        alpha = mutation.metadata["module_alphas"][MODULE]

        for name in (A, B):
            for row, values in enumerate(mutation.new_state[name]):
                for column, value in enumerate(values):
                    expected = (
                        (1.0 - alpha) * current[name][row][column]
                        + alpha * incoming.local_state[name][row][column]
                    )
                    self.assertAlmostEqual(value, expected)

    def test_each_module_increment_uses_its_own_shared_a_b_alpha(self) -> None:
        method = self.configured()
        other = "base_model.model.layers.1.self_attn.v_proj"
        other_a = f"{other}.lora_A.default.weight"
        other_b = f"{other}.lora_B.default.weight"
        method._server_metadata[other] = {
            **metadata()[MODULE], "a_name": other_a, "b_name": other_b
        }
        current = {**state(3.0), other_a: [[5.0, 5.0]] * 2, other_b: [[5.0, 5.0]] * 2}
        base = {**state(1.0), other_a: [[2.0, 2.0]] * 2, other_b: [[2.0, 2.0]] * 2}
        local = {**state(2.0), other_a: [[4.0, 4.0]] * 2, other_b: [[4.0, 4.0]] * 2}
        result = method._fuse_modules(current, local, base, {MODULE: 0.2, other: 0.4})

        for name in (A, B):
            self.assertEqual(result[name], [[3.2, 3.2], [3.2, 3.2]])
        for name in (other_a, other_b):
            self.assertEqual(result[name], [[5.8, 5.8], [5.8, 5.8]])

    def test_increment_rejects_mismatched_base_keys_and_shapes(self) -> None:
        method = self.configured()
        with self.assertRaisesRegex(ValueError, "state keys differ"):
            method._fuse_modules(state(2.0), state(1.0), {A: [[0.0]]}, {MODULE: 0.5})
        with self.assertRaisesRegex(ValueError, "tensor shapes differ"):
            _add_local_increment([[2.0, 2.0]], [[1.0, 1.0]], [[0.0]], 0.5)

    def test_history_memory_uses_sensitivity_without_reliability(self) -> None:
        method = self.configured()
        mutation = method.on_arrival(update(), ServerContext(state(2.0), 3, 2))[0]

        self.assertLess(mutation.metadata["reliability"], 0.1)
        self.assertAlmostEqual(method.task_memory["vqa"][MODULE], 0.1)
        self.assertAlmostEqual(
            mutation.metadata["ours_diagnostics"]["memory"]["task_memory_after"][MODULE],
            1.0,
        )

    def test_frequency_weight_uses_pre_update_success_window(self) -> None:
        method = self.configured(alpha_max=1.0)
        method.task_frequency_window = ["vqa"] * 18
        mutation = method.on_arrival(
            update(task="cls"), ServerContext(state(0.0), 0, 2)
        )[0]
        frequency = mutation.metadata["ours_diagnostics"]["frequency"]

        self.assertAlmostEqual(frequency["raw_weight"], math.sqrt(10.0))
        self.assertAlmostEqual(frequency["weight"], 1.5)
        self.assertEqual(frequency["window_length_before"], 18)
        self.assertEqual(frequency["counts_before"], {"cls": 0, "vqa": 18})
        self.assertEqual(frequency["window_length_after"], 18)
        self.assertEqual(frequency["counts_after"], {"cls": 1, "vqa": 17})
        self.assertAlmostEqual(mutation.metadata["module_alphas"][MODULE], 0.6)

    def test_rejected_update_does_not_enter_frequency_window(self) -> None:
        method = self.configured()
        method.task_frequency_window = ["vqa"]
        mutation = method.on_arrival(
            update(local_a=0.0, task="cls"), ServerContext(state(0.0), 0, 2)
        )[0]

        self.assertFalse(mutation.increment_version)
        self.assertEqual(method.task_frequency_window, ["vqa"])
        frequency = mutation.metadata["ours_diagnostics"]["frequency"]
        self.assertEqual(frequency["counts_before"], frequency["counts_after"])

    def test_no_effective_update_is_rejected_without_advancing_version(self) -> None:
        method = self.configured()
        mutation = method.on_arrival(update(local_a=0.0), ServerContext(state(0.0), 0, 2))[0]

        self.assertEqual(mutation.applied_weight, 0.0)
        self.assertFalse(mutation.increment_version)
        self.assertEqual(mutation.metadata["rejected"], "no_effective_update")
        diagnostics = mutation.metadata["ours_diagnostics"]
        self.assertFalse(diagnostics["aggregation"]["accepted"])
        self.assertEqual(diagnostics["aggregation"]["rule"], "base_relative_delta")
        self.assertEqual(
            diagnostics["aggregation"]["rejection_reason"], "no_effective_update"
        )
        self.assertEqual(
            diagnostics["memory"]["task_memory_before"],
            diagnostics["memory"]["task_memory_after"],
        )

    def test_sensitivity_observations_are_validated(self) -> None:
        method = self.configured()
        changed = update()
        changed.metadata["sensitivity_observations"] = {MODULE: -1}

        with self.assertRaisesRegex(ValueError, "observation count"):
            method.prepare_upload(changed, None)

    def test_functionally_equal_but_parameter_changed_update_is_rejected(self) -> None:
        method = self.configured()
        # BA is unchanged: (0.5 B) @ (2 A) == B @ A, but A/B moved.
        mutation = method.on_arrival(
            update(local_a=2.0, local_b=0.5, base_a=1.0, base_b=1.0),
            ServerContext(state(1.0), 0, 2),
        )[0]

        self.assertFalse(mutation.increment_version)
        self.assertEqual(mutation.metadata["rejected"], "no_effective_update")
        self.assertAlmostEqual(mutation.metadata["local_update"], 0.0)

    def test_state_round_trip_preserves_memory_weights_and_metadata(self) -> None:
        method = self.configured()
        method.on_arrival(update(), ServerContext(state(0.0), 0, 2))
        restored = Ours()
        restored.load_state_dict(method.state_dict())

        self.assertEqual(restored.task_memory, method.task_memory)
        self.assertEqual(restored.task_memory_counts, method.task_memory_counts)
        self.assertEqual(restored.task_weights, method.task_weights)
        self.assertEqual(restored.task_frequency_window, method.task_frequency_window)
        self.assertEqual(restored._server_metadata, method._server_metadata)

    def test_rank_gate_checkpoint_is_rejected(self) -> None:
        method = self.configured()
        saved = method.state_dict()
        saved["state_schema_version"] = 2
        saved.pop("method_variant")

        with self.assertRaisesRegex(ValueError, "not a Module-Gate"):
            Ours().load_state_dict(saved)

    def test_history_strength_only_scales_corrected_task_memory(self) -> None:
        method = self.configured(history_strength=2.0)
        mutation = method.on_arrival(update(), ServerContext(state(0.0), 0, 2))[0]

        diagnostics = mutation.metadata["ours_diagnostics"]
        self.assertAlmostEqual(
            diagnostics["memory"]["historical_precision_after"][MODULE], 2.0
        )

    def test_invalid_module_sensitivity_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "uniform_mix"):
            Ours({"sensitivity_uniform_mix": 1.1})
        with self.assertRaisesRegex(ValueError, "frequency_window_size"):
            Ours({"frequency_window_size": 0})
        with self.assertRaisesRegex(ValueError, "frequency bounds"):
            Ours({"frequency_min": 2.0, "frequency_max": 1.0})
        with self.assertRaisesRegex(ValueError, "Unknown parameter"):
            Ours({"frequency_tau": 0.5})

    def test_client_scaling_must_match_first_validated_update(self) -> None:
        method = self.configured()
        method.prepare_upload(update(), None)
        changed = update()
        changed.metadata["lora_metadata"] = metadata(scaling=2.0)
        with self.assertRaisesRegex(ValueError, "scaling or shape differs"):
            method.prepare_upload(changed, None)

    def test_non_lora_trainable_state_is_rejected(self) -> None:
        method = Ours()
        with self.assertRaisesRegex(ValueError, "LoRA-only"):
            method.configure_server({"mm_projector.weight": [[0.0]]}, [client("c", "vqa")])


if __name__ == "__main__":
    unittest.main()
