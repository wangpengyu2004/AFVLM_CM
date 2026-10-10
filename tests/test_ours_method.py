from __future__ import annotations

import copy
import math
import unittest

from afl_vlm.federation.server import FederatedServer
from afl_vlm.federation.types import ClientSpec, ScheduledEvent, ServerContext, Update
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


class TemporalHistoryTests(unittest.TestCase):
    def configured(
        self,
        jobs: list[tuple[str, str, int]] | None = None,
        **params: object,
    ) -> Ours:
        method = Ours({
            "history_age_enabled": True,
            "history_age_boost": 1.0,
            "history_age_time_scale": 2.0,
            **params,
        })
        method.configure_server(state(0.0), [
            client("vqa/client_0", "vqa"),
            client("vqa/client_1", "vqa"),
            client("cls/client_0", "cls"),
        ])
        jobs = jobs if jobs is not None else [
            ("vqa", "vqa/client_0", 0),
            ("cls", "cls/client_0", 0),
            ("cls", "cls/client_0", 1),
            ("cls", "cls/client_0", 2),
        ]
        method.configure_schedule([
            ScheduledEvent(
                event_id=index, client_id=client_id, task=task, dataset=task,
                local_round=local_round, start_time=float(index),
                arrival_time=float(index + 1), speed_factor=1.0,
                estimated_train_time=1.0, network_delay=0.0, local_steps=1,
            )
            for index, (task, client_id, local_round) in enumerate(jobs)
        ])
        return method

    def incoming(self, task: str, local_round: int = 0, **kwargs: object) -> Update:
        incoming = update(task=task, **kwargs)
        incoming.local_round = local_round
        incoming.update_id = f"{incoming.client_id}-r{local_round}"
        return incoming

    def test_retirement_waits_for_all_clients_and_out_of_order_rounds(self) -> None:
        method = self.configured([
            ("vqa", "vqa/client_0", 0),
            ("vqa", "vqa/client_0", 1),
            ("vqa", "vqa/client_1", 0),
            ("cls", "cls/client_0", 0),
        ])
        server = FederatedServer(None, method, 3, initial_state=state(0.0))
        server.receive(self.incoming("vqa", 1))
        self.assertNotIn("vqa", method.task_finished_at)
        server.receive(self.incoming("vqa", client_id="vqa/client_1"))
        self.assertEqual(method.task_pending_updates["vqa"], 1)
        self.assertEqual(method._task_history_age_weight("vqa"), 1.0)
        server.receive(self.incoming("vqa"))
        self.assertEqual(method.task_finished_at["vqa"], server.accepted_updates)
        self.assertEqual(method._task_history_age_weight("vqa"), 1.0)
        self.assertEqual(method.task_pending_updates["cls"], 1)

    def test_only_completed_task_weight_grows_with_successful_update_age(self) -> None:
        method = self.configured()
        server = FederatedServer(None, method, 3, initial_state=state(0.0))
        server.receive(self.incoming("vqa"))
        vqa_memory = copy.deepcopy(method.task_memory["vqa"])
        first = server.receive(self.incoming("cls"))[0].metadata["ours_diagnostics"]
        self.assertEqual(first["memory"]["history_age_before"]["weight_by_task"]["vqa"], 1.0)
        expected = 1.0 + (1.0 - math.exp(-0.5))
        self.assertAlmostEqual(method._task_history_age_weight("vqa"), expected)
        self.assertEqual(method._task_history_age_weight("cls"), 1.0)
        second = server.receive(self.incoming("cls", 1))[0].metadata["ours_diagnostics"]
        self.assertAlmostEqual(
            second["memory"]["history_age_before"]["weight_by_task"]["vqa"], expected
        )
        self.assertEqual(method.task_memory["vqa"], vqa_memory)
        self.assertEqual(method.task_weights, {"cls": 0.5, "vqa": 0.5})
        self.assertEqual(method.history_accepted_updates, server.accepted_updates)
        self.assertGreater(
            second["memory"]["historical_precision_before"][MODULE],
            second["memory"]["historical_precision_unweighted_before"][MODULE],
        )

    def test_weighted_precision_does_not_renormalize_or_scale_base(self) -> None:
        method = self.configured(history_strength=2.0, base_precision=3.0)
        method.task_memory = {"vqa": {MODULE: 0.2}, "cls": {MODULE: 0.4}}
        method.task_memory_counts = {"vqa": 1, "cls": 1}
        method.task_finished_at = {"vqa": 0}
        method.history_accepted_updates = 2
        weight = 1.0 + (1.0 - math.exp(-1.0))
        self.assertAlmostEqual(method._historical_precision(MODULE), 3.0 + 2.0 * (weight + 2.0))
        self.assertAlmostEqual(method._historical_precision(MODULE, apply_age_weight=False), 9.0)
        method.history_accepted_updates = 10**9
        self.assertAlmostEqual(method._task_history_age_weight("vqa"), 2.0)

    def test_rejected_terminal_job_retires_task_without_advancing_clock(self) -> None:
        method = self.configured()
        server = FederatedServer(None, method, 3, initial_state=state(0.0))
        rejected = server.receive(self.incoming("vqa", local_a=0.0))[0]
        self.assertEqual(method.task_finished_at["vqa"], 0)
        self.assertEqual(method.history_accepted_updates, 0)
        self.assertEqual(method.task_memory_counts["vqa"], 0)
        self.assertEqual(method.task_frequency_window, [])
        self.assertEqual(
            rejected.metadata["ours_diagnostics"]["memory"]["history_age_after"]["finished_at"],
            {"vqa": 0},
        )
        server.receive(self.incoming("cls"))
        age = method.history_accepted_updates
        precision = method._historical_precision(MODULE)
        server.receive(self.incoming("cls", 1, local_a=0.0))
        self.assertEqual(method.history_accepted_updates, age)
        self.assertEqual(method.history_accepted_updates, server.accepted_updates)
        self.assertAlmostEqual(method._historical_precision(MODULE), precision)

    def test_disabled_and_zero_boost_match_original_alpha_memory_and_frequency(self) -> None:
        original = self.configured(history_age_enabled=False)
        zero_boost = self.configured(history_age_boost=0.0)
        original_server = FederatedServer(None, original, 3, initial_state=state(0.0))
        zero_server = FederatedServer(None, zero_boost, 3, initial_state=state(0.0))
        for task, local_round in (("vqa", 0), ("cls", 0), ("cls", 1), ("cls", 2)):
            left = original_server.receive(self.incoming(task, local_round))[0]
            right = zero_server.receive(self.incoming(task, local_round))[0]
            self.assertEqual(left.applied_weight, right.applied_weight)
            self.assertEqual(original_server.state, zero_server.state)
            self.assertEqual(original.task_memory, zero_boost.task_memory)
            self.assertEqual(original.task_frequency_window, zero_boost.task_frequency_window)

    def test_temporal_state_round_trip_preserves_next_aggregation(self) -> None:
        method = self.configured()
        server = FederatedServer(None, method, 3, initial_state=state(0.0))
        server.receive(self.incoming("vqa"))
        server.receive(self.incoming("cls"))
        saved = method.state_dict()
        restored = Ours()
        restored.load_state_dict(saved)
        self.assertEqual(restored.state_dict(), saved)
        self.assertEqual(restored.task_pending_updates, method.task_pending_updates)
        left = method.on_arrival(self.incoming("cls", 1), server.context())[0]
        right = restored.on_arrival(self.incoming("cls", 1), server.context())[0]
        self.assertEqual(left.new_state, right.new_state)
        self.assertEqual(left.metadata, right.metadata)

    def test_legacy_checkpoint_keeps_age_disabled_and_missing_enabled_state_fails(self) -> None:
        method = self.configured(history_age_enabled=False)
        saved = method.state_dict()
        saved.pop("history_age_state")
        saved["params"].pop("history_age_enabled")
        restored = Ours()
        restored.load_state_dict(saved)
        self.assertEqual(restored._task_history_age_weight("vqa"), 1.0)
        saved["params"]["history_age_enabled"] = True
        with self.assertRaisesRegex(ValueError, "missing history_age_state"):
            Ours().load_state_dict(saved)

    def test_enabled_requires_exact_plan_and_refuses_duplicate_arrivals(self) -> None:
        method = self.configured()
        with self.assertRaisesRegex(ValueError, "absent"):
            method.on_arrival(self.incoming("cls", 99), ServerContext(state(0.0), 0, 3))
        method.on_arrival(self.incoming("vqa"), ServerContext(state(0.0), 0, 3))
        with self.assertRaisesRegex(ValueError, "already resolved"):
            method.on_arrival(self.incoming("vqa"), ServerContext(state(0.0), 1, 3))
        method.history_planned_jobs = {}
        with self.assertRaisesRegex(RuntimeError, "configure_schedule"):
            method.on_arrival(self.incoming("cls"), ServerContext(state(0.0), 1, 3))

    def test_invalid_temporal_parameters_and_checkpoint_are_rejected(self) -> None:
        for key, value in (
            ("history_age_enabled", "true"),
            ("history_age_boost", -1),
            ("history_age_time_scale", 0),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                Ours({key: value})
        method = self.configured()
        method.on_arrival(self.incoming("vqa"), ServerContext(state(0.0), 0, 3))
        saved = method.state_dict()
        saved["history_age_state"]["finished_at"]["vqa"] = 2
        with self.assertRaisesRegex(ValueError, "completion time"):
            Ours().load_state_dict(saved)

    def test_other_methods_schedule_hook_leaves_algorithm_state_unchanged(self) -> None:
        from afl_vlm.methods.registry import create_method, method_names

        for name in method_names():
            if name == "ours":
                continue
            with self.subTest(method=name):
                baseline = create_method(name)
                before = copy.deepcopy(baseline.state_dict())
                baseline.configure_schedule([])
                self.assertEqual(baseline.state_dict(), before)

    def test_reconfigured_schedule_preserves_completion_and_rejects_different_plan(self) -> None:
        method = self.configured()
        method.on_arrival(self.incoming("vqa"), ServerContext(state(0.0), 0, 3))
        restored = Ours()
        restored.load_state_dict(method.state_dict())
        restored.configure_server(state(0.0), [
            client("vqa/client_0", "vqa"), client("cls/client_0", "cls"),
        ])
        events = [
            ScheduledEvent(
                event_id=index, client_id=client_id, task=task, dataset=task,
                local_round=local_round, start_time=0.0, arrival_time=1.0,
                speed_factor=1.0, estimated_train_time=1.0, network_delay=0.0, local_steps=1,
            )
            for index, ((client_id, local_round), task) in enumerate(
                method.history_planned_jobs.items()
            )
        ]
        original_events = copy.deepcopy(events)
        restored.configure_schedule(events)
        self.assertEqual(restored.task_finished_at, method.task_finished_at)
        self.assertEqual(restored.task_pending_updates, method.task_pending_updates)
        self.assertEqual(events, original_events)
        with self.assertRaisesRegex(ValueError, "differs"):
            restored.configure_schedule(events[:-1])

    def test_age_only_changes_precision_and_increment_coefficient(self) -> None:
        enhanced = self.configured()
        original = self.configured(history_age_enabled=False)
        enhanced_server = FederatedServer(None, enhanced, 3, initial_state=state(0.0))
        original_server = FederatedServer(None, original, 3, initial_state=state(0.0))
        for task in ("vqa", "cls"):
            enhanced_server.receive(self.incoming(task))
            original_server.receive(self.incoming(task))
        self.assertEqual(enhanced_server.state, original_server.state)
        before = copy.deepcopy(enhanced_server.state)
        incoming = self.incoming("cls", 1)
        changed = enhanced_server.receive(copy.deepcopy(incoming))[0]
        unchanged = original_server.receive(copy.deepcopy(incoming))[0]
        self.assertLess(changed.applied_weight, unchanged.applied_weight)
        for field in ("reliability", "frequency_weight", "relative_staleness"):
            self.assertEqual(changed.metadata[field], unchanged.metadata[field])
        self.assertEqual(enhanced.task_memory, original.task_memory)
        self.assertEqual(enhanced.task_memory_counts, original.task_memory_counts)
        self.assertEqual(enhanced.task_frequency_window, original.task_frequency_window)
        alpha = changed.metadata["module_alphas"][MODULE]
        for name in (A, B):
            self.assertAlmostEqual(
                enhanced_server.state[name][0][0],
                before[name][0][0] + alpha * incoming.delta[name][0][0],
            )


if __name__ == "__main__":
    unittest.main()
