from __future__ import annotations

import math
import unittest

from afl_vlm.federation.types import ClientSpec, ServerContext, Update
from afl_vlm.methods.ours import (
    Ours,
    compute_functional_staleness,
    rank_functional_distance_sq,
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
) -> Update:
    base = state(base_a, base_b)
    local = state(local_a, local_b)
    return Update(
        update_id="vqa/client_0-r0",
        client_id="vqa/client_0",
        task="vqa",
        dataset="aokvqa",
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
            "rank_sensitivity": {MODULE: list(sensitivity)},
            "module_sensitivity": {MODULE: sum(sensitivity) / len(sensitivity)},
            "sensitivity_valid": True,
            "sensitivity_observations": {MODULE: 2},
            "lora_metadata": metadata(),
        },
    )


class RankFunctionalDistanceTests(unittest.TestCase):
    def test_rank_distance_uses_outer_product_identity(self) -> None:
        distance = rank_functional_distance_sq(
            [2.0, 2.0],
            [2.0, 2.0],
            [0.0, 0.0],
            [0.0, 0.0],
            1.0,
        )
        self.assertAlmostEqual(distance, 64.0)

    def test_functional_staleness_is_rank_weighted_and_size_normalized(self) -> None:
        drift, local, relative = compute_functional_staleness(
            state(0.0),
            state(2.0),
            state(1.0),
            {MODULE: [1.0, 1.0]},
            metadata(),
        )
        self.assertAlmostEqual(drift, 32.0)
        self.assertAlmostEqual(local, 2.0)
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
        self.assertAlmostEqual(method.task_memory["vqa"][MODULE], 0.05)
        self.assertEqual(method.task_memory["cls"], {MODULE: 0.0})

    def test_functional_staleness_reduces_reliability_and_module_alpha(self) -> None:
        fresh = self.configured()
        stale = self.configured()
        fresh_mutation = fresh.on_arrival(update(), ServerContext(state(0.0), 0, 2))[0]
        stale_mutation = stale.on_arrival(update(), ServerContext(state(2.0), 3, 2))[0]

        self.assertAlmostEqual(stale_mutation.metadata["relative_staleness"], 4.0)
        self.assertAlmostEqual(stale_mutation.metadata["reliability"], math.exp(-4.0))
        self.assertLess(stale_mutation.applied_weight, fresh_mutation.applied_weight)

    def test_no_effective_update_is_rejected_without_advancing_version(self) -> None:
        method = self.configured()
        mutation = method.on_arrival(update(local_a=0.0), ServerContext(state(0.0), 0, 2))[0]

        self.assertEqual(mutation.applied_weight, 0.0)
        self.assertFalse(mutation.increment_version)
        self.assertEqual(mutation.metadata["rejected"], "no_effective_update")

    def test_functionally_equal_but_parameter_changed_update_is_not_noop(self) -> None:
        method = self.configured()
        # BA is unchanged: (0.5 B) @ (2 A) == B @ A, but A/B moved.
        mutation = method.on_arrival(
            update(local_a=2.0, local_b=0.5, base_a=1.0, base_b=1.0),
            ServerContext(state(1.0), 0, 2),
        )[0]

        self.assertTrue(mutation.increment_version)
        self.assertNotIn("rejected", mutation.metadata)
        self.assertAlmostEqual(mutation.metadata["local_update"], 0.0)

    def test_state_round_trip_preserves_memory_weights_and_metadata(self) -> None:
        method = self.configured()
        method.on_arrival(update(), ServerContext(state(0.0), 0, 2))
        restored = Ours()
        restored.load_state_dict(method.state_dict())

        self.assertEqual(restored.task_memory, method.task_memory)
        self.assertEqual(restored.task_weights, method.task_weights)
        self.assertEqual(restored._server_metadata, method._server_metadata)

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
