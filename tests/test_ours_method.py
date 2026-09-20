from __future__ import annotations

import math
import unittest

from afl_vlm.federation.types import ClientSpec, ServerContext, Update
from afl_vlm.methods.ours import Ours

MODULE = "base_model.model.layers.0.self_attn.q_proj"
A = f"{MODULE}.lora_A.default.weight"
B = f"{MODULE}.lora_B.default.weight"


def state(value: float) -> dict[str, list[list[float]]]:
    return {
        A: [[value, value], [value, value]],
        B: [[value, value], [value, value]],
    }


def client(client_id: str, task: str) -> ClientSpec:
    return ClientSpec(client_id, task, task, 4, 1.0, 0.0, 0.0, 1.0)


def update(local: float = 1.0, optimizer_steps: int = 1) -> Update:
    return Update(
        update_id="vqa/client_0-r0",
        client_id="vqa/client_0",
        task="vqa",
        dataset="aokvqa",
        num_samples=4,
        local_round=0,
        base_version=0,
        arrival_time=1.0,
        base_state=state(0.0),
        local_state=state(local),
        delta=state(local),
        seed=42,
        sample_ids_hash="hash",
        losses=[1.0],
        optimizer_steps=optimizer_steps,
        metadata={
            "group_sensitivity": {MODULE: 1.0},
            "sensitivity_estimator": "module_gate",
            "sensitivity_steps": optimizer_steps,
        },
    )


class OursMethodTests(unittest.TestCase):
    def configured(self, **params: object) -> Ours:
        method = Ours(params)
        method.configure_server(
            state(0.0),
            [client("vqa/client_0", "vqa"), client("cls/client_0", "cls")],
        )
        return method

    def test_fresh_update_uses_group_precision_and_updates_only_source_task(self) -> None:
        method = self.configured()
        mutation = method.on_arrival(update(), ServerContext(state(0.0), 0, 2))[0]

        self.assertAlmostEqual(mutation.applied_weight, 0.5)
        for row in mutation.new_state[A]:
            for value in row:
                self.assertAlmostEqual(value, 0.5)
        self.assertEqual(mutation.metadata["functional_staleness"], 0.0)
        self.assertAlmostEqual(method.task_memory["vqa"][MODULE], 0.1)
        self.assertEqual(method.task_memory["cls"], {})

    def test_functional_staleness_reduces_reliability_and_fusion_weight(self) -> None:
        fresh = self.configured()
        stale = self.configured()
        fresh_mutation = fresh.on_arrival(update(), ServerContext(state(0.0), 0, 2))[0]
        stale_mutation = stale.on_arrival(update(), ServerContext(state(2.0), 3, 2))[0]

        self.assertAlmostEqual(stale_mutation.metadata["functional_staleness"], 2.0)
        self.assertAlmostEqual(stale_mutation.metadata["reliability"], math.exp(-2.0))
        self.assertLess(stale_mutation.applied_weight, fresh_mutation.applied_weight)

    def test_negligible_update_is_rejected_without_advancing_version(self) -> None:
        method = self.configured()
        mutation = method.on_arrival(update(local=0.0), ServerContext(state(0.0), 0, 2))[0]

        self.assertEqual(mutation.applied_weight, 0.0)
        self.assertFalse(mutation.increment_version)
        self.assertEqual(mutation.metadata["rejected"], "negligible_update_energy")

    def test_state_round_trip_preserves_task_memory_and_weights(self) -> None:
        method = self.configured()
        method.on_arrival(update(), ServerContext(state(0.0), 0, 2))
        restored = Ours()
        restored.load_state_dict(method.state_dict())

        self.assertEqual(restored.task_memory, method.task_memory)
        self.assertEqual(restored.task_weights, method.task_weights)

    def test_rank_gate_builds_one_group_per_lora_rank(self) -> None:
        method = Ours({"sensitivity_estimator": "rank_gate"})
        method.configure_server(state(0.0), [client("vqa/client_0", "vqa")])

        self.assertEqual(set(method._groups), {f"{MODULE}::rank_0", f"{MODULE}::rank_1"})

    def test_non_lora_trainable_state_is_rejected(self) -> None:
        method = Ours()
        with self.assertRaisesRegex(ValueError, "LoRA-only"):
            method.configure_server({"mm_projector.weight": [[0.0]]}, [client("c", "vqa")])


if __name__ == "__main__":
    unittest.main()
