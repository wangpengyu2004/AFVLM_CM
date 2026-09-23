from __future__ import annotations

import unittest

from afl_vlm.execution import resolve_runtime_backend
from afl_vlm.methods.registry import create_method
from afl_vlm.scheduling.train_plan import optimizer_steps_for_distributed_epochs


class ExecutorSelectionTests(unittest.TestCase):
    def config(self, policy: str = "capability") -> dict[str, object]:
        return {
            "runtime": {
                "backend": "client_parallel",
                "executor_policy": policy,
            }
        }

    def test_standard_and_module_gate_methods_select_ddp(self) -> None:
        for name in (
            "local",
            "fedavg",
            "fedprox",
            "fedadam",
            "fedasync",
            "fedbuff",
            "unifed_lora",
            "ours",
        ):
            with self.subTest(method=name):
                self.assertEqual(
                    resolve_runtime_backend(self.config(), create_method(name, {})),
                    "client_ddp",
                )

    def test_step_sensitive_methods_keep_client_parallel_workers(self) -> None:
        configurations = {
            "fedcompass": {
                "min_local_steps": 1,
                "max_local_steps": 2,
                "group_window": 0.25,
            },
            "fedasmu": {},
            "masfl": {},
            "adamasfl": {},
            "pilot": {},
        }
        for name, params in configurations.items():
            with self.subTest(method=name):
                self.assertEqual(
                    resolve_runtime_backend(self.config(), create_method(name, params)),
                    "client_parallel",
                )

    def test_fixed_policy_preserves_configured_backend(self) -> None:
        self.assertEqual(
            resolve_runtime_backend(self.config("fixed"), create_method("fedavg", {})),
            "client_parallel",
        )

    def test_distributed_optimizer_steps_match_padded_rank_shards(self) -> None:
        # ceil(100 / 8) = 13 samples/rank, 7 micro-batches/rank, ceil(7 / 4) = 2.
        self.assertEqual(
            optimizer_steps_for_distributed_epochs(
                num_samples=100,
                local_epochs=1,
                per_device_batch_size=2,
                gradient_accumulation=4,
                world_size=8,
            ),
            2,
        )


if __name__ == "__main__":
    unittest.main()
