from __future__ import annotations

import unittest
from typing import Any

from afl_vlm.runner import _initialize_server


class _Model:
    def __init__(self) -> None:
        self.value = 1.0
        self.calls: list[str] = []

    def snapshot_trainable(self) -> dict[str, float]:
        self.calls.append("snapshot")
        return {"lora": self.value}

    def enable_distributed_data_parallel(self, local_rank: int) -> None:
        self.calls.append(f"ddp:{local_rank}")
        # Simulate DDP's rank-0 parameter synchronization.
        self.value = 42.0


class _Method:
    name = "fake"

    def __init__(self) -> None:
        self.server_state: dict[str, Any] | None = None

    def configure_model(self, model: _Model, profiles: list[Any]) -> None:
        del profiles
        model.calls.append("configure_model")

    def configure_server(self, initial_state: dict[str, Any], profiles: list[Any]) -> None:
        del profiles
        self.server_state = initial_state


class DDPInitializationTests(unittest.TestCase):
    def test_server_snapshot_is_created_after_ddp_parameter_sync(self) -> None:
        model = _Model()
        method = _Method()

        server = _initialize_server(
            model,
            method,
            profiles=[],
            client_count=2,
            distributed=True,
            local_rank=3,
        )

        self.assertEqual(model.calls, ["configure_model", "ddp:3", "snapshot"])
        self.assertEqual(server.state, {"lora": 42.0})
        self.assertEqual(method.server_state, {"lora": 42.0})


if __name__ == "__main__":
    unittest.main()
