from __future__ import annotations

import unittest
from types import SimpleNamespace

from afl_vlm.evaluation_schedule import (
    PeriodicEvaluationSchedule,
    evaluation_progress_count,
)
from afl_vlm.federation.server import FederatedServer
from afl_vlm.federation.types import Update
from afl_vlm.methods.standard import FedBuff


def _update(index: int) -> Update:
    return Update(
        update_id=f"vqa/client_0-r{index}",
        client_id="vqa/client_0",
        task="vqa",
        dataset="aokvqa",
        num_samples=1,
        local_round=index,
        base_version=0,
        arrival_time=float(index),
        base_state={"p": [0.0]},
        local_state={"p": [1.0]},
        delta={"p": [1.0]},
        seed=index,
        sample_ids_hash=str(index),
        losses=[1.0],
        optimizer_steps=1,
    )


class PeriodicEvaluationScheduleTests(unittest.TestCase):
    def test_single_update_methods_keep_every_ten_behavior(self) -> None:
        schedule = PeriodicEvaluationSchedule.from_config(
            {
                "interval_unit": "incorporated_client_updates",
                "eval_every_client_updates": 10,
            }
        )

        self.assertEqual(schedule.observe(9), ())
        self.assertEqual(schedule.observe(10), (10,))
        self.assertEqual(schedule.observe(10), ())
        self.assertEqual(schedule.next_threshold, 20)

    def test_grouped_aggregation_crosses_milestones_without_duplicate_evaluation(self) -> None:
        schedule = PeriodicEvaluationSchedule(
            unit="incorporated_client_updates", interval=10, next_threshold=10
        )

        self.assertEqual(schedule.observe(8), ())
        self.assertEqual(schedule.observe(13), (10,))
        self.assertEqual(schedule.observe(35), (20, 30))
        self.assertEqual(schedule.next_threshold, 40)

    def test_global_methods_count_only_incorporated_client_updates(self) -> None:
        server = SimpleNamespace(version=2, accepted_updates=10, received_updates=13)

        self.assertEqual(
            evaluation_progress_count(
                server, "server_global", "incorporated_client_updates"
            ),
            10,
        )

    def test_fedbuff_advances_by_buffer_contributors_not_server_versions(self) -> None:
        method = FedBuff(
            {
                "buffer_size": 5,
                "staleness_weighting": False,
                "flush_last_buffer": True,
            }
        )
        server = FederatedServer(None, method, 1, initial_state={"p": [0.0]})
        schedule = PeriodicEvaluationSchedule(
            unit="incorporated_client_updates", interval=10, next_threshold=10
        )

        for index in range(1, 6):
            server.receive(_update(index))
        self.assertEqual(server.version, 1)
        self.assertEqual(server.accepted_updates, 5)
        self.assertEqual(schedule.observe(server.accepted_updates), ())

        for index in range(6, 11):
            server.receive(_update(index))
        self.assertEqual(server.version, 2)
        self.assertEqual(server.accepted_updates, 10)
        self.assertEqual(schedule.observe(server.accepted_updates), (10,))

    def test_local_only_counts_completed_local_updates(self) -> None:
        server = SimpleNamespace(version=0, accepted_updates=0, received_updates=10)

        self.assertEqual(
            evaluation_progress_count(
                server, "client_local_mean", "incorporated_client_updates"
            ),
            10,
        )

    def test_legacy_profile_is_migrated_to_incorporated_update_budget(self) -> None:
        schedule = PeriodicEvaluationSchedule.from_config(
            {
                "interval_unit": "server_updates",
                "eval_every_server_updates": 10,
            }
        )
        server = SimpleNamespace(version=10, accepted_updates=50, received_updates=50)

        count = evaluation_progress_count(server, "server_global", schedule.unit)
        self.assertEqual(schedule.unit, "incorporated_client_updates")
        self.assertEqual(count, 50)
        self.assertEqual(schedule.observe(count), (10, 20, 30, 40, 50))


if __name__ == "__main__":
    unittest.main()
