from __future__ import annotations

import multiprocessing as mp
import queue
import unittest

from afl_vlm.federation.worker_pool import ClientWorkerPool, EvaluationJob


class WorkerPoolCleanupTests(unittest.TestCase):
    @staticmethod
    def _queue_only_pool() -> ClientWorkerPool:
        context = mp.get_context("spawn")
        pool = ClientWorkerPool.__new__(ClientWorkerPool)
        pool._inputs = [context.Queue(), context.Queue()]
        pool._output = context.Queue()
        pool._queues_closed = False
        return pool

    def test_graceful_queue_cleanup_is_idempotent(self) -> None:
        pool = self._queue_only_pool()
        pool._close_queues(cancel_pending=False)
        pool._close_queues(cancel_pending=False)
        self.assertTrue(pool._queues_closed)

    def test_abort_queue_cleanup_is_idempotent(self) -> None:
        pool = self._queue_only_pool()
        pool._close_queues(cancel_pending=True)
        pool._close_queues(cancel_pending=True)
        self.assertTrue(pool._queues_closed)

    def test_suspended_dispatch_preserves_all_idle_workers_for_evaluation_barrier(self) -> None:
        pool = ClientWorkerPool.__new__(ClientWorkerPool)
        pool._processes = [object(), object()]
        pool._inputs = [queue.Queue(), queue.Queue()]
        pool._idle = {0, 1}
        pool._busy = {}
        pool._pending = []
        pool._submission_sequence = 0
        pool._dispatch_suspended = False

        pool.suspend_dispatch()
        for shard_index in range(2):
            pool.submit(
                EvaluationJob(
                    job_id=f"eval:{shard_index}",
                    group_id="eval",
                    states={},
                    split="validation",
                    shard_index=shard_index,
                    shard_count=2,
                ),
                priority=True,
            )
        self.assertEqual(pool.busy_count, 0)
        self.assertEqual(len(pool._pending), 2)

        pool.resume_dispatch()
        self.assertEqual(pool.busy_count, 2)
        self.assertFalse(pool._pending)

    def test_priority_dispatch_does_not_dispatch_training_behind_task_jobs(self) -> None:
        pool = ClientWorkerPool.__new__(ClientWorkerPool)
        pool._processes = [object(), object(), object()]
        pool._inputs = [queue.Queue(), queue.Queue(), queue.Queue()]
        pool._idle = {0, 1, 2}
        pool._busy = {}
        pool._pending = []
        pool._submission_sequence = 0
        pool._dispatch_suspended = True

        for shard_index in range(2):
            pool.submit(
                EvaluationJob(
                    f"eval:{shard_index}", "eval", {}, "validation", shard_index, 2
                ),
                priority=True,
            )
        # A lower-priority placeholder represents queued client training.
        pool.submit(EvaluationJob("later", "later", {}, "validation", 0, 1))

        pool.dispatch_priority_jobs()
        self.assertEqual(pool.busy_count, 2)
        self.assertEqual(len(pool._pending), 1)
        self.assertTrue(pool._dispatch_suspended)


if __name__ == "__main__":
    unittest.main()
