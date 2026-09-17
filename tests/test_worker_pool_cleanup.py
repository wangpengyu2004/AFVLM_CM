from __future__ import annotations

import multiprocessing as mp
import unittest

from afl_vlm.federation.worker_pool import ClientWorkerPool


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


if __name__ == "__main__":
    unittest.main()
