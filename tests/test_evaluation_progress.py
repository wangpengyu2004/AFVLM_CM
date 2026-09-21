from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from afl_vlm.federation.worker_pool import _evaluate_states


class _TaskAdapter:
    def __init__(self, task_key: str, count: int) -> None:
        self.task_key = task_key
        self._samples = [SimpleNamespace(id=f"{task_key}-{index}") for index in range(count)]

    def load_split(self, split: str) -> list[Any]:
        return self._samples


class _Model:
    def load_trainable(self, state: Any) -> None:
        self.state = state

    def set_evaluation_context(self, task: str, client_id: str | None) -> None:
        self.context = (task, client_id)

    def evaluate(
        self,
        task_adapter: _TaskAdapter,
        sample_ids: list[str],
        mode: str,
        progress_hook: Any | None = None,
    ) -> dict[str, float]:
        for completed, _sample_id in enumerate(sample_ids, start=1):
            if progress_hook is not None:
                progress_hook(completed, len(sample_ids))
        return {"score": float(len(sample_ids))}


class EvaluationProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data_module = SimpleNamespace(
            tasks={"cls": _TaskAdapter("cls", 2), "vqa": _TaskAdapter("vqa", 3)}
        )

    def test_global_evaluation_reports_monotonic_cross_task_progress(self) -> None:
        progress: list[tuple[str, int, int]] = []
        metrics = _evaluate_states(
            _Model(),
            self.data_module,
            {"global": {}},
            "validation",
            lambda _message: None,
            lambda label, current, total: progress.append((label, current, total)),
        )

        self.assertEqual(metrics, {"cls": {"score": 2.0}, "vqa": {"score": 3.0}})
        self.assertEqual([current for _, current, _ in progress], [1, 2, 3, 4, 5])
        self.assertEqual({total for _, _, total in progress}, {5})
        self.assertEqual(progress[-1], ("vqa", 5, 5))

    def test_local_evaluation_total_includes_each_client_model(self) -> None:
        progress: list[tuple[str, int, int]] = []
        _evaluate_states(
            _Model(),
            self.data_module,
            {"cls/client_0": {}, "cls/client_1": {}},
            "final",
            lambda _message: None,
            lambda label, current, total: progress.append((label, current, total)),
        )

        self.assertEqual([current for _, current, _ in progress], [1, 2, 3, 4])
        self.assertEqual({total for _, _, total in progress}, {4})
        self.assertEqual(progress[-1], ("cls/client_1", 4, 4))


if __name__ == "__main__":
    unittest.main()
