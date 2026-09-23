from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

from afl_vlm.evaluation import merge_evaluation_outputs
from afl_vlm.federation.worker_pool import _evaluate_states, _evaluate_states_shard
from afl_vlm.parallel_runner import _merge_parallel_evaluation
from afl_vlm.runner import _primary_corpus_metrics


class _TaskAdapter:
    def __init__(self, task_key: str, count: int) -> None:
        self.task_key = task_key
        self._samples = [SimpleNamespace(id=f"{task_key}-{index}") for index in range(count)]

    def load_split(self, split: str) -> list[Any]:
        return self._samples

    def metric(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        correct = sum(
            prediction == reference
            for prediction, reference in zip(predictions, references, strict=True)
        )
        return {"score": correct / len(references)}


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

    def generate_evaluation_outputs(
        self,
        task_adapter: _TaskAdapter,
        sample_ids: list[str],
        progress_hook: Any | None = None,
        progress_label: str | None = None,
    ) -> tuple[list[str], list[str]]:
        del task_adapter, progress_label
        for completed, _sample_id in enumerate(sample_ids, start=1):
            if progress_hook is not None:
                progress_hook(completed, len(sample_ids))
        outputs = [f"answer:{sample_id}" for sample_id in sample_ids]
        return outputs, list(outputs)


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


class DistributedEvaluationTests(unittest.TestCase):
    def test_only_primary_rank_computes_complete_corpus_metric(self) -> None:
        adapter = Mock()
        adapter.metric.return_value = {"score": 1.0}
        gathered = [
            {"rows": [(0, "p0", "r0")]},
            {"rows": [(1, "p1", "r1")]},
        ]

        self.assertIsNone(_primary_corpus_metrics(adapter, gathered, 2, rank=1))
        self.assertEqual(
            _primary_corpus_metrics(adapter, gathered, 2, rank=0), {"score": 1.0}
        )
        adapter.metric.assert_called_once_with(["p0", "p1"], ["r0", "r1"])

    def test_shards_are_merged_in_original_corpus_order(self) -> None:
        predictions, references = merge_evaluation_outputs(
            [
                [(0, "p0", "r0"), (2, "p2", "r2")],
                [(1, "p1", "r1"), (3, "p3", "r3")],
            ],
            total=4,
        )

        self.assertEqual(predictions, ["p0", "p1", "p2", "p3"])
        self.assertEqual(references, ["r0", "r1", "r2", "r3"])

    def test_missing_or_duplicate_samples_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            merge_evaluation_outputs(
                [[(0, "p0", "r0")], [(0, "duplicate", "duplicate")]],
                total=2,
            )

    def test_client_parallel_shards_use_the_complete_corpus_metric(self) -> None:
        data_module = SimpleNamespace(
            tasks={"cls": _TaskAdapter("cls", 2), "vqa": _TaskAdapter("vqa", 3)}
        )
        payloads = [
            _evaluate_states_shard(
                _Model(),
                data_module,
                {"global": {}},
                "validation",
                shard_index=index,
                shard_count=2,
                status=lambda _message: None,
                progress=lambda _label, _current, _total: None,
            )
            for index in range(2)
        ]

        metrics = _merge_parallel_evaluation(
            data_module,
            {"global": {}},
            "validation",
            payloads,
        )

        self.assertEqual(metrics, {"cls": {"score": 1.0}, "vqa": {"score": 1.0}})

    def test_client_local_states_are_reconstructed_before_task_macro_average(self) -> None:
        data_module = SimpleNamespace(tasks={"cls": _TaskAdapter("cls", 3)})
        states = {"cls/client_0": {}, "cls/client_1": {}}
        payloads = [
            _evaluate_states_shard(
                _Model(),
                data_module,
                states,
                "validation",
                shard_index=index,
                shard_count=2,
                status=lambda _message: None,
                progress=lambda _label, _current, _total: None,
            )
            for index in range(2)
        ]

        metrics = _merge_parallel_evaluation(
            data_module,
            states,
            "validation",
            payloads,
        )

        self.assertEqual(metrics, {"cls": {"score": 1.0}})


if __name__ == "__main__":
    unittest.main()
