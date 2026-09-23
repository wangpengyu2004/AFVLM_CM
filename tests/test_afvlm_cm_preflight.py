from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from afl_vlm.data.afvlm_cm import (
    TASK_DATASETS,
    AFVLMDataModule,
    AFVLMTaskAdapter,
    validate_caption_metric_runtime,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _conversation(image: str) -> dict[str, object]:
    return {
        "id": "sample",
        "image": image,
        "conversations": [
            {"from": "human", "value": "<image>\nQuestion"},
            {"from": "gpt", "value": "Answer"},
        ],
    }


class AFVLMPreflightTests(unittest.TestCase):
    def test_adapter_accepts_all_observed_annotation_formats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = AFVLMTaskAdapter(
                "grounding",
                {"partition_dir": root, "image_root": root, "require_images": False},
            )
            conversation = adapter._sample(_conversation("image.jpg"), "train", 0)
            flat = adapter._sample(
                {
                    "question_id": "1",
                    "image": "image.jpg",
                    "text": "Question",
                    "answer": "A",
                },
                "validation",
                0,
            )
            grounding = adapter._sample(
                {
                    "question_id": "2",
                    "image": "image.jpg",
                    "text": "Locate it",
                    "answer_bbox": "[0.1,0.2,0.3,0.4]",
                },
                "test",
                0,
            )

            self.assertEqual(
                (conversation.instruction, conversation.answer), ("Question", "Answer")
            )
            self.assertEqual(conversation.turns, (("Question", "Answer"),))
            self.assertEqual((flat.instruction, flat.answer), ("Question", "A"))
            self.assertEqual(grounding.answer, "[0.1,0.2,0.3,0.4]")

    def test_preflight_checks_every_task_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partition_root = root / "partitioned"
            image_root = root / "images"
            image_root.mkdir()
            (image_root / "image.jpg").write_bytes(b"image")
            for task in TASK_DATASETS:
                task_root = partition_root / task
                _write(task_root / "client_0.json", [_conversation("image.jpg")])
                validation = {
                    "question_id": "1",
                    "image": "image.jpg",
                    "text": "Question",
                    "answer": "A",
                }
                if task == "caption":
                    validation["references"] = ["A", "B", "C", "D", "E"]
                _write(
                    task_root / "val.json",
                    [validation],
                )
                answer = (
                    {"answer_bbox": "[0.1,0.2,0.3,0.4]"} if task == "grounding" else {"answer": "A"}
                )
                final = {
                    "question_id": "2",
                    "image": "image.jpg",
                    "text": "Question",
                    **answer,
                }
                if task == "caption":
                    final["references"] = ["A", "B", "C", "D", "E"]
                _write(
                    task_root / "test.json",
                    [final],
                )

            data_module = AFVLMDataModule(
                {
                    "partition_root": partition_root,
                    "image_root": image_root,
                    "require_images": True,
                    "clients_per_task": 1,
                    "tasks": list(TASK_DATASETS),
                }
            )

            self.assertEqual(
                data_module.preflight_validate(),
                {
                    "files": 18,
                    "train": 6,
                    "validation": 6,
                    "test": 6,
                    "records": 18,
                    "unique_images": 1,
                },
            )

    def test_normal_sample_loading_does_not_precheck_image_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = AFVLMTaskAdapter(
                "vqa",
                {"partition_dir": root, "image_root": root, "require_images": True},
            )

            sample = adapter._sample(_conversation("missing.jpg"), "train", 0)

            self.assertEqual(Path(sample.image), root / "missing.jpg")

    def test_explicit_validation_still_reports_missing_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = AFVLMTaskAdapter(
                "vqa",
                {"partition_dir": root, "image_root": root, "require_images": True},
            )
            path = root / "client_0.json"
            _write(path, [_conversation("missing.jpg")])

            with self.assertRaisesRegex(FileNotFoundError, "Missing image"):
                adapter.validate_file(path, "train")

    def test_preflight_reports_late_malformed_conversation_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = AFVLMTaskAdapter(
                "grounding",
                {"partition_dir": root, "image_root": root, "require_images": False},
            )
            path = root / "val.json"
            record = _conversation("image.jpg")
            conversations = record["conversations"]
            assert isinstance(conversations, list)
            conversations.extend(
                [
                    {"from": "human", "value": "Second question"},
                    {"from": "human", "value": "wrong role"},
                ]
            )
            _write(path, [record])

            with self.assertRaisesRegex(ValueError, r"grounding/validation.*val\.json\[0\].*role"):
                adapter.validate_file(path, "validation")

    def test_grounding_keeps_all_training_turns_and_expands_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = AFVLMTaskAdapter(
                "grounding",
                {"partition_dir": root, "image_root": root, "require_images": False},
            )
            record = _conversation("image.jpg")
            conversations = record["conversations"]
            assert isinstance(conversations, list)
            conversations.extend(
                [
                    {"from": "human", "value": "Second question"},
                    {"from": "gpt", "value": "Second answer"},
                ]
            )
            path = root / "records.json"
            _write(path, [record])

            train = adapter.load_file(path, "train")
            validation = adapter.load_file(path, "validation")

            self.assertEqual(len(train), 1)
            self.assertEqual(
                train[0].turns,
                (("Question", "Answer"), ("Second question", "Second answer")),
            )
            self.assertEqual(len(validation), 2)
            self.assertEqual([sample.answer for sample in validation], ["Answer", "Second answer"])

    def test_fcit_task_metrics_and_grounding_mean_iou(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {
                "partition_dir": root,
                "image_root": root,
                "require_images": False,
                "evaluation": {"grounding_iou_threshold": 0.5},
            }
            cls = AFVLMTaskAdapter("cls", common)
            reasoning = AFVLMTaskAdapter("visual_reasoning", common)
            grounding = AFVLMTaskAdapter("grounding", common)

            self.assertEqual(cls.metric([" cat "], ["CAT"]), {"accuracy": 100.0})
            self.assertEqual(
                reasoning.metric(["A"], ["The answer is A"]),
                {"answer_accuracy": 100.0},
            )
            metrics = grounding.metric(
                ["[0,0,1,1]", "invalid"],
                ["[0,0,1,1]", "[0,0,1,1]"],
            )
            self.assertEqual(metrics["mean_IoU"], 0.5)
            self.assertEqual(metrics["IoU@0.5_accuracy"], 50.0)

    def test_caption_passes_five_references_to_shared_coco_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = AFVLMTaskAdapter(
                "caption",
                {"partition_dir": root, "image_root": root, "require_images": False},
            )
            references = [("a", "b", "c", "d", "e")]
            expected = {"CIDEr": 123.0}
            with patch(
                "afl_vlm.data.afvlm_cm.coco_caption_metrics", return_value=expected
            ) as scorer:
                self.assertEqual(adapter.metric(["caption"], references), expected)
            scorer.assert_called_once_with(["caption"], references)

    def test_caption_runtime_validation_fails_fast_without_java(self) -> None:
        components = (object(), object(), object(), object(), object())
        with (
            patch(
                "afl_vlm.data.afvlm_cm._load_coco_caption_components",
                return_value=components,
            ) as loader,
            patch("afl_vlm.data.afvlm_cm.shutil.which", return_value=None),
            self.assertRaisesRegex(RuntimeError, "Java runtime"),
        ):
            validate_caption_metric_runtime()
        loader.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
