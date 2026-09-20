from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from afl_vlm.data.afvlm_cm import TASK_DATASETS, AFVLMDataModule, AFVLMTaskAdapter


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
                _write(
                    task_root / "val.json",
                    [
                        {
                            "question_id": "1",
                            "image": "image.jpg",
                            "text": "Question",
                            "answer": "A",
                        }
                    ],
                )
                answer = (
                    {"answer_bbox": "[0.1,0.2,0.3,0.4]"} if task == "grounding" else {"answer": "A"}
                )
                _write(
                    task_root / "test.json",
                    [{"question_id": "2", "image": "image.jpg", "text": "Question", **answer}],
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


if __name__ == "__main__":
    unittest.main()
