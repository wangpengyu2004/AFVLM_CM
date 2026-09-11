"""FedMLLM JSON/JSONL normalization for Hateful-Memes and medical VQA tasks."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from afl_vlm.data.base import Sample, TaskAdapter
from afl_vlm.data.registry import register_task_backend


def _normalized_answer(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()


def _binary_label(text: str) -> int:
    normalized = _normalized_answer(text)
    return int(normalized in {"1", "yes", "true", "hateful", "hate"})


def _binary_auc(scores: list[float], labels: list[int]) -> float:
    positives = [score for score, label in zip(scores, labels, strict=True) if label == 1]
    negatives = [score for score, label in zip(scores, labels, strict=True) if label == 0]
    if not positives or not negatives:
        return 0.5
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


@register_task_backend("fedmllm")
class FedMLLMTaskAdapter(TaskAdapter):
    """Accept common FedMLLM, MiniCPM-V chat, and normalized record layouts."""

    def __init__(self, task_key: str, config: Mapping[str, Any]) -> None:
        self.task_key = task_key
        self.config = dict(config)
        self.task_name = str(config["name"])
        self._cache: dict[str, list[Sample]] = {}
        self._by_id: dict[str, Sample] = {}

    def _synthetic(self, split: str) -> list[Sample]:
        if split == "train":
            count = max(384, int(self.config.get("train_per_client", 128)) * 3)
        elif split == "probe":
            count = int(self.config.get("probe_samples", 16))
        else:
            count = int(self.config.get("final_eval_samples", 64))
        samples: list[Sample] = []
        for index in range(count):
            if self.task_key == "fast":
                instruction = "Is this meme hateful? Answer yes or no."
                answer = "yes" if index % 2 else "no"
            else:
                instruction = f"Answer the medical visual question for synthetic case {index}."
                answer = "normal" if index % 2 else "abnormal"
            samples.append(
                Sample(
                    id=f"{self.task_key}-{split}-{index:04d}",
                    task_name=self.task_name,
                    image=None,
                    instruction=instruction,
                    answer=answer,
                    split=split,
                    metadata={"synthetic": True, "label": _binary_label(answer)},
                )
            )
        return samples

    def _records(self, path: Path) -> list[dict[str, Any]]:
        if path.suffix.lower() == ".jsonl":
            return [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        for key in ("data", "samples", "annotations"):
            if isinstance(payload.get(key), list):
                return payload[key]
        raise ValueError(f"Unsupported JSON container in {path}")

    def _normalize(self, record: Mapping[str, Any], split: str, index: int, root: Path) -> Sample:
        conversations = record.get("conversations") or record.get("messages") or []
        question = record.get("instruction") or record.get("question") or record.get("text")
        answer = record.get("answer") or record.get("label")
        if conversations:
            question = question or conversations[0].get("value") or conversations[0].get("content")
            if len(conversations) > 1:
                answer = (
                    answer
                    if answer is not None
                    else (conversations[1].get("value") or conversations[1].get("content"))
                )
        if question is None or answer is None:
            raise ValueError(
                f"Record {index} in split '{split}' lacks question/instruction or answer"
            )
        image = record.get("image") or record.get("image_path") or record.get("img")
        if image:
            image_path = Path(str(image))
            if not image_path.is_absolute():
                image_path = (root / image_path).resolve()
            image = str(image_path)
        sample_id = str(
            record.get("id") or record.get("question_id") or f"{self.task_key}-{split}-{index}"
        )
        metadata = dict(record.get("metadata") or {})
        for key in ("label", "category", "answer_type"):
            if key in record:
                metadata[key] = record[key]
        return Sample(
            id=sample_id,
            task_name=self.task_name,
            image=str(image) if image else None,
            instruction=str(question).replace("<image>", "").strip(),
            answer=str(answer),
            split=split,
            metadata=metadata,
        )

    def load_split(self, split: str) -> list[Sample]:
        if split in self._cache:
            return list(self._cache[split])
        data_files = dict(self.config.get("data_files") or {})
        path_value = data_files.get(split)
        if path_value:
            path = Path(path_value).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"Configured {self.task_key}.{split} file not found: {path}"
                )
            samples = [
                self._normalize(record, split, index, path.parent)
                for index, record in enumerate(self._records(path))
            ]
        elif self.config.get("allow_synthetic", False):
            samples = self._synthetic(split)
        else:
            raise ValueError(
                f"tasks.{self.task_key}.data_files.{split} is required "
                "when synthetic data is disabled"
            )
        if len({sample.id for sample in samples}) != len(samples):
            raise ValueError(f"Duplicate sample IDs in {self.task_key}.{split}")
        self._cache[split] = samples
        self._by_id.update({sample.id: sample for sample in samples})
        return list(samples)

    def format_sample(self, sample: Sample) -> tuple[list[dict[str, Any]], str | None]:
        content: list[dict[str, Any]] = []
        if sample.image:
            content.append({"type": "image", "image": sample.image})
        content.append({"type": "text", "text": sample.instruction})
        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": sample.answer}]},
        ], sample.image

    def collate(self, samples: list[Sample], model_adapter: Any) -> list[Sample]:
        return list(samples)

    def metric(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        if len(predictions) != len(references) or not references:
            raise ValueError("Predictions and references must have equal non-zero length")
        if self.task_name == "hateful_memes" or self.task_key == "fast":
            predicted_labels = [_binary_label(value) for value in predictions]
            labels = [_binary_label(value) for value in references]
            accuracy = sum(a == b for a, b in zip(predicted_labels, labels, strict=True)) / len(
                labels
            )
            return {
                "accuracy": accuracy,
                "auc": _binary_auc([float(value) for value in predicted_labels], labels),
            }
        exact = sum(
            _normalized_answer(prediction) == _normalized_answer(reference)
            for prediction, reference in zip(predictions, references, strict=True)
        ) / len(references)
        return {"exact": exact, "official_accuracy": exact}

    def probe_loss(self, model: Any, sample_ids: list[str]) -> float:
        return float(model.evaluate(self, sample_ids, "probe")["loss"])

    def samples_by_id(self, sample_ids: list[str]) -> list[Sample]:
        missing = [sample_id for sample_id in sample_ids if sample_id not in self._by_id]
        if missing:
            # Populate all standard splits before failing to support direct lookup.
            for split in ("train", "probe", "final"):
                if split not in self._cache:
                    self.load_split(split)
            missing = [sample_id for sample_id in sample_ids if sample_id not in self._by_id]
        if missing:
            raise KeyError(f"Unknown sample IDs: {missing[:3]}")
        return [self._by_id[sample_id] for sample_id in sample_ids]
