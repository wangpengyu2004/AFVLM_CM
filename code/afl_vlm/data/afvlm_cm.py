"""Read-only adapter for the existing six-task AFVLM-CM partitions."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from afl_vlm.data.base import Sample, TaskAdapter
from afl_vlm.data.registry import register_task_backend

TASK_DATASETS = {
    "cls": "ImageNet-R",
    "caption": "Flickr30k",
    "vqa": "AOKVQA",
    "chart_vqa": "DVQA",
    "visual_reasoning": "FigureQA",
    "grounding": "Grounding",
}


def resolve_image_path(
    sample: Mapping[str, Any] | str,
    dataset_config: Mapping[str, Any],
) -> Path:
    """Resolve one annotation path without dataset-specific conditionals."""
    value = sample.get("image", "") if isinstance(sample, Mapping) else sample
    relative = str(value).replace("\\", "/")
    while relative.startswith("./"):
        relative = relative[2:]
    image_root = Path(str(dataset_config["image_root"])).resolve()
    image = (image_root / relative).resolve()
    if not image.is_relative_to(image_root):
        raise ValueError(f"Image path escapes image_root: {relative}")
    if bool(dataset_config.get("require_images", True)) and not image.is_file():
        raise FileNotFoundError(
            f"Missing image: {image}. Run tools/download_afvlm_cm_images.py "
            "for the required source."
        )
    return image


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"[^a-z0-9.-]+", " ", text).strip()


def _lcs(left: list[str], right: list[str]) -> int:
    row = [0] * (len(right) + 1)
    for token in left:
        previous = 0
        for index, other in enumerate(right, 1):
            old = row[index]
            row[index] = previous + 1 if token == other else max(row[index], row[index - 1])
            previous = old
    return row[-1]


def rouge_l(prediction: str, reference: str) -> float:
    pred, ref = prediction.lower().split(), reference.lower().split()
    if not pred or not ref:
        return 0.0
    length = _lcs(pred, ref)
    precision, recall = length / len(pred), length / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def cider(predictions: list[str], references: list[str]) -> float:
    """Corpus TF-IDF n-gram cosine CIDEr for one-reference Flickr30k records.

    This follows CIDEr's 1--4 gram TF-IDF cosine construction.  Scores are on
    the conventional 0--10 scale; use the official COCO scorer for direct
    leaderboard comparison when multiple references are available.
    """
    documents = [[token for token in _normalize(ref).split() if token] for ref in references]
    result = 0.0
    for n in range(1, 5):
        document_frequency: Counter[tuple[str, ...]] = Counter()
        for tokens in documents:
            document_frequency.update(set(_ngrams(tokens, n)))
        per_example = []
        for prediction, ref_tokens in zip(predictions, documents, strict=True):
            pred_counts = _ngrams(_normalize(prediction).split(), n)
            ref_counts = _ngrams(ref_tokens, n)
            keys = set(pred_counts) | set(ref_counts)
            pred_vector = {}
            ref_vector = {}
            for key in keys:
                idf = math.log(max(1.0, len(documents) / max(1, document_frequency[key])))
                pred_vector[key] = pred_counts[key] * idf
                ref_vector[key] = ref_counts[key] * idf
            dot = sum(pred_vector[key] * ref_vector[key] for key in keys)
            pnorm = math.sqrt(sum(value * value for value in pred_vector.values()))
            rnorm = math.sqrt(sum(value * value for value in ref_vector.values()))
            per_example.append(dot / (pnorm * rnorm) if pnorm and rnorm else 0.0)
        result += sum(per_example) / len(per_example)
    return 10.0 * result / 4.0


def _bbox(text: str) -> tuple[float, float, float, float] | None:
    values = re.findall(r"[-+]?\d*\.?\d+", text)
    if len(values) < 4:
        return None
    box = tuple(float(value) for value in values[-4:])
    return box if len(box) == 4 else None


def _iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(0.0, min(ly2, ry2) - max(ly1, ry1))
    union = (
        max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
        + max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
        - intersection
    )
    return intersection / union if union else 0.0


@dataclass(frozen=True, slots=True)
class ClientPartition:
    client_id: str
    task: str
    dataset: str
    train_file: Path
    num_samples: int


@register_task_backend("afvlm_cm")
class AFVLMTaskAdapter(TaskAdapter):
    def __init__(self, task_key: str, config: Mapping[str, Any]) -> None:
        if task_key not in TASK_DATASETS:
            raise ValueError(f"Unknown AFVLM-CM task: {task_key}")
        self.task_key = task_key
        self.task_name = task_key
        self.dataset_name = TASK_DATASETS[task_key]
        self.partition_dir = Path(str(config["partition_dir"])).resolve()
        self.image_root = Path(str(config["image_root"])).resolve()
        self.require_images = bool(config.get("require_images", True))
        self._by_id: dict[str, Sample] = {}
        self._split_cache: dict[str, list[Sample]] = {}

    @staticmethod
    def _records(path: Path) -> list[dict[str, Any]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list: {path}")
        return payload

    def _sample(self, record: Mapping[str, Any], split: str, index: int) -> Sample:
        conversations = record.get("conversations") or []
        if len(conversations) < 2:
            raise ValueError(f"AFVLM-CM record has no user/assistant pair: {record}")
        image = resolve_image_path(
            record,
            {"image_root": self.image_root, "require_images": self.require_images},
        )
        sample_id = str(record.get("id") or record.get("question_id") or index)
        sample = Sample(
            id=f"{self.task_key}:{split}:{sample_id}:{index}",
            task_name=self.task_key,
            image=str(image),
            instruction=str(conversations[0].get("value", "")).replace("<image>", "").strip(),
            answer=str(conversations[1].get("value", "")).strip(),
            split=split,
            metadata={
                "dataset": self.dataset_name,
                "source_id": sample_id,
                "raw_record": dict(record),
            },
        )
        self._by_id[sample.id] = sample
        return sample

    def load_file(self, path: str | Path, split: str = "train") -> list[Sample]:
        source = Path(path).resolve()
        return [self._sample(record, split, i) for i, record in enumerate(self._records(source))]

    def load_split(self, split: str) -> list[Sample]:
        if split in self._split_cache:
            return list(self._split_cache[split])
        filename = {"validation": "val", "final": "test"}.get(split, split)
        samples = self.load_file(self.partition_dir / f"{filename}.json", split)
        self._split_cache[split] = samples
        return list(samples)

    def format_sample(self, sample: Sample) -> tuple[list[dict[str, Any]], str | None]:
        return (
            [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": sample.instruction}],
                },
                {"role": "assistant", "content": [{"type": "text", "text": sample.answer}]},
            ],
            sample.image,
        )

    def collate(self, samples: list[Sample], model_adapter: Any) -> list[Sample]:
        return samples

    def metric(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        if not predictions or len(predictions) != len(references):
            raise ValueError("Predictions and references must have equal non-zero length")
        if self.task_key == "caption":
            return {
                "CIDEr": cider(predictions, references),
                "ROUGE-L": sum(rouge_l(p, r) for p, r in zip(predictions, references, strict=True))
                / len(references),
            }
        if self.task_key == "grounding":
            values = [
                _iou(pred, ref) if pred is not None and ref is not None else 0.0
                for pred, ref in zip(map(_bbox, predictions), map(_bbox, references), strict=True)
            ]
            return {
                "mean_IoU": sum(values) / len(values),
                "IoU@0.5_accuracy": sum(v >= 0.5 for v in values) / len(values),
            }
        accuracy = sum(
            _normalize(p) == _normalize(r) for p, r in zip(predictions, references, strict=True)
        ) / len(references)
        name = (
            "VQA_accuracy"
            if self.task_key == "vqa"
            else "answer_accuracy"
            if self.task_key in {"chart_vqa", "visual_reasoning"}
            else "accuracy"
        )
        return {name: accuracy}

    def probe_loss(self, model: Any, sample_ids: list[str]) -> float:
        return float(model.evaluate(self, sample_ids, "probe")["loss"])

    def samples_by_id(self, sample_ids: list[str]) -> list[Sample]:
        missing = [item for item in sample_ids if item not in self._by_id]
        if missing:
            raise KeyError(f"Unknown sample IDs for {self.task_key}: {missing[:3]}")
        return [self._by_id[item] for item in sample_ids]


def scan_partitions(
    dataset_config: Mapping[str, Any],
) -> tuple[dict[str, AFVLMTaskAdapter], list[ClientPartition]]:
    """Discover actual client files; never assumes or rewrites a partition."""
    root = Path(str(dataset_config["partition_root"])).resolve()
    image_root = Path(str(dataset_config["image_root"])).resolve()
    tasks: dict[str, AFVLMTaskAdapter] = {}
    clients: list[ClientPartition] = []
    for task in dataset_config["tasks"]:
        task_dir = root / task
        adapter = AFVLMTaskAdapter(
            task,
            {
                "partition_dir": str(task_dir),
                "image_root": str(image_root),
                "require_images": dataset_config.get("require_images", True),
            },
        )
        tasks[task] = adapter
        for path in sorted(
            task_dir.glob("client_*.json"), key=lambda item: int(item.stem.split("_")[-1])
        ):
            payload = adapter._records(path)
            clients.append(
                ClientPartition(
                    client_id=f"{task}/{path.stem}",
                    task=task,
                    dataset=TASK_DATASETS[task],
                    train_file=path,
                    num_samples=len(payload),
                )
            )
    expected = int(dataset_config["clients_per_task"])
    counts = Counter(client.task for client in clients)
    if any(counts[task] != expected for task in dataset_config["tasks"]):
        raise ValueError(
            f"AFVLM-CM client count mismatch: detected {dict(counts)}, expected {expected} per task"
        )
    return tasks, clients


class AFVLMDataModule:
    """One loader for all three immutable client-count settings."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.tasks, partitions = scan_partitions(config)
        self.clients = {item.client_id: item for item in partitions}

    def get_client_ids(self) -> list[str]:
        return list(self.clients)

    def get_client_task(self, client_id: str) -> str:
        return self.clients[client_id].task

    def get_client_dataset(self, client_id: str) -> str:
        return self.clients[client_id].dataset

    def get_client_train_data(self, client_id: str) -> list[Sample]:
        item = self.clients[client_id]
        return self.tasks[item.task].load_file(item.train_file, "train")

    def get_client_num_samples(self, client_id: str) -> int:
        return self.clients[client_id].num_samples

    def get_task_val_data(self, task: str) -> list[Sample]:
        return self.tasks[task].load_split("validation")

    def get_task_test_data(self, task: str) -> list[Sample]:
        return self.tasks[task].load_split("final")

    def get_global_eval_sets(self) -> dict[str, dict[str, list[Sample]]]:
        return {
            task: {
                "validation": self.get_task_val_data(task),
                "test": self.get_task_test_data(task),
            }
            for task in self.tasks
        }
