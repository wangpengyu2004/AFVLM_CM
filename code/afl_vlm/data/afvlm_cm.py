"""Read-only adapter for the existing six-task AFVLM-CM partitions."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from collections.abc import Callable, Mapping
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
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Image path escapes image_root: {relative}")
    image_root = Path(str(dataset_config["image_root"]))
    image = image_root / relative_path
    if bool(dataset_config.get("require_images", True)):
        resolved_root = image_root.resolve()
        resolved_image = image.resolve()
        if not resolved_image.is_relative_to(resolved_root):
            raise ValueError(f"Image path escapes image_root through a symlink: {relative}")
        if not resolved_image.is_file():
            raise FileNotFoundError(
                f"Missing image: {resolved_image}. Run tools/download_afvlm_cm_images.py "
                "for the required source."
            )
        return resolved_image
    return image


def _fcit_text(text: str) -> str:
    """FCIT's task evaluators compare stripped, case-insensitive strings."""

    return str(text).strip().upper()


def _load_coco_caption_components() -> tuple[Any, Any, Any, Any, Any]:
    """Load the FCIT caption evaluator with one actionable dependency error."""

    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Caption evaluation requires pycocoevalcap==1.2. Run "
            "`python -m pip install pycocoevalcap==1.2`, or reinstall the current "
            "requirements.txt, before starting training."
        ) from exc
    return Bleu, Cider, Meteor, Rouge, PTBTokenizer


def validate_caption_metric_runtime() -> None:
    """Fail fast before training if the exact FCIT caption metric cannot run."""

    _load_coco_caption_components()
    if shutil.which("java") is None:
        raise RuntimeError(
            "Caption METEOR evaluation requires a Java runtime. Install Java and "
            "ensure the `java` executable is available on PATH before starting training."
        )


def coco_caption_metrics(
    predictions: list[str], references: list[tuple[str, ...]]
) -> dict[str, float]:
    """Run the same non-SPICE COCO caption scorers used by FCIT.

    FCIT reports each raw scorer value multiplied by 100 and averages BLEU-1
    through BLEU-4, METEOR, ROUGE-L, and CIDEr.  SPICE is intentionally not
    included in the FCIT score.
    """

    Bleu, Cider, Meteor, Rouge, PTBTokenizer = _load_coco_caption_components()

    ground_truth = {
        index: [{"caption": caption} for caption in captions]
        for index, captions in enumerate(references)
    }
    results = {
        index: [{"caption": prediction}]
        for index, prediction in enumerate(predictions)
    }
    tokenizer = PTBTokenizer()
    ground_truth = tokenizer.tokenize(ground_truth)
    results = tokenizer.tokenize(results)
    scorers: list[tuple[Any, str | tuple[str, ...]]] = [
        (Bleu(4), ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4")),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE-L"),
        (Cider(), "CIDEr"),
    ]
    metrics: dict[str, float] = {}
    try:
        for scorer, names in scorers:
            score, _ = scorer.compute_score(ground_truth, results)
            if isinstance(names, tuple):
                for name, value in zip(names, score, strict=True):
                    metrics[name] = 100.0 * float(value)
            else:
                metrics[names] = 100.0 * float(score)
    finally:
        for scorer, _ in scorers:
            close = getattr(scorer, "close", None)
            if callable(close):
                close()
    metrics["FCIT_caption_average"] = sum(metrics.values()) / len(metrics)
    return metrics


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
        evaluation = config.get("evaluation", {})
        self.grounding_iou_threshold = float(evaluation.get("grounding_iou_threshold", 0.5))
        self._by_id: dict[str, Sample] = {}
        self._split_cache: dict[str, list[Sample]] = {}

    @staticmethod
    def _records(path: Path) -> list[dict[str, Any]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list: {path}")
        for index, record in enumerate(payload):
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}[{index}]")
        return payload

    @staticmethod
    def _required_text(value: Any, field: str) -> str:
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"{field} must be a scalar value")
        text = str(value).strip()
        if not text:
            raise ValueError(f"{field} must not be empty")
        return text

    def _conversation_pairs(self, record: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
        """Read both official AFVLM-CM annotation representations.

        FCIT-derived partitions mix LLaVA ``conversations`` records with flat
        ``text``/``answer`` evaluation records. Grounding test records use
        ``answer_bbox`` for the normalized target box. Multi-turn grounding
        Training preserves every user/assistant pair. Evaluation expansion is
        handled in :meth:`load_file` so each Grounding query receives one
        prediction and one IoU score.
        """
        if "conversations" in record:
            conversations = record["conversations"]
            if not isinstance(conversations, list) or len(conversations) < 2:
                raise ValueError("conversations must contain at least one user/assistant pair")
            if len(conversations) % 2:
                raise ValueError("conversations must contain complete user/assistant pairs")
            values: list[str] = []
            for index, message in enumerate(conversations):
                if not isinstance(message, Mapping):
                    raise ValueError(f"conversations[{index}] must be an object")
                role = str(message.get("from", message.get("role", ""))).lower().strip()
                allowed = {"human", "user"} if index % 2 == 0 else {"gpt", "assistant"}
                if role not in allowed:
                    expected = "user/human" if index % 2 == 0 else "assistant/gpt"
                    raise ValueError(f"conversations[{index}] must have role {expected}")
                values.append(
                    self._required_text(message.get("value"), f"conversations[{index}].value")
                )
            return tuple(
                (values[index].replace("<image>", "").strip(), values[index + 1])
                for index in range(0, len(values), 2)
            )

        if "text" in record:
            instruction = self._required_text(record["text"], "text")
            answer_key = "answer_bbox" if "answer_bbox" in record else "answer"
            if answer_key not in record:
                raise ValueError("flat annotation must contain answer or answer_bbox")
            answer = self._required_text(record[answer_key], answer_key)
            return ((instruction.replace("<image>", "").strip(), answer),)

        raise ValueError(
            "record must use conversations or the flat text + answer/answer_bbox format"
        )

    def _instruction_answer(self, record: Mapping[str, Any]) -> tuple[str, str]:
        """Return the first pair for compatibility with callers inspecting records."""

        return self._conversation_pairs(record)[0]

    def _validated_record(
        self,
        record: Mapping[str, Any],
        *,
        canonicalize_image: bool = True,
        verify_image: bool = False,
    ) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...], Path]:
        turns = self._conversation_pairs(record)
        raw_references = record.get("references", ())
        if not isinstance(raw_references, (list, tuple)):
            raise ValueError("references must be a list when present")
        references = tuple(
            self._required_text(value, f"references[{index}]")
            for index, value in enumerate(raw_references)
        )
        image_value = record.get("image")
        if not isinstance(image_value, str) or not image_value.strip():
            raise ValueError("image must be a non-empty relative path")
        if canonicalize_image:
            image = resolve_image_path(
                record,
                {
                    "image_root": self.image_root,
                    "require_images": self.require_images and verify_image,
                },
            )
        else:
            relative = image_value.replace("\\", "/")
            while relative.startswith("./"):
                relative = relative[2:]
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Image path escapes image_root: {relative}")
            image = self.image_root / relative_path
            if verify_image and self.require_images and not image.is_file():
                raise FileNotFoundError(
                    f"Missing image: {image}. Run tools/download_afvlm_cm_images.py "
                    "for the required source."
                )
        return turns, references, image

    def _sample(
        self,
        record: Mapping[str, Any],
        split: str,
        index: int,
        turn_index: int | None = None,
    ) -> Sample:
        turns, references, image = self._validated_record(record)
        selected_turns = turns if turn_index is None else (turns[turn_index],)
        instruction, answer = selected_turns[0]
        sample_id = str(record.get("id") or record.get("question_id") or index)
        suffix = f":turn{turn_index}" if turn_index is not None else ""
        sample = Sample(
            id=f"{self.task_key}:{split}:{sample_id}:{index}{suffix}",
            task_name=self.task_key,
            image=str(image),
            instruction=instruction,
            answer=answer,
            split=split,
            turns=selected_turns,
            references=references,
            metadata={
                "dataset": self.dataset_name,
                "source_id": sample_id,
                "raw_record": dict(record),
            },
        )
        self._by_id[sample.id] = sample
        return sample

    def validate_file(
        self,
        path: str | Path,
        split: str,
        *,
        verify_images: bool = True,
        image_origins: dict[Path, str] | None = None,
        progress_callback: Callable[[], None] | None = None,
    ) -> int:
        """Validate every annotation and optionally collect/check its image path."""
        source = Path(path).resolve()
        records = self._records(source)
        for index, record in enumerate(records):
            try:
                _, references, image = self._validated_record(
                    record,
                    canonicalize_image=False,
                    verify_image=verify_images,
                )
                if self.task_key == "caption" and split != "train" and len(references) != 5:
                    raise ValueError(
                        "Caption evaluation records must contain exactly five references"
                    )
                if image_origins is not None:
                    image_origins.setdefault(
                        image,
                        f"{self.task_key}/{split} at {source}[{index}]",
                    )
            except (FileNotFoundError, ValueError) as exc:
                error_type = FileNotFoundError if isinstance(exc, FileNotFoundError) else ValueError
                raise error_type(
                    f"Invalid AFVLM-CM {self.task_key}/{split} annotation "
                    f"at {source}[{index}]: {exc}"
                ) from exc
            if progress_callback is not None:
                progress_callback()
        return len(records)

    def load_file(self, path: str | Path, split: str = "train") -> list[Sample]:
        source = Path(path).resolve()
        samples: list[Sample] = []
        for index, record in enumerate(self._records(source)):
            turns = self._conversation_pairs(record)
            if self.task_key == "grounding" and split != "train" and len(turns) > 1:
                samples.extend(
                    self._sample(record, split, index, turn_index)
                    for turn_index in range(len(turns))
                )
            else:
                samples.append(self._sample(record, split, index))
        if self.task_key == "caption" and split != "train":
            invalid = [sample.id for sample in samples if len(sample.references) != 5]
            if invalid:
                raise ValueError(
                    "Caption evaluation requires exactly five references per image; "
                    f"invalid samples include {invalid[:3]}. Regenerate AFVLM-CM partitions."
                )
        return samples

    def load_split(self, split: str) -> list[Sample]:
        if split in self._split_cache:
            return list(self._split_cache[split])
        filename = {"validation": "val", "final": "test"}.get(split, split)
        samples = self.load_file(self.partition_dir / f"{filename}.json", split)
        self._split_cache[split] = samples
        return list(samples)

    def format_sample(self, sample: Sample) -> tuple[list[dict[str, Any]], str | None]:
        messages: list[dict[str, Any]] = []
        turns = sample.turns or ((sample.instruction, sample.answer),)
        for index, (instruction, answer) in enumerate(turns):
            content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
            if index == 0:
                content.insert(0, {"type": "image"})
            messages.append({"role": "user", "content": content})
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            )
        return messages, sample.image

    def collate(self, samples: list[Sample], model_adapter: Any) -> list[Sample]:
        return samples

    def metric(self, predictions: list[str], references: list[Any]) -> dict[str, float]:
        if not predictions or len(predictions) != len(references):
            raise ValueError("Predictions and references must have equal non-zero length")
        if self.task_key == "caption":
            if any(not isinstance(reference, (list, tuple)) for reference in references):
                raise ValueError("Caption references must be lists of five strings")
            caption_references = [
                tuple(str(item) for item in reference) for reference in references
            ]
            if any(len(reference) != 5 for reference in caption_references):
                raise ValueError("FCIT Caption evaluation requires five references per prediction")
            return coco_caption_metrics(predictions, caption_references)
        if self.task_key == "grounding":
            values = [
                _iou(pred, ref) if pred is not None and ref is not None else 0.0
                for pred, ref in zip(map(_bbox, predictions), map(_bbox, references), strict=True)
            ]
            return {
                "mean_IoU": sum(values) / len(values),
                "IoU@0.5_accuracy": 100.0
                * sum(v > self.grounding_iou_threshold for v in values)
                / len(values),
            }
        normalized_references = [str(reference) for reference in references]
        if self.task_key in {"chart_vqa", "visual_reasoning"}:
            correct = sum(
                _fcit_text(prediction) in _fcit_text(reference)
                for prediction, reference in zip(
                    predictions, normalized_references, strict=True
                )
            )
        else:
            correct = sum(
                _fcit_text(prediction) == _fcit_text(reference)
                for prediction, reference in zip(
                    predictions, normalized_references, strict=True
                )
            )
        accuracy = 100.0 * correct / len(references)
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
                "evaluation": dataset_config.get("evaluation", {}),
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

    def preflight_validate(self, *, progress: bool = False) -> dict[str, int]:
        """Explicitly validate all records and unique images without loading LLaVA."""
        counts = {"files": 0, "train": 0, "validation": 0, "test": 0}
        required_images: dict[Path, str] = {}
        annotation_bar = None
        if progress:
            from tqdm.auto import tqdm

            annotation_bar = tqdm(
                desc="validate AFVLM-CM annotations",
                unit="record",
                dynamic_ncols=True,
            )
        advance = annotation_bar.update if annotation_bar is not None else None
        try:
            for partition in self.clients.values():
                adapter = self.tasks[partition.task]
                counts["train"] += adapter.validate_file(
                    partition.train_file,
                    "train",
                    verify_images=False,
                    image_origins=required_images if adapter.require_images else None,
                    progress_callback=(lambda: advance(1)) if advance is not None else None,
                )
                counts["files"] += 1
            for adapter in self.tasks.values():
                for split, filename in (("validation", "val.json"), ("test", "test.json")):
                    counts[split] += adapter.validate_file(
                        adapter.partition_dir / filename,
                        split,
                        verify_images=False,
                        image_origins=required_images if adapter.require_images else None,
                        progress_callback=(lambda: advance(1)) if advance is not None else None,
                    )
                    counts["files"] += 1
        finally:
            if annotation_bar is not None:
                annotation_bar.close()

        image_bar = None
        if progress:
            from tqdm.auto import tqdm

            image_bar = tqdm(
                total=len(required_images),
                desc="validate unique AFVLM-CM images",
                unit="image",
                dynamic_ncols=True,
            )
        try:
            image_root = Path(str(self.config["image_root"])).resolve()
            for image, origin in required_images.items():
                resolved_image = image.resolve()
                if not resolved_image.is_relative_to(image_root):
                    raise ValueError(
                        f"Image path referenced by {origin} escapes image_root through a "
                        f"symlink: {image}"
                    )
                if not resolved_image.is_file():
                    raise FileNotFoundError(
                        f"Missing image referenced by {origin}: {resolved_image}. "
                        "Run tools/download_afvlm_cm_images.py for the required source."
                    )
                if image_bar is not None:
                    image_bar.update(1)
        finally:
            if image_bar is not None:
                image_bar.close()
        counts["records"] = counts["train"] + counts["validation"] + counts["test"]
        counts["unique_images"] = len(required_images)
        return counts

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
