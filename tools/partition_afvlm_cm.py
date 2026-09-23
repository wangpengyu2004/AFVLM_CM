#!/usr/bin/env python3
"""Build the fixed-task AFVLM-CM benchmark used by every baseline.

The generator is deterministic and writes a versioned candidate directory by
default.  It never replaces ``data/AFVLM_CM/partitioned`` implicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    task: str
    dataset: str
    train_file: str
    final_file: str
    train_cap: int


TASKS = (
    TaskSpec("cls", "ImageNet-R", "ImageNet-R/train.json", "ImageNet-R/test.json", 13_000),
    TaskSpec(
        "caption",
        "Flickr30k",
        "Flickr30k-cap/train_brief_4w.json",
        "Flickr30k-cap/val_brief.json",
        15_000,
    ),
    TaskSpec("vqa", "AOKVQA", "AOKVQA/train.json", "AOKVQA/val.json", 11_000),
    TaskSpec("chart_vqa", "DVQA", "DVQA/train.json", "DVQA/test.json", 9_500),
    TaskSpec(
        "visual_reasoning",
        "FigureQA",
        "FigureQA/train.json",
        "FigureQA/test.json",
        11_500,
    ),
    TaskSpec(
        "grounding",
        "Grounding",
        "Grounding/train_11w.json",
        "Grounding/test_6000.json",
        12_000,
    ),
)


def stable_seed(seed: int, *parts: object) -> int:
    material = "::".join(str(item) for item in (seed, *parts))
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")


def canonical(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def record_hash(record: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(record).encode()).hexdigest()


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing AFVLM-CM source annotation: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_records(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise ValueError(f"Expected a JSON list of objects: {path}")
    return payload


def write_json(path: Path, value: Any, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def image_key(record: dict[str, Any]) -> str:
    image = record.get("image", record.get("image_path"))
    if not isinstance(image, str) or not image.strip():
        raise ValueError(f"Record has no image path: {record}")
    return image.replace("\\", "/").removeprefix("./")


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for record in records:
        digest = record_hash(record)
        if digest not in seen:
            seen.add(digest)
            result.append(record)
    return result


def deterministic_subset(
    records: list[dict[str, Any]], size: int, seed: int, *parts: object
) -> list[dict[str, Any]]:
    if size > len(records):
        raise ValueError(f"Requested {size} records but only {len(records)} are available")
    indices = list(range(len(records)))
    random.Random(stable_seed(seed, *parts)).shuffle(indices)
    return [records[index] for index in sorted(indices[:size])]


def carve_validation(
    train: list[dict[str, Any]], size: int, seed: int, task: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Carve a question/record-level validation split from training."""

    selected = deterministic_subset(train, size, seed, task, "validation")
    selected_hashes = {record_hash(record) for record in selected}
    remaining = [record for record in train if record_hash(record) not in selected_hashes]
    return remaining, selected


def load_flickr_eval(source_root: Path) -> list[dict[str, Any]]:
    brief = read_records(source_root / "Flickr30k-cap/val_brief.json")
    coco = read_json(source_root / "Flickr30k-cap/val_coco_type.json")
    if not isinstance(coco, dict):
        raise ValueError("Flickr30k val_coco_type.json must be a COCO object")
    image_id_by_filename = {
        str(item["file_name"]): int(item["id"]) for item in coco.get("images", [])
    }
    captions: dict[int, list[str]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        captions[int(annotation["image_id"])].append(str(annotation["caption"]))

    enriched: list[dict[str, Any]] = []
    for record in brief:
        filename = Path(image_key(record)).name
        if filename not in image_id_by_filename:
            raise ValueError(f"Flickr30k COCO references do not contain {filename}")
        references = captions[image_id_by_filename[filename]]
        if len(references) != 5:
            raise ValueError(f"Flickr30k {filename} has {len(references)} references, expected 5")
        item = dict(record)
        item["references"] = references
        enriched.append(item)
    return enriched


def generate_client_sizes(
    total: int,
    clients: int,
    seed: int,
    task: str,
    min_ratio: float,
    max_ratio: float,
) -> list[int]:
    average = total / clients
    minimum = int(min_ratio * average)
    maximum = min(10_000, int(max_ratio * average + 0.999999))
    if total < clients * minimum or total > clients * maximum:
        raise ValueError(
            f"Cannot allocate {task}/{total} to {clients} clients within "
            f"[{minimum}, {maximum}]"
        )
    rng = random.Random(stable_seed(seed, task, clients, "client_sizes"))
    remaining = total
    sizes: list[int] = []
    for index in range(clients):
        left = clients - index - 1
        lower = max(minimum, remaining - left * maximum)
        upper = min(maximum, remaining - left * minimum)
        if left == 0:
            choice = remaining
        else:
            center = min(upper, max(lower, round(remaining / (left + 1))))
            choice = round(rng.triangular(lower, upper, center))
        sizes.append(choice)
        remaining -= choice
    if clients > 1 and len(set(sizes)) == 1:
        if sizes[0] > minimum and sizes[1] < maximum:
            sizes[0] -= 1
            sizes[1] += 1
        else:
            raise ValueError(f"{task}/{clients} clients cannot be made quantity-heterogeneous")
    return sizes


def partition_pool(
    pool: list[dict[str, Any]], sizes: list[int], seed: int, task: str
) -> list[list[dict[str, Any]]]:
    shuffled = list(pool)
    random.Random(stable_seed(seed, task, len(sizes), "partition")).shuffle(shuffled)
    partitions: list[list[dict[str, Any]]] = []
    offset = 0
    for size in sizes:
        partitions.append(shuffled[offset : offset + size])
        offset += size
    if offset != len(pool):
        raise AssertionError("Client partitions did not consume the complete fixed task pool")
    return partitions


def validate_task(
    task: str,
    pool: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    final: list[dict[str, Any]],
    partitions: dict[int, list[list[dict[str, Any]]]],
) -> None:
    pool_hashes = {record_hash(item) for item in pool}
    if len(pool_hashes) != len(pool):
        raise AssertionError(f"{task}: duplicate records in selected training pool")
    val_hashes = {record_hash(item) for item in validation}
    final_hashes = {record_hash(item) for item in final}
    if pool_hashes & val_hashes or pool_hashes & final_hashes or val_hashes & final_hashes:
        raise AssertionError(f"{task}: train/validation/final record leakage detected")
    for clients, client_partitions in partitions.items():
        hashes = [{record_hash(item) for item in part} for part in client_partitions]
        overlaps = any(
            hashes[left] & hashes[right]
            for left in range(clients)
            for right in range(left + 1, clients)
        )
        if overlaps:
            raise AssertionError(f"{task}/{clients}: duplicated records across clients")
        if set().union(*hashes) != pool_hashes:
            raise AssertionError(f"{task}/{clients}: client union differs from fixed task pool")


def prepare_task(
    spec: TaskSpec,
    source_root: Path,
    eval_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_source = deduplicate(read_records(source_root / spec.train_file))
    if spec.task == "caption":
        official = load_flickr_eval(source_root)
        selected = deterministic_subset(official, 2 * eval_size, seed, spec.task, "eval")
        validation, final = selected[:eval_size], selected[eval_size:]
        candidates = train_source
    else:
        official = deduplicate(read_records(source_root / spec.final_file))
        final = deterministic_subset(official, eval_size, seed, spec.task, "final")
        final_hashes = {record_hash(record) for record in final}
        candidates = [record for record in train_source if record_hash(record) not in final_hashes]
        candidates, validation = carve_validation(candidates, eval_size, seed, spec.task)

    eval_hashes = {record_hash(item) for item in (*validation, *final)}
    candidates = [record for record in candidates if record_hash(record) not in eval_hashes]
    pool = deterministic_subset(candidates, spec.train_cap, seed, spec.task, "train_pool")
    details = {
        "dataset": spec.dataset,
        "train_source": spec.train_file,
        "final_source": spec.final_file,
        "source_train_records": len(train_source),
        "candidate_train_records": len(candidates),
        "selected_train_records": len(pool),
        "validation_records": len(validation),
        "final_records": len(final),
    }
    return pool, validation, final, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_root",
        type=Path,
        default=Path("data/AFVLM_CM/instruction"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/AFVLM_CM/partitioned_fcitev1_72k"),
    )
    parser.add_argument("--client_settings", nargs="+", type=int, default=[2, 5, 10])
    parser.add_argument("--eval_size", type=int, default=1_000)
    parser.add_argument("--min_client_ratio", type=float, default=0.7)
    parser.add_argument("--max_client_ratio", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Candidate output already exists and is not empty: {output_dir}. "
            "Choose a new --output_dir; existing partitions are never overwritten."
        )
    if args.eval_size <= 0:
        raise ValueError("--eval_size must be positive")
    if sorted(set(args.client_settings)) != sorted(args.client_settings):
        raise ValueError("--client_settings must contain unique positive values")
    if any(value <= 0 for value in args.client_settings):
        raise ValueError("--client_settings must contain unique positive values")

    summary: dict[str, Any] = {
        "version": "fcit_eval_v1_72k",
        "seed": args.seed,
        "task_order": [spec.task for spec in TASKS],
        "total_selected_train_records": sum(spec.train_cap for spec in TASKS),
        "tasks": {},
    }
    report_rows: list[str] = []
    for spec in TASKS:
        pool, validation, final, details = prepare_task(
            spec, source_root, args.eval_size, args.seed
        )
        settings: dict[int, list[list[dict[str, Any]]]] = {}
        client_sizes: dict[str, list[int]] = {}
        for clients in args.client_settings:
            sizes = generate_client_sizes(
                len(pool),
                clients,
                args.seed,
                spec.task,
                args.min_client_ratio,
                args.max_client_ratio,
            )
            parts = partition_pool(pool, sizes, args.seed, spec.task)
            settings[clients] = parts
            client_sizes[str(clients)] = sizes
            task_dir = output_dir / f"{clients}_clients" / spec.task
            for index, records in enumerate(parts):
                write_json(task_dir / f"client_{index}.json", records)
            write_json(task_dir / "val.json", validation)
            write_json(task_dir / "test.json", final)
            write_json(
                task_dir / "statistics.json",
                {
                    **details,
                    "num_clients": clients,
                    "client_samples": {
                        f"client_{index}": size for index, size in enumerate(sizes)
                    },
                    "mean_client_samples": mean(sizes),
                    "quantity_heterogeneous": len(set(sizes)) > 1,
                    "seed": args.seed,
                },
                pretty=True,
            )
        validate_task(spec.task, pool, validation, final, settings)
        summary["tasks"][spec.task] = {**details, "client_sizes": client_sizes}
        report_rows.append(
            f"| {spec.task} | {spec.dataset} | {len(pool):,} | "
            f"{len(validation):,} / {len(final):,} | "
            f"{' ; '.join(f'{key}: {value}' for key, value in client_sizes.items())} |"
        )

    write_json(output_dir / "metadata/dataset_summary.json", summary, pretty=True)
    report = "\n".join(
        [
            "# AFVLM-CM FCIT-compatible 72k partition",
            "",
            f"Seed: {args.seed}",
            "",
            "| Task | Dataset | Train records | Validation / final records | Client sizes |",
            "|---|---|---:|---:|---|",
            *report_rows,
            "",
            "All client-count settings reuse the same fixed task pool. Client sizes are "
            "quantity-heterogeneous, and train/validation/final records are disjoint per task.",
            "Official FCIT question-level splits are retained; DVQA, FigureQA, and Grounding may "
            "legitimately contain different questions/regions for the same source image.",
            "Grounding validation records remain multi-turn on disk and are expanded into "
            "independent queries by the runtime evaluator.",
            "Caption validation/final records each contain exactly five references.",
            "",
        ]
    )
    (output_dir / "metadata/partition_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Candidate partition written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
