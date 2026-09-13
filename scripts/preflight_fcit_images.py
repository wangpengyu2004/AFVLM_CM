#!/usr/bin/env python3
"""Verify that every image referenced by a generated FCIT benchmark exists and is readable."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("data/fcit/fixed_task_benchmark"),
    )
    parser.add_argument("--image-root", type=Path, help="Override benchmark_summary image_root.")
    parser.add_argument("--report", type=Path, help="Output JSON report path.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write the report and exit successfully even when images are missing.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def has_valid_header(path: Path) -> bool:
    try:
        header = path.read_bytes()[:16]
    except OSError:
        return False
    suffix = path.suffix.lower()
    if suffix == ".png":
        return header.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix in {".jpg", ".jpeg"}:
        return header.startswith(b"\xff\xd8\xff")
    if suffix == ".gif":
        return header.startswith((b"GIF87a", b"GIF89a"))
    if suffix == ".webp":
        return header.startswith(b"RIFF") and header[8:12] == b"WEBP"
    return bool(header)


def resolve_under_root(image_root: Path, relative_path: str) -> Path:
    raw = Path(relative_path)
    resolved = raw.resolve() if raw.is_absolute() else (image_root / raw).resolve()
    if not resolved.is_relative_to(image_root):
        raise ValueError(f"Image reference escapes image root: {relative_path}")
    return resolved


def collect_images(benchmark_root: Path, summary: dict[str, Any]) -> dict[str, set[str]]:
    images: dict[str, set[str]] = defaultdict(set)
    first_variant = benchmark_root / summary["variants"][0]["directory"]
    variant_summary = json.loads((first_variant / "summary.json").read_text(encoding="utf-8"))
    for client in read_jsonl(first_variant / variant_summary["client_manifest"]):
        for record in read_jsonl(first_variant / client["train_file"]):
            images[client["task_id"]].add(str(record["image"]).replace("\\", "/"))
    for task in summary["common_tasks"]:
        for key in ("probe_file", "final_file"):
            for record in read_jsonl(benchmark_root / task[key]):
                images[task["task_id"]].add(str(record["image"]).replace("\\", "/"))
    return images


def preflight(benchmark_root: Path, image_root: Path) -> dict[str, Any]:
    summary = json.loads((benchmark_root / "benchmark_summary.json").read_text(encoding="utf-8"))
    images = collect_images(benchmark_root, summary)
    task_names = {item["task_id"]: item["task_name"] for item in summary["common_tasks"]}
    task_reports: list[dict[str, Any]] = []
    for task_id in sorted(images, key=lambda value: int(value.removeprefix("Task"))):
        present = 0
        missing: list[str] = []
        unreadable: list[str] = []
        for relative in sorted(images[task_id]):
            path = resolve_under_root(image_root, relative)
            if not path.is_file():
                missing.append(relative)
            elif not has_valid_header(path):
                unreadable.append(relative)
            else:
                present += 1
        expected = len(images[task_id])
        task_reports.append(
            {
                "task_id": task_id,
                "task_name": task_names[task_id],
                "expected_unique_images": expected,
                "present_readable_images": present,
                "missing_images": len(missing),
                "unreadable_images": len(unreadable),
                "coverage": present / expected if expected else 1.0,
                "ready": not missing and not unreadable,
                "missing_examples": missing[:20],
                "unreadable_examples": unreadable[:20],
            }
        )
    expected_total = sum(item["expected_unique_images"] for item in task_reports)
    present_total = sum(item["present_readable_images"] for item in task_reports)
    return {
        "benchmark_root": benchmark_root.as_posix(),
        "image_root": image_root.as_posix(),
        "ready_for_training": all(item["ready"] for item in task_reports),
        "expected_unique_images": expected_total,
        "present_readable_images": present_total,
        "missing_images": sum(item["missing_images"] for item in task_reports),
        "unreadable_images": sum(item["unreadable_images"] for item in task_reports),
        "coverage": present_total / expected_total if expected_total else 1.0,
        "tasks": task_reports,
    }


def main() -> int:
    args = parse_args()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    summary = json.loads((benchmark_root / "benchmark_summary.json").read_text(encoding="utf-8"))
    image_root = (
        args.image_root.expanduser().resolve()
        if args.image_root
        else Path(summary["image_root"]).expanduser().resolve()
    )
    report = preflight(benchmark_root, image_root)
    report_path = (
        args.report.expanduser().resolve()
        if args.report
        else benchmark_root / "image_preflight.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_for_training"] or args.allow_missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
