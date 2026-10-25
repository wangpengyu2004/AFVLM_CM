"""Preserve completed rank experiments when extending or combining their results.

Only configuration/annotation metadata and recorded numeric results are read here.
No checkpoint, image, torch/CUDA, model, or training code is loaded or executed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from afl_vlm.config import load_config
from afl_vlm.experiments.rank_probe import (
    TASK_METRICS,
    ProbeJob,
    export_probe_results,
    hash_file,
    steps_per_epoch,
    validate_probe_config,
    write_json,
)

SHARED_SOURCES = ("code/afl_vlm/models/llava15.py", "code/afl_vlm/data/afvlm_cm.py",
                  "code/afl_vlm/evaluation.py")


def read_completed_run(root: Path) -> tuple[dict, dict, dict]:
    """Check frozen input hashes without requiring original images or weights."""
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    if (manifest.get("kind") not in {"local_rank_probe", "local_rank_probe_merged"}
            or manifest["status"] != "completed" or not manifest["jobs"]
            or any(job["status"] != "completed" for job in manifest["jobs"])):
        raise ValueError(f"A fully completed rank experiment is required: {root}")
    for name, field in (("resolved_config.yaml", "config_sha256"),
                        ("data_manifest.json", "data_sha256")):
        if hash_file(root / name) != manifest[field]:
            raise ValueError(f"Frozen input changed: {root / name}")
    config = load_config(root / "resolved_config.yaml")
    validate_probe_config(config)
    data = json.loads((root / "data_manifest.json").read_text("utf-8"))
    if manifest["local_epochs"] != config["training"]["local_epochs"]:
        raise ValueError(f"Inconsistent epoch budget: {root}")
    return manifest, config, data


def comparison_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    """Ranks/output/UI may differ; training/data/evaluation recipes must match."""
    result = copy.deepcopy({key: value for key, value in config.items()
                            if not key.startswith("_") and key != "output"})
    result["experiment"].pop("ranks")
    result["experiment"].pop("name")
    for key in ("r", "alpha"):
        result["model"]["lora"].pop(key, None)
    # Process startup timeout is not an optimizer or evaluation setting.
    result["runtime"] = {"world_size": result["runtime"]["world_size"]}
    return result


def data_signature(data: Mapping[str, Any]) -> dict[str, Any]:
    return {"source_files": data["source_files"], "tasks": data["tasks"]}


def shared_source_signature(manifest: Mapping[str, Any]) -> dict[str, str]:
    return {path: manifest["source_sha256"][path] for path in SHARED_SOURCES}


def validate_extension(root: Path, config: Mapping[str, Any],
                       data: Mapping[str, Any], jobs: Sequence[ProbeJob], repo: Path) -> dict:
    """Refuse changed data/training code and previously completed jobs; never write root."""
    manifest, old_config, old_data = read_completed_run(root)
    if comparison_signature(config) != comparison_signature(old_config):
        raise ValueError("Continuation changed the original training/evaluation recipe")
    if data_signature(data) != data_signature(old_data):
        raise ValueError("Continuation data/validation differs from the completed experiment")
    if shared_source_signature(manifest) != {path: hash_file(repo / path)
                                            for path in SHARED_SOURCES}:
        raise ValueError(
            "Original model/trainer/data/metric source changed; review before combining"
        )
    existing = {job["job_id"] for job in manifest["jobs"]}
    duplicates = existing.intersection(job.key for job in jobs)
    if duplicates:
        raise ValueError(f"Refusing to retrain completed jobs: {sorted(duplicates)}")
    return {"run_directory": str(root), "manifest_sha256": hash_file(root / "manifest.json")}


def _validated_job_files(root: Path, job: Mapping[str, Any], config: Mapping[str, Any],
                         data: Mapping[str, Any]) -> dict[str, bytes]:
    expected = ProbeJob(**{key: job[key] for key in (
        "task", "client", "lora_rank", "seed", "lora_alpha",
    )})
    if (expected.key != job["job_id"] or expected.task not in config["dataset"]["tasks"]
            or expected.client not in config["experiment"]["clients"]
            or expected.seed not in config["experiment"]["seeds"]
            or expected.lora_rank not in config["experiment"]["ranks"]
            or expected.lora_alpha != config["experiment"]["lora_alpha_per_rank"]
            * expected.lora_rank):
        raise ValueError(f"Invalid recorded job: {root}, {job}")
    folder = root / expected.key
    required = ("metrics.jsonl", "result.json", "metadata.json")
    optional = ("training_steps.jsonl", "epoch_training.jsonl", "job_config.yaml",
                "expected_sampler.json")
    files = {name: (folder / name).read_bytes() for name in required}
    files.update({name: (folder / name).read_bytes() for name in optional
                  if (folder / name).is_file()})
    epochs = config["training"]["local_epochs"]
    task_data = data["tasks"][expected.task]
    count = task_data["client_records"][expected.client]
    per_epoch = steps_per_epoch(count, config["training"]["batch_size"],
        config["training"]["gradient_accumulation"], config["runtime"]["world_size"])
    result = json.loads(files["result.json"])
    if (result["job_id"] != expected.key or result["status"] != "completed"
            or result["optimizer_steps"] != epochs * per_epoch
            or result["completed_epochs"] != list(range(1, epochs + 1))):
        raise ValueError(f"Incomplete optimizer/epoch budget: {folder}")
    seen = set()
    for line in files["metrics.jsonl"].decode("utf-8").splitlines():
        record = json.loads(line)
        epoch, protocol = record["epoch"], record["protocol"]
        identity = (epoch, protocol)
        if identity in seen or epoch not in range(1, epochs + 1):
            raise ValueError(f"Duplicate or unexpected evaluation: {folder}, {identity}")
        seen.add(identity)
        if (protocol not in task_data["subsets"]
                or any(record[key] != value for key, value in expected.as_dict().items())
                or record["num_train_samples"] != count
                or record["optimizer_steps"] != epoch * per_epoch
                or record["num_eval_samples"] != len(task_data["subsets"][protocol])
                or record["primary"] != (protocol == task_data["primary_protocol"])
                or record["train_annotation_sha256"] != data["source_files"][
                    f"{expected.task}/{expected.client}.json"]
                or record["validation_ids_sha256"] != task_data["subset_ids_sha256"][protocol]
                or TASK_METRICS[expected.task] not in record["metrics"]
                or any(not math.isfinite(float(value)) for value in record["metrics"].values())):
            raise ValueError(f"Inconsistent metric budget/protocol/identity: {folder}, {identity}")
    if seen != {(epoch, protocol) for epoch in range(1, epochs + 1)
                for protocol in task_data["subsets"]}:
        raise ValueError(f"Missing epoch/protocol metrics: {folder}")
    return files


def merge_probe_runs(sources: Sequence[Path], output: Path) -> dict[str, int]:
    """Copy only small records to a NEW directory; sources are never edited/exported."""
    roots = [source.resolve() for source in sources]
    output = output.resolve()
    if len(roots) < 2 or len(set(roots)) != len(roots):
        raise ValueError("Supply at least two different completed experiment directories")
    if output.exists() or any(output == root or root in output.parents for root in roots):
        raise FileExistsError("Combined output must be new and outside every source directory")
    base_manifest, base_config, base_data = read_completed_run(roots[0])
    combined, snapshots = [], []
    seen_jobs, versions = set(), None
    paired_ranks: dict[tuple, set[int]] = defaultdict(set)
    for root in roots:
        manifest, config, data = read_completed_run(root)
        if (comparison_signature(config) != comparison_signature(base_config)
                or data_signature(data) != data_signature(base_data)
                or shared_source_signature(manifest) != shared_source_signature(base_manifest)):
            raise ValueError(f"Different training/data/evaluation recipe or shared sources: {root}")
        snapshots.append({"run_directory": str(root),
                          "manifest_sha256": hash_file(root / "manifest.json")})
        for job in manifest["jobs"]:
            if job["job_id"] in seen_jobs:
                raise ValueError(f"Duplicate job across source experiments: {job['job_id']}")
            seen_jobs.add(job["job_id"])
            files = _validated_job_files(root, job, config, data)
            observed_versions = json.loads(files["metadata.json"])["versions"]
            if versions is None:
                versions = observed_versions
            elif observed_versions != versions:
                raise ValueError(
                    f"Package versions differ across rank jobs: {root}/{job['job_id']}"
                )
            paired_ranks[job["task"], job["client"], job["seed"]].add(job["lora_rank"])
            combined.append(({**job, "source_run": str(root)}, files))
    ranks = sorted({rank for group in paired_ranks.values() for rank in group})
    if any(group != set(ranks) for group in paired_ranks.values()):
        raise ValueError("Incomplete paired rank matrix: each client/seed needs all combined ranks")
    combined.sort(key=lambda item: (list(TASK_METRICS).index(item[0]["task"]),
                                    item[0]["client"], item[0]["seed"], item[0]["lora_rank"]))
    # All source validation completes before any output is created. Copy the validated
    # bytes, not a later re-read of possibly changed source records.
    output.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy({key: value for key, value in base_config.items()
                            if not key.startswith("_")})
    config["experiment"]["ranks"] = ranks
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    write_json(output / "data_manifest.json", base_data)
    manifest = {"schema_version": 1, "kind": "local_rank_probe_merged", "status": "merging",
        "local_epochs": config["training"]["local_epochs"], "world_size": 8,
        "source_runs": snapshots, "source_sha256": shared_source_signature(base_manifest),
        "config_sha256": hash_file(output / "resolved_config.yaml"),
        "data_sha256": hash_file(output / "data_manifest.json"), "jobs": [],
        "copied_files_sha256": {}}
    for job, files in combined:
        folder = output / job["job_id"]
        folder.mkdir(parents=True, exist_ok=False)
        for name, content in files.items():
            (folder / name).write_bytes(content)
            manifest["copied_files_sha256"][f"{job['job_id']}/{name}"] = hashlib.sha256(
                content).hexdigest()
        manifest["jobs"].append(job)
    write_json(output / "manifest.json", manifest)
    try:
        exported = export_probe_results(output)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "manifest.json", manifest)
        raise
    manifest["status"] = "completed"
    write_json(output / "manifest.json", manifest)
    return {"jobs": len(combined), "ranks": len(ranks), **exported}
