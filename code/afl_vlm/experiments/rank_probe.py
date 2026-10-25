"""Pure configuration, data provenance and exports for the isolated local rank probe.

Importing this module never imports torch, constructs a model or starts training.
There is deliberately no method registry entry, server, TrainPlan or aggregation.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from afl_vlm.data.afvlm_cm import AFVLMTaskAdapter

TASK_METRICS = {"cls": "accuracy", "caption": "CIDEr", "vqa": "VQA_accuracy"}
SUPPORTED_RANKS = (2, 4, 8, 16, 32)
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


@dataclass(frozen=True)
class ProbeJob:
    task: str
    client: str
    lora_rank: int
    seed: int
    lora_alpha: int

    @property
    def key(self) -> str:
        return f"{self.task}/{self.client}/rank{self.lora_rank}/seed{self.seed}"

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "job_id": self.key}


def positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def validate_probe_config(config: Mapping[str, Any]) -> None:
    sections = {"experiment", "runtime", "dataset", "training", "model", "evaluation", "output"}
    visible = {key for key in config if not key.startswith("_")}
    if visible != sections or any(not isinstance(config[key], Mapping) for key in sections):
        raise ValueError(
            "Rank probe requires only its standalone sections; no method/federation/Plan"
        )
    experiment, dataset, training = (config[key] for key in ("experiment", "dataset", "training"))
    if not SAFE_NAME.fullmatch(str(experiment["name"])):
        raise ValueError("Unsafe experiment name")
    if tuple(dataset["tasks"]) != tuple(TASK_METRICS):
        raise ValueError("Rank probe tasks must be [cls, caption, vqa]")
    if tuple(experiment["clients"]) != ("client_0", "client_1"):
        raise ValueError("Rank probe preserves the existing two clients per task")
    ranks = experiment["ranks"]
    if (not isinstance(ranks, list) or not ranks
            or any(isinstance(rank, bool) or not isinstance(rank, int)
                   or rank not in SUPPORTED_RANKS for rank in ranks)
            or ranks != sorted(set(ranks))):
        raise ValueError("Ranks must be an increasing unique subset of [2, 4, 8, 16, 32]")
    seeds = experiment["seeds"]
    if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be a non-empty unique list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
           for seed in seeds):
        raise ValueError("Seeds must be integers in [0, 2**32)")
    positive_int(experiment["lora_alpha_per_rank"], "lora_alpha_per_rank")
    if positive_int(config["runtime"]["world_size"], "world_size") != 8:
        raise ValueError("This experiment requires the original eight-GPU DDP setup")
    timeout = float(config["runtime"]["ddp_timeout_seconds"])
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("DDP timeout must be finite and positive")
    if set(training) != {"local_epochs", "batch_size", "gradient_accumulation", "learning_rate",
                         "max_text_length", "optimizer"}:
        raise ValueError("Only epoch-controlled historical DDP training is supported")
    for field in ("local_epochs", "batch_size", "gradient_accumulation", "max_text_length"):
        positive_int(training[field], field)
    if training["optimizer"] != "adamw":
        raise ValueError("Rank probe reuses the historical AdamW DDP path")
    if not math.isfinite(float(training["learning_rate"])) or training["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    model, lora = config["model"], config["model"]["lora"]
    if model["adapter"] != "llava15" or model.get("local_files_only") is not True:
        raise ValueError("Rank probe uses only local original LLaVA weights")
    if model.get("quantization", "none") != "none" or model.get("dtype") not in {"fp16", "bf16"}:
        raise ValueError("Use the unquantized original FP16/BF16 model")
    if lora.get("enabled") is not True or lora.get("bias") != "none":
        raise ValueError("Only ordinary trainable LoRA factors are supported")
    if lora.get("train_mm_projector") is not False or tuple(lora["target_modules"]) != (
        "q_proj", "k_proj", "v_proj", "o_proj",
    ):
        raise ValueError("Keep the projector frozen and the original Q/K/V/O targets")
    if not 0 <= float(lora["dropout"]) < 1:
        raise ValueError("LoRA dropout must be in [0, 1)")
    if dataset.get("require_images") is not True:
        raise ValueError("Real training requires images")
    evaluation = config["evaluation"]
    if not set(evaluation["image_disjoint_tasks"]) <= set(dataset["tasks"]):
        raise ValueError("Unknown image-disjoint task")
    positive_int(evaluation["generation_max_new_tokens"], "generation_max_new_tokens")
    for section, key in ((evaluation, "supplemental_standard_validation"),
                         (config["output"], "progress_bar")):
        if not isinstance(section[key], bool):
            raise ValueError(f"{key} must be boolean")


def build_jobs(config: Mapping[str, Any], *, task: str | None = None,
               client: str | None = None, lora_rank: int | None = None,
               seed: int | None = None) -> list[ProbeJob]:
    validate_probe_config(config)
    exp = config["experiment"]
    for value, allowed, label in ((task, config["dataset"]["tasks"], "task"),
                                  (client, exp["clients"], "client"),
                                  (lora_rank, exp["ranks"], "rank"),
                                  (seed, exp["seeds"], "seed")):
        if value is not None and value not in allowed:
            raise ValueError(f"Selected {label} is not present in the standalone config: {value}")
    return [ProbeJob(t, c, r, s, exp["lora_alpha_per_rank"] * r)
            for t in config["dataset"]["tasks"] for c in exp["clients"]
            for s in exp["seeds"] for r in exp["ranks"]
            if (task is None or t == task) and (client is None or c == client)
            and (lora_rank is None or r == lora_rank) and (seed is None or s == seed)]


def job_config(config: Mapping[str, Any], job: ProbeJob) -> dict[str, Any]:
    result = copy.deepcopy({key: value for key, value in config.items() if not key.startswith("_")})
    result["model"]["lora"].update(r=job.lora_rank, alpha=job.lora_alpha)
    return result


def steps_per_epoch(samples: int, batch: int, accumulation: int, world: int = 8) -> int:
    """Exactly the historical rank-strided padded sampler and epoch-tail flush."""
    for value, label in ((samples, "samples"), (batch, "batch"),
                         (accumulation, "accumulation"), (world, "world")):
        positive_int(value, label)
    per_rank = (samples + world - 1) // world
    micro_batches = (per_rank + batch - 1) // batch
    return (micro_batches + accumulation - 1) // accumulation


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_ids(ids: Sequence[Any]) -> str:
    return hashlib.sha256(json.dumps(list(ids), ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def append_json(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validation_protocols(task: str, records: list[dict[str, Any]],
                         train_images: set[str], config: Mapping[str, Any]) -> dict[str, Any]:
    def normalize(image: Any) -> str:
        return str(image).replace("\\", "/").removeprefix("./")
    all_indices = list(range(len(records)))
    unseen = [i for i, row in enumerate(records) if normalize(row["image"]) not in train_images]
    disjoint = task in config["evaluation"]["image_disjoint_tasks"]
    primary = "validation_image_disjoint" if disjoint else "validation"
    subsets = {primary: unseen if disjoint else all_indices}
    if disjoint and config["evaluation"]["supplemental_standard_validation"]:
        subsets["validation"] = all_indices
    if any(not indices for indices in subsets.values()):
        raise ValueError(f"Empty validation protocol for {task}; cannot compute a real metric")
    return {"primary_protocol": primary, "subsets": subsets,
            "overlapping_validation_records": len(records) - len(unseen)}


def prepare_data_manifest(config: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    """Annotation-only preparation; do not scan images or touch test annotations."""
    root = (repo / config["dataset"]["partition_root"]).resolve()
    manifest: dict[str, Any] = {"partition_root": str(root), "source_files": {}, "tasks": {}}
    for task in config["dataset"]["tasks"]:
        images: set[str] = set()
        counts = {}
        for client in config["experiment"]["clients"]:
            relative = f"{task}/{client}.json"
            path = root / relative
            records = AFVLMTaskAdapter._records(path)
            if not records:
                raise ValueError(f"Empty client partition: {path}")
            counts[client] = len(records)
            images.update(str(row["image"]).replace("\\", "/").removeprefix("./")
                          for row in records)
            manifest["source_files"][relative] = hash_file(path)
        relative = f"{task}/val.json"
        records = AFVLMTaskAdapter._records(root / relative)
        adapter = AFVLMTaskAdapter(task, {"partition_dir": str(root / task),
            "image_root": str(repo / config["dataset"]["image_root"]), "require_images": False})
        samples = adapter.load_split("validation")
        protocols = validation_protocols(task, records, images, config)
        ids = [sample.id for sample in samples]
        if len(ids) != len(records) or len(set(ids)) != len(ids):
            raise ValueError(f"Probe needs one unique validation ID per source record: {task}")
        manifest["source_files"][relative] = hash_file(root / relative)
        manifest["tasks"][task] = {
            "client_records": counts, "validation_ids": ids, "validation_ids_sha256": hash_ids(ids),
            **protocols,
            "subset_ids_sha256": {name: hash_ids([ids[i] for i in indices])
                                  for name, indices in protocols["subsets"].items()},
        }
    return manifest


def verify_sources(manifest: Mapping[str, Any]) -> None:
    root = Path(manifest["partition_root"])
    for relative, digest in manifest["source_files"].items():
        path = root / relative
        if hash_file(path) != digest:
            raise ValueError(f"Annotation changed after experiment preparation: {path}")


def sampler_manifest(count: int, epochs: int, seed: int, batch: int,
                     accumulation: int, world: int = 8) -> list[dict[str, Any]]:
    """Expected original trainer order, not a new sampler or claimed observed trace."""
    rng = random.Random(seed)
    result = []
    padded = ((count + world - 1) // world) * world
    for epoch in range(1, epochs + 1):
        indices = list(range(count))
        rng.shuffle(indices)
        missing = padded - count
        if missing:
            indices.extend((indices * math.ceil(missing / count))[:missing])
        result.append({"epoch": epoch, "original_records": count, "padded_records": padded,
            "padding_duplicates": missing, "global_indices_sha256": hash_ids(indices),
            "per_rank_indices_sha256": [hash_ids(indices[rank::world]) for rank in range(world)],
            "optimizer_steps": steps_per_epoch(count, batch, accumulation, world)})
    return result


LONG_FIELDS = ["job_id", "task", "client", "lora_rank", "lora_alpha", "seed", "epoch",
               "optimizer_steps", "num_train_samples", "num_eval_samples", "protocol", "primary",
               "metric", "value", "train_annotation_sha256", "validation_ids_sha256", "job_status"]
CURVE_FIELDS = ["task", "seed", "lora_rank", "epoch", "protocol", "metric", "mean",
                "client_count", "expected_client_count", "complete"]
STEP_FIELDS = ["job_id", "task", "client", "lora_rank", "lora_alpha", "seed", "epoch",
               "optimizer_step", "epoch_step", "mean_loss_across_ddp_ranks",
               "elapsed_seconds_including_evaluation", "job_status"]
EPOCH_FIELDS = ["job_id", "task", "client", "lora_rank", "lora_alpha", "seed", "epoch",
                "optimizer_steps", "num_train_samples", "train_annotation_sha256",
                "epoch_train_loss", "epoch_train_seconds",
                "epoch_train_peak_allocated_bytes_max_rank", "expected_sample_order_sha256",
                "job_status"]


def export_probe_results(run_root: Path) -> dict[str, int]:
    """Export raw metrics, equal-client curves and paired gains, including failure labels."""
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    steps, epochs = [], []
    grouped: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    scores: dict[tuple[Any, ...], dict[int, float]] = defaultdict(dict)
    expected = defaultdict(set)
    expected_ranks = sorted({job["lora_rank"] for job in manifest["jobs"]})
    for job in manifest["jobs"]:
        expected[job["task"]].add(job["client"])
        for filename, target, fields, identity_field in (
            ("training_steps.jsonl", steps, STEP_FIELDS, "optimizer_step"),
            ("epoch_training.jsonl", epochs, EPOCH_FIELDS, "epoch"),
        ):
            source = run_root / job["job_id"] / filename
            identities = set()
            for line in source.read_text(encoding="utf-8").splitlines() if source.is_file() else []:
                record = json.loads(line)
                if record["job_id"] != job["job_id"] or record[identity_field] in identities:
                    raise ValueError(f"Different job or duplicate training record: {source}")
                identities.add(record[identity_field])
                target.append({**{key: record[key] for key in fields if key != "job_status"},
                               "job_status": job["status"]})
        path = run_root / job["job_id"] / "metrics.jsonl"
        seen = set()
        for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
            record = json.loads(line)
            if record["job_id"] != job["job_id"]:
                raise ValueError(f"Metric belongs to a different job: {path}")
            for metric, value in record["metrics"].items():
                identity = (record["epoch"], record["protocol"], metric)
                if identity in seen or not math.isfinite(float(value)):
                    raise ValueError(f"Duplicate/non-finite metric: {path}, {identity}")
                seen.add(identity)
                row = {key: record[key] for key in LONG_FIELDS if key not in {
                    "metric", "value", "job_status"}}
                row.update(metric=metric, value=float(value), job_status=job["status"])
                rows.append(row)
                if record["primary"] and job["status"] == "completed":
                    key = (job["task"], job["seed"], job["lora_rank"], record["epoch"],
                           record["protocol"], metric)
                    grouped[key][job["client"]] = float(value)
                    if metric == TASK_METRICS[job["task"]]:
                        scores[job["task"], job["client"], job["seed"],
                               record["epoch"], metric][job["lora_rank"]] = float(value)
    curves = [{"task": key[0], "seed": key[1], "lora_rank": key[2], "epoch": key[3],
               "protocol": key[4], "metric": key[5], "mean": statistics.fmean(values.values()),
               "client_count": len(values), "expected_client_count": len(expected[key[0]]),
               "complete": set(values) == expected[key[0]]}
              for key, values in sorted(grouped.items())]
    gains = []
    pairs = list(zip(expected_ranks[:-1], expected_ranks[1:], strict=True))
    gain_fields = (["task", "client", "seed", "epoch", "metric"]
                   + [f"score_rank{rank}" for rank in expected_ranks]
                   + [f"gain_{low}_to_{high}" for low, high in pairs]
                   + [f"gain_per_added_rank_{low}_to_{high}" for low, high in pairs]
                   + ["gain_ratio_48_over_24"])
    for key, ranks in sorted(scores.items()):
        if set(ranks) != set(expected_ranks) or not pairs:
            continue
        row = {"task": key[0], "client": key[1], "seed": key[2], "epoch": key[3],
               "metric": key[4], "gain_ratio_48_over_24": ""}
        row.update({f"score_rank{rank}": value for rank, value in ranks.items()})
        for low, high in pairs:
            gain = ranks[high] - ranks[low]
            row[f"gain_{low}_to_{high}"] = gain
            row[f"gain_per_added_rank_{low}_to_{high}"] = gain / (high - low)
        if {2, 4, 8} <= ranks.keys() and ranks[4] - ranks[2] > 1e-12:
            row["gain_ratio_48_over_24"] = (ranks[8] - ranks[4]) / (ranks[4] - ranks[2])
        gains.append(row)
    write_csv(run_root / "metrics_long.csv", rows, LONG_FIELDS)
    write_csv(run_root / "task_rank_curves.csv", curves, CURVE_FIELDS)
    write_csv(run_root / "rank_gains.csv", gains, gain_fields)
    write_csv(run_root / "training_steps.csv", steps, STEP_FIELDS)
    write_csv(run_root / "epoch_training.csv", epochs, EPOCH_FIELDS)
    final_epoch = int(manifest["local_epochs"])
    write_csv(run_root / "final_epoch_metrics.csv", [row for row in rows
              if row["epoch"] == final_epoch and row["primary"]], LONG_FIELDS)
    return {"metric_rows": len(rows), "curve_rows": len(curves), "gain_rows": len(gains),
            "training_step_rows": len(steps), "training_epoch_rows": len(epochs)}
