"""Opt-in runtime for one fresh eight-GPU local rank probe job.

All imports that can initialize CUDA or LLaVA are inside run_worker. Existing
models, optimizers, DDP bodies, evaluation metrics and federated methods are not patched.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import os
import random
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from afl_vlm.config import load_config
from afl_vlm.data.afvlm_cm import AFVLMTaskAdapter
from afl_vlm.evaluation import merge_evaluation_outputs
from afl_vlm.experiments.rank_probe import (
    ProbeJob,
    append_json,
    hash_file,
    hash_ids,
    job_config,
    sampler_manifest,
    steps_per_epoch,
    validate_probe_config,
    verify_sources,
    write_json,
)
from afl_vlm.models.base import TrainConfig


def primary_call(torch: Any, action: Callable[[], Any]) -> Any:
    """Broadcast rank0 IO/metric errors rather than stranding other ranks in a barrier."""
    payload: list[dict[str, Any]] = [{"result": None, "error": None}]
    if torch.distributed.get_rank() == 0:
        try:
            payload[0]["result"] = action()
        except Exception as exc:
            payload[0]["error"] = f"{type(exc).__name__}: {exc}"
    torch.distributed.broadcast_object_list(payload, src=0)
    if payload[0]["error"]:
        raise RuntimeError("Rank probe primary process failed: " + payload[0]["error"])
    return payload[0]["result"]


@contextmanager
def preserved_training_state(model: Any, torch: Any) -> Iterator[None]:
    """Evaluation observes parameters only; restore modes and all locally used RNGs."""
    import numpy as np

    modes = [(module, module.training) for module in model.model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.get_rng_state()
    device = torch.cuda.current_device()
    cuda_rng = torch.cuda.get_rng_state(device)
    network = model._network()
    cache = network.config.use_cache
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training
        network.config.use_cache = cache
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)


def _max_on_ranks(torch: Any, value: float) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=f"cuda:{torch.cuda.current_device()}")
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return float(tensor.item())


def evaluate_epoch(torch: Any, model: Any, adapter: AFVLMTaskAdapter,
                   task_data: Mapping[str, Any], job: ProbeJob, epoch: int,
                   output: Path, common: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Generate one full validation corpus across8 ranks; score subsets only after gathering."""
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    ids = task_data["validation_ids"]
    local_items = list(enumerate(ids))[rank::world]
    started = time.perf_counter()
    with preserved_training_state(model, torch):
        try:
            predictions, references = model.generate_evaluation_outputs(
                adapter, [sample_id for _, sample_id in local_items],
                progress_label=f"evaluate {job.task}/{job.client} r{job.lora_rank} "
                               f"epoch{epoch} rank{rank + 1}/{world}",
            )
            if len(predictions) != len(local_items) or len(references) != len(local_items):
                raise ValueError("Incomplete evaluation shard")
            local = {"rows": [(index, pred, ref) for (index, _), pred, ref in
                              zip(local_items, predictions, references, strict=True)],
                     "error": None}
        except Exception as exc:
            local = {"rows": [], "error": f"rank{rank}: {type(exc).__name__}: {exc}"}
        gathered = [None] * world if rank == 0 else None
        torch.distributed.gather_object(local, gathered, dst=0)
        eval_seconds = _max_on_ranks(torch, time.perf_counter() - started)
        eval_peak = int(_max_on_ranks(torch, torch.cuda.max_memory_allocated()))

        def score_and_save() -> list[dict[str, Any]]:
            if gathered is None or any(item is None for item in gathered):
                raise RuntimeError("Missing evaluation rank payload")
            errors = [item["error"] for item in gathered if item["error"]]
            if errors:
                raise RuntimeError("; ".join(errors))
            predictions, references = merge_evaluation_outputs(
                [item["rows"] for item in gathered], len(ids)
            )
            # Keep predictions in memory only until all corpus metrics are computed.
            records = []
            for protocol, indices in task_data["subsets"].items():
                metrics = adapter.metric([predictions[i] for i in indices],
                                         [references[i] for i in indices])
                if not metrics or any(not math.isfinite(float(value))
                                      for value in metrics.values()):
                    raise ValueError("Non-finite or empty validation metrics")
                record = {**common, "epoch": epoch, "protocol": protocol,
                    "primary": protocol == task_data["primary_protocol"],
                    "num_eval_samples": len(indices),
                    "validation_ids_sha256": task_data["subset_ids_sha256"][protocol],
                    "metrics": metrics, "evaluation_seconds": eval_seconds,
                    "evaluation_peak_allocated_bytes_max_rank": eval_peak,
                    "prediction_storage": "memory_only",
                    "decoding": "greedy_num_beams1"}
                append_json(output / "metrics.jsonl", record)
                records.append(record)
            print(f"[rank-probe] evaluated {job.key} epoch={epoch}: "
                  f"{[(row['protocol'], row['metrics']) for row in records]}", flush=True)
            return records

        return primary_call(torch, score_and_save)


def _model_provenance(model: Any, job: ProbeJob, torch: Any) -> dict[str, Any]:
    import hashlib

    parameters = model.named_federated_parameters()
    if not parameters or any("vision_tower" in name or "mm_projector" in name
                             or not any(part in name for part in (".lora_A.", ".lora_B."))
                             for name, _ in parameters):
        raise ValueError("Probe must train only language-side LoRA A/B, with no private adapters")
    b_nonzero = [torch.count_nonzero(value).to(dtype=torch.float32)
                 for name, value in parameters if ".lora_B." in name]
    if not b_nonzero or float(torch.stack(b_nonzero).sum().item()) != 0:
        raise ValueError("Fresh standard LoRA must start from zero B, not a previous adapter")
    digest = hashlib.sha256()
    dtypes: dict[str, int] = {}
    for name, value in parameters:
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
        dtypes[str(value.dtype)] = dtypes.get(str(value.dtype), 0) + value.numel()
    versions = {}
    for package in ("torch", "transformers", "peft", "numpy", "pycocoevalcap"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"trainable_parameters": sum(value.numel() for _, value in parameters),
            "trainable_tensor_count": len(parameters), "trainable_dtypes": dtypes,
            "initial_adapter_sha256": digest.hexdigest(), "initial_B_is_zero": True,
            "lora_scaling": job.lora_alpha / job.lora_rank, "versions": versions}


def _execute_job(torch: Any, config: dict[str, Any], data: Mapping[str, Any],
                 job: ProbeJob, output: Path, repo: Path) -> dict[str, Any]:
    import numpy as np

    from afl_vlm.models.llava15 import Llava15Adapter

    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ["AFVLM_PRIMARY_PROCESS"] = "1" if rank == 0 else "0"
    random.seed(job.seed)
    np.random.seed(job.seed)
    torch.manual_seed(job.seed)
    torch.cuda.manual_seed_all(job.seed)
    training, dataset = config["training"], config["dataset"]
    partition = Path(data["partition_root"]) / job.task
    adapter = AFVLMTaskAdapter(job.task, {"partition_dir": str(partition),
        "image_root": str((repo / dataset["image_root"]).resolve()), "require_images": True})
    samples = adapter.load_file(partition / f"{job.client}.json", "train")
    task_data = data["tasks"][job.task]
    if len(samples) != task_data["client_records"][job.client]:
        raise ValueError("Client record count changed")
    validation = adapter.load_split("validation")
    if [sample.id for sample in validation] != task_data["validation_ids"]:
        raise ValueError("Validation IDs changed")
    epoch_steps = steps_per_epoch(len(samples), training["batch_size"],
                                   training["gradient_accumulation"], world)
    total_steps = epoch_steps * training["local_epochs"]
    sampler = sampler_manifest(len(samples), training["local_epochs"], job.seed,
                               training["batch_size"], training["gradient_accumulation"], world)

    def initialize_output() -> None:
        if (output / "job_config.yaml").exists() or (output / "metrics.jsonl").exists():
            raise FileExistsError(f"Job output already exists: {output}")
        output.mkdir(parents=True, exist_ok=True)
        (output / "job_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        write_json(output / "expected_sampler.json", sampler)
        for filename in ("metrics.jsonl", "training_steps.jsonl", "epoch_training.jsonl"):
            (output / filename).touch(exist_ok=False)

    primary_call(torch, initialize_output)
    if rank == 0:
        effective_batch = world * training['batch_size'] * training['gradient_accumulation']
        print(f"[rank-probe] loading fresh LLaVA: {job.key}; samples={len(samples)}, "
              f"epochs={training['local_epochs']}, steps={total_steps}, DDP={world}, "
              f"effective_batch={effective_batch}",
              flush=True)
    model_config = copy.deepcopy(config["model"])
    model_config.update(device_map={"": local_rank},
        model_path=str((repo / model_config["model_path"]).resolve()),
        vision_tower_path=str((repo / model_config["vision_tower_path"]).resolve()),
        max_text_length=training["max_text_length"],
        max_new_tokens=config["evaluation"]["generation_max_new_tokens"],
        progress_bar=bool(config["output"]["progress_bar"]) and rank == 0)
    model = Llava15Adapter()
    model.load(model_config)
    model.enable_distributed_data_parallel(local_rank)
    network = model._network()
    network.generation_config = copy.deepcopy(network.generation_config)
    network.generation_config.do_sample = False
    network.generation_config.num_beams = 1
    provenance = primary_call(torch, lambda: _model_provenance(model, job, torch))
    metadata = {**job.as_dict(), **provenance, "world_size": world,
        "training_path": "Llava15Adapter._local_train_historical_ddp",
        "num_train_samples": len(samples), "local_epochs": training["local_epochs"],
        "optimizer_steps_per_epoch": epoch_steps, "planned_optimizer_steps": total_steps,
        "effective_batch_size": world * training["batch_size"] * training["gradient_accumulation"],
        "optimizer": "AdamW", "optimizer_defaults": {"betas": [0.9, 0.999],
                                                       "eps": 1e-8, "weight_decay": 0.01},
        "train_annotation_sha256": data["source_files"][f"{job.task}/{job.client}.json"],
        "training_sample_ids_sha256": hash_ids([sample.id for sample in samples]),
        "primary_protocol": task_data["primary_protocol"], "federated_aggregation": False,
        "async_plan": None, "saves_predictions": False, "saves_adapter_parameters": False}
    primary_call(torch, lambda: write_json(output / "metadata.json", metadata))
    started = epoch_started = time.perf_counter()
    completed_epochs: list[int] = []
    all_metrics: list[dict[str, Any]] = []
    epoch_losses: list[float] = []
    torch.cuda.reset_peak_memory_stats()

    def progress(step: int, total: int, loss: float) -> None:
        if total != total_steps or not 1 <= step <= total:
            raise ValueError("Unexpected historical trainer step count")
        value = torch.tensor(loss, dtype=torch.float64, device=f"cuda:{local_rank}")
        torch.distributed.all_reduce(value)
        mean_loss = float(value.item()) / world
        if not math.isfinite(mean_loss):
            raise RuntimeError(
                "Non-finite local training loss; refusing to publish a completed run"
            )
        epoch_losses.append(mean_loss)
        primary_call(torch, lambda: append_json(output / "training_steps.jsonl", {
            **job.as_dict(), "epoch": (step - 1) // epoch_steps + 1, "optimizer_step": step,
            "epoch_step": (step - 1) % epoch_steps + 1, "mean_loss_across_ddp_ranks": mean_loss,
            "elapsed_seconds_including_evaluation": time.perf_counter() - started,
        }))

    def epoch_hook(adapter_model: Any, step: int, total: int,
                   context: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal epoch_started
        del total, context
        if step % epoch_steps:
            return {}
        epoch = step // epoch_steps
        if epoch != len(completed_epochs) + 1 or len(epoch_losses) != epoch_steps:
            raise RuntimeError("Epoch callbacks do not match the original sampler windows")
        train_seconds = _max_on_ranks(torch, time.perf_counter() - epoch_started)
        train_peak = int(_max_on_ranks(torch, torch.cuda.max_memory_allocated()))
        common = {**job.as_dict(), "optimizer_steps": step, "num_train_samples": len(samples),
                  "train_annotation_sha256": metadata["train_annotation_sha256"],
                  "epoch_train_loss": sum(epoch_losses) / len(epoch_losses),
                  "epoch_train_seconds": train_seconds,
                  "epoch_train_peak_allocated_bytes_max_rank": train_peak}
        torch.cuda.reset_peak_memory_stats()
        metrics = evaluate_epoch(
            torch, adapter_model, adapter, task_data, job, epoch, output, common
        )
        all_metrics.extend(metrics)

        def save_epoch() -> None:
            append_json(output / "epoch_training.jsonl", {**common, "epoch": epoch,
                "expected_sample_order_sha256": sampler[epoch - 1]["global_indices_sha256"]})

        primary_call(torch, save_epoch)
        completed_epochs.append(epoch)
        epoch_losses.clear()
        torch.cuda.reset_peak_memory_stats()
        epoch_started = time.perf_counter()
        return {}

    train_config = TrainConfig(
        local_epochs=training["local_epochs"], batch_size=training["batch_size"],
        gradient_accumulation=training["gradient_accumulation"],
        learning_rate=training["learning_rate"], max_text_length=training["max_text_length"],
        seed=job.seed, optimizer="adamw", planned_optimizer_steps=total_steps,
        distributed_rank=rank, distributed_world_size=world,
        context={"client": SimpleNamespace(client_id=f"{job.task}/{job.client}", local_round=0)},
        progress_hook=progress, step_hook=epoch_hook,
    )
    # One invocation retains the same optimizer throughout all FIVE continuous epochs.
    # No loss/gradient hook, method adapter, fresh-global mixing or server is involved.
    trained = model.local_train(samples, train_config, job.task)
    if trained.optimizer_steps != total_steps or completed_epochs != list(
        range(1, training["local_epochs"] + 1)
    ):
        raise RuntimeError("Incomplete five-epoch local experiment")
    result = {**metadata, "status": "completed", "optimizer_steps": trained.optimizer_steps,
              "completed_epochs": completed_epochs, "validation_records": len(all_metrics),
              "elapsed_seconds_including_evaluation": time.perf_counter() - started,
              "final_primary_metrics": next(row["metrics"] for row in reversed(all_metrics)
                                            if row["primary"])}
    primary_call(torch, lambda: write_json(output / "result.json", result))
    return result


def run_worker(run_root: Path, job_id: str, repo: Path) -> None:
    import torch

    world = int(os.environ.get("WORLD_SIZE", "0"))
    if world != 8 or not torch.cuda.is_available():
        raise RuntimeError("Rank probe workers require torchrun with exactly8 CUDA ranks")
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "local_rank_probe" or manifest["status"] != "running":
        raise ValueError("Not an active isolated rank-probe manifest")
    candidates = [row for row in manifest["jobs"] if row["job_id"] == job_id]
    if len(candidates) != 1 or candidates[0]["status"] != "running":
        raise ValueError("Worker job is absent, completed or not authorized by its manifest")
    row = candidates[0]
    job = ProbeJob(**{field: row[field] for field in (
        "task", "client", "lora_rank", "seed", "lora_alpha",
    )})
    if job.key != job_id:
        raise ValueError("Unsafe worker job path")
    frozen = run_root / "resolved_config.yaml"
    data_path = run_root / "data_manifest.json"
    if (hash_file(frozen) != manifest["config_sha256"]
            or hash_file(data_path) != manifest["data_sha256"]):
        raise ValueError("Frozen experiment inputs changed")
    config = load_config(frozen)
    validate_probe_config(config)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    verify_sources(data)
    for path, digest in manifest["source_sha256"].items():
        if hash_file(repo / path) != digest:
            raise ValueError(f"Training/evaluation source changed during the experiment: {path}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    try:
        torch.distributed.init_process_group(backend="nccl", init_method="env://",
            timeout=timedelta(seconds=float(config["runtime"]["ddp_timeout_seconds"])))
        _execute_job(torch, job_config(config, job), data, job, run_root / job.key, repo)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
