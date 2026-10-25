"""Launch isolated, fresh eight-GPU DDP local rank experiments, never federated runs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "code"))

from afl_vlm.config import load_config  # noqa: E402
from afl_vlm.data.afvlm_cm import validate_caption_metric_runtime  # noqa: E402
from afl_vlm.experiments.rank_probe import (  # noqa: E402
    SAFE_NAME,
    SUPPORTED_RANKS,
    build_jobs,
    export_probe_results,
    hash_file,
    prepare_data_manifest,
    validate_probe_config,
    write_json,
)
from afl_vlm.experiments.rank_probe_history import (  # noqa: E402
    read_completed_run,
    validate_extension,
)


def _relay_output(source: BinaryIO, terminal: BinaryIO, log: BinaryIO) -> None:
    """Preserve tqdm CR/cursor sequences; text-mode iteration turns CR into newlines."""
    while chunk := source.read1(8192):
        terminal.write(chunk)
        terminal.flush()
        log.write(chunk)
        log.flush()


def _stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def launch(args: argparse.Namespace) -> Path:
    import torch

    previous_manifest = None
    if args.extend_from:
        previous_manifest, config, _ = read_completed_run(args.extend_from.resolve())
        if args.ranks is None and args.lora_rank is None:
            raise ValueError("Use --ranks 16 32 (or --rank) to specify only the additional ranks")
    else:
        config = load_config(args.config)
    if args.ranks is not None:
        config["experiment"]["ranks"] = args.ranks
    elif args.extend_from and args.lora_rank is not None:
        config["experiment"]["ranks"] = [args.lora_rank]
    validate_probe_config(config)
    jobs = build_jobs(config, task=args.task, client=args.client,
                      lora_rank=args.lora_rank, seed=args.seed)
    if previous_manifest is not None:
        previous_groups = {(row["task"], row["client"], row["seed"])
                           for row in previous_manifest["jobs"]}
        jobs = [job for job in jobs if (job.task, job.client, job.seed) in previous_groups]
        if not jobs:
            raise ValueError("No selected clients/seeds match the previous experiment")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("This probe requires exactly8 visible CUDA GPUs; no single-GPU fallback")
    for key in ("model_path", "vision_tower_path"):
        path = (REPOSITORY_ROOT / config["model"][key]).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Provision local weights first; no automatic download: {path}")
    if any(job.task == "caption" for job in jobs):
        validate_caption_metric_runtime()
    name = args.run_name or datetime.now(timezone.utc).strftime("rank_probe_%Y%m%dT%H%M%S_%fZ")
    if not SAFE_NAME.fullmatch(name):
        raise ValueError("Unsafe run name; use letters, digits, underscores, hyphens or dots")
    data = prepare_data_manifest(config, REPOSITORY_ROOT)
    extension = validate_extension(args.extend_from.resolve(), config, data, jobs,
                                   REPOSITORY_ROOT) if args.extend_from else None
    root = (REPOSITORY_ROOT / config["output"]["directory"] / name).resolve()
    root.mkdir(parents=True, exist_ok=False)  # Never delete/overwrite a previous experiment.
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    frozen = root / "resolved_config.yaml"
    frozen.write_text(yaml.safe_dump(public, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_json(root / "data_manifest.json", data)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
                              capture_output=True, text=True, check=False)
    manifest = {"schema_version": 1, "kind": "local_rank_probe", "status": "running",
        "run_name": name, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "config_sha256": hash_file(frozen), "data_sha256": hash_file(root / "data_manifest.json"),
        "local_epochs": config["training"]["local_epochs"], "world_size": 8,
        "independent_fresh_process_group_per_job": True, "uses_federation": False,
        "extends_run": extension,
        "primary_protocol_by_task": {task: row["primary_protocol"]
                                     for task, row in data["tasks"].items()},
        "source_sha256": {path: hash_file(REPOSITORY_ROOT / path) for path in (
            "code/afl_vlm/models/llava15.py", "code/afl_vlm/data/afvlm_cm.py",
            "code/afl_vlm/evaluation.py", "code/afl_vlm/experiments/rank_probe.py",
            "code/afl_vlm/experiments/rank_probe_runtime.py", "scripts/run_local_rank_probe.py",
        )},
        "jobs": [{**job.as_dict(), "status": "pending"} for job in jobs]}
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    print(f"[rank-probe] output={root}; independent jobs={len(jobs)}; "
          f"epochs={manifest['local_epochs']}",
          flush=True)
    current = None
    try:
        for number, current in enumerate(manifest["jobs"], start=1):
            directory = root / current["job_id"]
            directory.mkdir(parents=True, exist_ok=False)
            current["status"] = "running"
            write_json(manifest_path, manifest)
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nnodes=1", "--nproc_per_node=8", "--max_restarts=0",
                str(Path(__file__).resolve()),
                "--prepared-root", str(root), "--worker-job", current["job_id"]]
            print(f"[rank-probe] job {number}/{len(jobs)}: {current['job_id']}", flush=True)
            environment = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
                           "TRANSFORMERS_OFFLINE": "1"}
            with (directory / "launcher.log").open("wb") as log:
                process = subprocess.Popen(command, cwd=REPOSITORY_ROOT, env=environment,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    start_new_session=os.name != "nt")
                try:
                    if process.stdout is None:
                        raise RuntimeError("Cannot capture worker logs")
                    _relay_output(process.stdout, sys.stdout.buffer, log)
                    code = process.wait()
                except BaseException:
                    _stop_child(process)
                    raise
            if code:
                raise RuntimeError(
                    f"Job failed (exit={code}): {current['job_id']}; see {directory}"
                )
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            if result["status"] != "completed" or result["job_id"] != current["job_id"]:
                raise RuntimeError("Worker did not report a completed independent job")
            current["status"] = "completed"
            write_json(manifest_path, manifest)
            export_probe_results(root)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        if current is not None and current["status"] != "completed":
            current["status"] = "failed"
        raise
    finally:
        write_json(manifest_path, manifest)
        export_probe_results(root)
    print(f"[rank-probe] completed; metrics and paired gains saved under {root}", flush=True)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--config", type=Path,
                        default=REPOSITORY_ROOT / "configs/probes/local_rank_probe.yaml")
    inputs.add_argument("--extend-from", type=Path,
                        help="Read a completed run's frozen config; keep its results untouched")
    parser.add_argument("--run-name", help="New output name; existing runs are never overwritten")
    parser.add_argument("--task", choices=("cls", "caption", "vqa"))
    parser.add_argument("--client", choices=("client_0", "client_1"))
    ranks = parser.add_mutually_exclusive_group()
    ranks.add_argument("--rank", dest="lora_rank", type=int, choices=SUPPORTED_RANKS)
    ranks.add_argument("--ranks", nargs="+", type=int, choices=SUPPORTED_RANKS,
                       help="Only train these increasing ranks, e.g. --ranks 16 32")
    parser.add_argument("--seed", type=int, help="Select one seed already listed in the config")
    parser.add_argument("--prepared-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-job", help=argparse.SUPPRESS)
    parser.add_argument("--local-rank", "--local_rank", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(REPOSITORY_ROOT)
    if args.worker_job or args.prepared_root:
        if not args.worker_job or not args.prepared_root:
            parser.error("Internal worker arguments must be supplied together")
        from afl_vlm.experiments.rank_probe_runtime import run_worker

        run_worker(args.prepared_root.resolve(), args.worker_job, REPOSITORY_ROOT)
    else:
        launch(args)


if __name__ == "__main__":
    main()
