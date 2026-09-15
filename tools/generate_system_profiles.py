"""Create immutable AFVLM-CM config, system-profile, and TrainPlan snapshots."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from afl_vlm.config import load_config, validate_config  # noqa: E402
from afl_vlm.federation.types import ClientSpec  # noqa: E402
from afl_vlm.methods.registry import method_names  # noqa: E402
from afl_vlm.scheduling.train_plan import (  # noqa: E402
    build_train_plan,
    event_records,
    load_train_plan,
    optimizer_steps_for_epochs,
)

SETTINGS = (2, 5, 10)
PROFILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

TASK_DATASETS = {
    "cls": "ImageNet-R",
    "caption": "Flickr30k",
    "vqa": "AOKVQA",
    "chart_vqa": "DVQA",
    "visual_reasoning": "FigureQA",
    "grounding": "Grounding",
}


def stable_seed(seed: int, value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode()).digest()[:8], "big")


def _write_text_lf(path: Path, content: str) -> None:
    """Write deterministic text bytes on Windows and Linux."""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def build(root: Path, setting: int, seed: int) -> dict[str, Any]:
    partition = root / "data" / "AFVLM_CM" / "partitioned" / f"{setting}_clients"
    clients = []
    for task, dataset in TASK_DATASETS.items():
        files = sorted(
            (partition / task).glob("client_*.json"), key=lambda p: int(p.stem.split("_")[-1])
        )
        for path in files:
            rng = random.Random(stable_seed(seed, f"{setting}:{task}:{path.stem}"))
            clients.append(
                {
                    "client_id": f"{task}/{path.stem}",
                    "task": task,
                    "dataset": dataset,
                    "num_samples": len(json.loads(path.read_text(encoding="utf-8"))),
                    "speed_factor": round(rng.uniform(0.5, 2.0), 6),
                    "network_delay": round(rng.uniform(0.05, 0.5), 6),
                    "initial_availability": round(rng.uniform(0.0, 1.0), 6),
                    "base_training_cost": round(rng.uniform(0.8, 1.2), 6),
                }
            )
    expected = 6 * setting
    if len(clients) != expected:
        raise FileNotFoundError(
            f"Expected {expected} clients under {partition}, found {len(clients)}"
        )
    return {
        "schema_version": 1,
        "setting": setting,
        "seed": seed,
        "generation_scope": "method_independent",
        "clients": clients,
    }


def _client_specs(payload: dict[str, Any]) -> list[ClientSpec]:
    return [
        ClientSpec(
            id=str(item["client_id"]),
            task=str(item["task"]),
            dataset=str(item["dataset"]),
            num_samples=int(item["num_samples"]),
            speed_factor=float(item["speed_factor"]),
            network_delay=float(item["network_delay"]),
            initial_availability=float(item["initial_availability"]),
            base_training_cost=float(item["base_training_cost"]),
        )
        for item in payload["clients"]
    ]


def _validate_profile_name(value: str) -> str:
    if not PROFILE_NAME.fullmatch(value) or value in {".", "..", "current"}:
        raise ValueError(
            "Profile name must contain only letters, numbers, '.', '_' or '-', "
            "start with a letter/number, and must not be 'current'"
        )
    return value


def _validate_plan(
    clients: list[ClientSpec],
    plan: list[Any],
    training: dict[str, Any],
    rounds: int,
) -> None:
    expected_steps = {
        item.id: optimizer_steps_for_epochs(
            item.num_samples,
            int(training["local_epochs"]),
            int(training["batch_size"]),
            int(training["gradient_accumulation"]),
        )
        for item in clients
    }
    if len(plan) != len(clients) * rounds:
        raise ValueError(f"TrainPlan has {len(plan)} events; expected {len(clients) * rounds}")
    jobs = {(item.client_id, item.local_round) for item in plan}
    if len(jobs) != len(plan):
        raise ValueError("TrainPlan contains duplicate client/round jobs")
    if any(item.client_id not in expected_steps for item in plan):
        raise ValueError("TrainPlan contains a client absent from its system profile")
    if any(item.local_round < 0 or item.local_round >= rounds for item in plan):
        raise ValueError("TrainPlan contains a local round outside federation.rounds")
    if any(item.local_steps != expected_steps[item.client_id] for item in plan):
        raise ValueError(
            "TrainPlan is incompatible with local_epochs, batch_size, or gradient_accumulation"
        )


def _source_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _list_profiles(root: Path) -> None:
    profiles = root / "experiment_profiles"
    if not profiles.is_dir():
        print("No experiment profiles found.")
        return
    for directory in sorted(path for path in profiles.iterdir() if path.is_dir()):
        manifest = directory / "manifest.yaml"
        if manifest.is_file():
            payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            print(
                f"{directory.name}\tcreated={payload.get('created_utc')}\t"
                f"plan_source={payload.get('plan_source')}"
            )
        else:
            print(f"{directory.name}\t(incomplete: manifest missing)")


def create_profile(
    root: Path,
    profile_name: str,
    seed: int,
    reuse_plans_from: str | None,
) -> Path:
    """Create one immutable, self-contained experiment configuration profile."""
    name = _validate_profile_name(profile_name)
    target = root / "experiment_profiles" / name
    if target.exists():
        raise FileExistsError(
            f"Experiment profile already exists and will not be overwritten: {target}"
        )
    base_config = yaml.safe_load((root / "configs" / "base.yaml").read_text(encoding="utf-8"))
    training = dict(base_config["training"])
    rounds = int(base_config["federation"]["rounds"])
    plans: dict[int, tuple[bytes, bytes]] = {}
    for setting in SETTINGS:
        if reuse_plans_from:
            if reuse_plans_from == "current":
                source = root / "plans" / "afvlm_cm" / f"{setting}clients"
            else:
                _validate_profile_name(reuse_plans_from)
                source = (
                    root / "experiment_profiles" / reuse_plans_from / "plans" / f"{setting}clients"
                )
            system_path = source / "system_profile.json"
            plan_path = source / "async_train_plan.json"
            if not system_path.is_file() or not plan_path.is_file():
                raise FileNotFoundError(f"Reusable plan set is incomplete: {source}")
            system_bytes = system_path.read_bytes()
            plan_bytes = plan_path.read_bytes()
            clients = _client_specs(json.loads(system_bytes))
            plan = load_train_plan(plan_path)
        else:
            system_payload = build(root, setting, seed)
            clients = _client_specs(system_payload)
            plan = build_train_plan(
                clients,
                rounds=rounds,
                local_epochs=int(training["local_epochs"]),
                batch_size=int(training["batch_size"]),
                gradient_accumulation=int(training["gradient_accumulation"]),
                mode="asynchronous",
            )
            system_bytes = (json.dumps(system_payload, indent=2) + "\n").encode()
            plan_bytes = (json.dumps(event_records(plan), indent=2) + "\n").encode()
        _validate_plan(clients, plan, training, rounds)
        plans[setting] = (system_bytes, plan_bytes)

    methods = method_names()
    resolved_configs: dict[tuple[int, str], str] = {}
    for setting in SETTINGS:
        system_path = f"experiment_profiles/{name}/plans/{setting}clients/system_profile.json"
        plan_path = f"experiment_profiles/{name}/plans/{setting}clients/async_train_plan.json"
        for method in methods:
            source = (
                root
                / "configs"
                / "experiments"
                / "llava"
                / "afvlm_cm"
                / f"{setting}clients"
                / f"{method}.yaml"
            )
            config = copy.deepcopy(load_config(source))
            config.pop("_config_path", None)
            config["run"]["seed"] = seed
            config["federation"]["system_profile"] = system_path
            config["federation"]["train_plan"] = plan_path
            config["output"]["directory"] = (
                f"runs/llava/afvlm_cm/profiles/{name}/{setting}clients/{method}/seed{seed}"
            )
            config["experiment_profile"] = {
                "name": name,
                "immutable_snapshot": True,
            }
            validate_config(config, check_paths=False)
            resolved_configs[(setting, method)] = yaml.safe_dump(
                config, sort_keys=False, allow_unicode=True
            )

    for setting, (system_bytes, plan_bytes) in plans.items():
        plan_target = target / "plans" / f"{setting}clients"
        plan_target.mkdir(parents=True, exist_ok=True)
        (plan_target / "system_profile.json").write_bytes(system_bytes)
        (plan_target / "async_train_plan.json").write_bytes(plan_bytes)
    for (setting, method), body in resolved_configs.items():
        config_target = target / "configs" / f"{setting}clients"
        config_target.mkdir(parents=True, exist_ok=True)
        _write_text_lf(config_target / f"{method}.yaml", body)

    hashes = {}
    for path in sorted(item for item in target.rglob("*") if item.is_file()):
        hashes[path.relative_to(target).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "name": name,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_commit": _source_commit(root),
        "plan_source": reuse_plans_from or "generated_from_local_data",
        "run_seed": seed,
        "training": training,
        "federation": {"rounds": rounds},
        "settings": list(SETTINGS),
        "methods": list(methods),
        "files_sha256": hashes,
    }
    _write_text_lf(
        target / "manifest.yaml",
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create immutable AFVLM-CM config + TrainPlan profiles."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--profile", help="New immutable profile name")
    parser.add_argument("--seed", type=int, help="Run/system seed; defaults to configs/base.yaml")
    parser.add_argument(
        "--reuse_plans_from",
        help="Copy compatible plans from another profile, or use 'current' for plans/afvlm_cm",
    )
    parser.add_argument("--list", action="store_true", help="List existing profiles and exit")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.list:
        _list_profiles(root)
        return
    if not args.profile:
        parser.error("--profile is required when creating a profile")
    base_config = yaml.safe_load((root / "configs" / "base.yaml").read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else base_config["run"]["seed"])
    try:
        target = create_profile(root, args.profile, seed, args.reuse_plans_from)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Created immutable experiment profile: {target}")


if __name__ == "__main__":
    main()
