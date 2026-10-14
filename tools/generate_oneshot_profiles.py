"""Standalone immutable OPCM one-shot configs and single-upload Plans; no training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from afl_vlm.batching import detect_reference_world_size, freeze_reference_training  # noqa: E402
from afl_vlm.config import load_config, validate_config  # noqa: E402
from afl_vlm.scheduling.train_plan import (  # noqa: E402
    build_train_plan,
    event_records,
    load_system_profile,
    load_train_plan,
    optimizer_steps_for_epochs,
)

from tools.generate_system_profiles import (  # noqa: E402
    SETTINGS,
    _client_specs,
    _source_commit,
    _validate_profile_name,
    _write_text_lf,
    build,
)


def _signatures(rows: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    return {
        (
            r["client_id"],
            r["task"],
            r["dataset"],
            r["num_samples"],
            float(r.get("task_compute_factor", 1.0)),
        )
        for r in rows
    }


def create_profile(
    root: Path,
    name: str,
    *,
    local_epochs: int | None = None,
    seed: int | None = None,
    reference_world_size: int | None = None,
    system_profile_from: str | None = None,
) -> Path:
    """Reuse only intrinsic client properties, NEVER an iterative async Plan."""
    name = _validate_profile_name(name)
    target = root / "one_shot_profiles" / name
    if target.exists():
        raise FileExistsError(f"One-shot profile already exists; will not overwrite: {target}")
    if local_epochs is not None and local_epochs <= 0:
        raise ValueError("local_epochs must be positive")
    if system_profile_from:
        _validate_profile_name(system_profile_from)
    reference = (
        detect_reference_world_size() if reference_world_size is None else reference_world_size
    )
    files: dict[str, bytes] = {}
    training_records: dict[str, Any] = {}
    run_seed: int | None = None
    for setting in SETTINGS:
        source = root / "configs/experiments/llava/afvlm_cm" / f"{setting}clients/opcm_oneshot.yaml"
        cfg = copy.deepcopy(load_config(source))
        cfg.pop("_config_path", None)
        if local_epochs is not None:
            cfg["training"]["local_epochs"] = local_epochs
        cfg["training"] = freeze_reference_training(cfg["training"], reference)
        current_seed = int(cfg["run"]["seed"] if seed is None else seed)
        if run_seed is not None and current_seed != run_seed:
            raise ValueError("One-shot source configs must share a run seed")
        run_seed = current_seed
        cfg["run"]["seed"] = run_seed
        costs = dict(cfg["federation"]["task_compute_factors"])
        actual_system = build(root, setting, run_seed, costs)
        if system_profile_from:
            system_source = (
                root
                / "experiment_profiles"
                / system_profile_from
                / "plans"
                / f"{setting}clients/system_profile.json"
            )
            system_bytes = system_source.read_bytes()
            system = json.loads(system_bytes)
            if _signatures(system["clients"]) != _signatures(actual_system["clients"]):
                raise ValueError(
                    f"Reused system metadata differs from actual data/task costs: {setting}"
                )
            if int(system["seed"]) != run_seed:
                raise ValueError("Reused system seed differs from run seed")
        else:
            system = actual_system
            system_bytes = (json.dumps(system, indent=2) + "\n").encode()
        clients = _client_specs(system)
        if len(clients) != 6 * setting or len({c.id for c in clients}) != len(clients):
            raise ValueError("One-shot system metadata has wrong/duplicate client population")
        train = cfg["training"]
        plan = build_train_plan(
            clients,
            rounds=1,
            local_epochs=int(train["local_epochs"]),
            batch_size=int(train["batch_size"]),
            gradient_accumulation=int(train["gradient_accumulation"]),
            mode="synchronous",
        )
        prefix = f"one_shot_profiles/{name}"
        cfg["federation"]["system_profile"] = f"{prefix}/plans/{setting}clients/system_profile.json"
        cfg["federation"]["train_plan"] = (
            f"{prefix}/plans/{setting}clients/one_shot_train_plan.json"
        )
        cfg["output"]["directory"] = (
            f"runs/llava/afvlm_cm/oneshot/profiles/{name}/{setting}clients/opcm_oneshot/seed{run_seed}"
        )
        cfg["experiment_profile"] = {
            "name": name,
            "family": "one_shot",
            "immutable_snapshot": True,
        }
        validate_config(cfg, check_paths=False)
        files[f"plans/{setting}clients/system_profile.json"] = system_bytes
        files[f"plans/{setting}clients/one_shot_train_plan.json"] = (
            json.dumps(event_records(plan), indent=2) + "\n"
        ).encode()
        files[f"configs/{setting}clients/opcm_oneshot.yaml"] = yaml.safe_dump(
            cfg,
            sort_keys=False,
            allow_unicode=True,
        ).encode()
        training_records[str(setting)] = train
    manifest = {
        "schema_version": 1,
        "family": "one_shot",
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _source_commit(root),
        "settings": list(SETTINGS),
        "methods": ["opcm_oneshot"],
        "run_seed": run_seed,
        "reference_world_size": reference,
        "rounds": 1,
        "uploads_per_client": 1,
        "training": training_records,
        "system_profile_source": system_profile_from or "local_data",
        "plan_source": "new_single_upload_synchronous_plan",
        "files_sha256": {path: hashlib.sha256(body).hexdigest() for path, body in files.items()},
    }
    for relative, body in files.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    _write_text_lf(target / "manifest.yaml", yaml.safe_dump(manifest, sort_keys=False))
    return target


def validate_profile(root: Path, name: str) -> dict[str, Any]:
    """Hash/config/Plan checks only; no image stat scan, GPU allocation or model load."""
    name = _validate_profile_name(name)
    profile = root / "one_shot_profiles" / name
    manifest = yaml.safe_load((profile / "manifest.yaml").read_text(encoding="utf-8"))
    if (
        manifest.get("family") != "one_shot"
        or manifest.get("name") != name
        or manifest.get("methods") != ["opcm_oneshot"]
        or tuple(manifest.get("settings", ())) != SETTINGS
        or manifest.get("rounds") != 1
        or manifest.get("uploads_per_client") != 1
    ):
        raise ValueError("Invalid one-shot profile manifest")
    expected_files = {
        f"{kind}/{setting}clients/{filename}"
        for setting in SETTINGS
        for kind, filename in (
            ("plans", "system_profile.json"),
            ("plans", "one_shot_train_plan.json"),
            ("configs", "opcm_oneshot.yaml"),
        )
    }
    hashes = manifest.get("files_sha256", {})
    if set(hashes) != expected_files:
        raise ValueError("One-shot profile file manifest differs from expected files")
    for relative, expected in hashes.items():
        if hashlib.sha256((profile / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"One-shot immutable profile hash mismatch: {relative}")
    from afl_vlm.methods.registry import create_method

    for setting in SETTINGS:
        cfg = load_config(profile / f"configs/{setting}clients/opcm_oneshot.yaml")
        validate_config(cfg, check_paths=False)
        plan_path = profile / f"plans/{setting}clients/one_shot_train_plan.json"
        system_path = profile / f"plans/{setting}clients/system_profile.json"
        if cfg["federation"]["train_plan"] != plan_path.relative_to(root).as_posix():
            raise ValueError("One-shot config points to another TrainPlan")
        if cfg["federation"]["system_profile"] != system_path.relative_to(root).as_posix():
            raise ValueError("One-shot config points to another system profile")
        if cfg["training"] != manifest["training"][str(setting)]:
            raise ValueError("One-shot training controls differ from manifest")
        if cfg["method"]["name"] != "opcm_oneshot":
            raise ValueError("One-shot profile selects another method")
        if cfg.get("experiment_profile") != {
            "name": name,
            "family": "one_shot",
            "immutable_snapshot": True,
        }:
            raise ValueError("One-shot configuration identity differs from profile")
        if (
            cfg["run"]["seed"] != manifest["run_seed"]
            or cfg["training"].get("reference_world_size") != manifest["reference_world_size"]
            or cfg["dataset"]["clients_per_task"] != setting
        ):
            raise ValueError("One-shot seed/batch reference/dataset setting differs from manifest")
        expected_output = (
            f"runs/llava/afvlm_cm/oneshot/profiles/{name}/{setting}clients/"
            f"opcm_oneshot/seed{manifest['run_seed']}"
        )
        if cfg["output"]["directory"] != expected_output:
            raise ValueError("One-shot output is not isolated in its profile namespace")
        clients, plan = load_system_profile(system_path), load_train_plan(plan_path)
        if len(clients) != 6 * setting or len({c.id for c in clients}) != len(clients):
            raise ValueError("One-shot system profile has wrong/duplicate client population")
        method = create_method("opcm_oneshot", cfg["method"]["params"])
        # Schedule validation does not need weights or a fake model.
        method.clients = {c.id: (c.task, c.dataset, c.num_samples) for c in clients}
        method.configure_schedule(plan)
        train = cfg["training"]
        expected_steps = {
            c.id: optimizer_steps_for_epochs(
                c.num_samples,
                int(train["local_epochs"]),
                int(train["batch_size"]),
                int(train["gradient_accumulation"]),
            )
            for c in clients
        }
        if any(e.local_steps != expected_steps[e.client_id] for e in plan):
            raise ValueError("One-shot Plan local work differs from training controls")
        compiled = build_train_plan(
            clients,
            1,
            int(train["local_epochs"]),
            int(train["batch_size"]),
            int(train["gradient_accumulation"]),
            "synchronous",
        )
        if event_records(plan) != event_records(compiled):
            raise ValueError("One-shot saved Plan differs from runtime's synchronous compiler")
    return {"profile": name, "family": "one_shot", "configs": 3, "uploads_per_client": 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--profile", help="New standalone immutable profile name")
    parser.add_argument("--local_epochs", type=int, help="Default: oneshot method YAML (10)")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--reference_world_size", type=int, help="Optional offline GPU count override"
    )
    parser.add_argument("--system_profile_from", help="Reuse an ordinary profile's SYSTEM only")
    parser.add_argument(
        "--validate", metavar="PROFILE", help="Check an existing one-shot snapshot only"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.validate:
            if args.profile:
                parser.error("Use --validate OR --profile, not both")
            print(json.dumps(validate_profile(root, args.validate), ensure_ascii=False, indent=2))
            return
        if not args.profile:
            parser.error("--profile is required when creating a one-shot snapshot")
        path = create_profile(
            root,
            args.profile,
            local_epochs=args.local_epochs,
            seed=args.seed,
            reference_world_size=args.reference_world_size,
            system_profile_from=args.system_profile_from,
        )
        print(f"Created standalone one-shot profile: {path}")
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
