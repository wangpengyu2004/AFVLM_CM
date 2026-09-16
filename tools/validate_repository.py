"""Static-only repository validation; never loads a model or trains."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from afl_vlm.config import load_config, validate_config  # noqa: E402
from afl_vlm.data.afvlm_cm import scan_partitions  # noqa: E402
from afl_vlm.methods.registry import method_names  # noqa: E402
from afl_vlm.scheduling.train_plan import (  # noqa: E402
    load_system_profile,
    load_train_plan,
    optimizer_steps_for_epochs,
)

EXPECTED_METHODS = (
    "local",
    "fedavg",
    "fedprox",
    "fedadam",
    "fedasync",
    "fedbuff",
    "fedcompass",
    "fedasmu",
    "masfl",
    "adamasfl",
    "pilot",
    "unifed_lora",
    "ours",
)
SETTINGS = (2, 5, 10)


def integrity() -> dict[str, object]:
    root = ROOT / "data" / "AFVLM_CM" / "partitioned"
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    total = 0
    for path in files:
        body = path.read_bytes()
        item_hash = hashlib.sha256(body).hexdigest()
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(item_hash.encode())
        total += len(body)
    return {"files": len(files), "bytes": total, "aggregate_sha256": digest.hexdigest()}


def validate_profiles(errors: list[str]) -> int:
    """Validate immutable profile manifests, configs, hashes, and plan compatibility."""
    profiles_root = ROOT / "experiment_profiles"
    if not profiles_root.is_dir():
        return 0
    count = 0
    for profile_root in sorted(path for path in profiles_root.iterdir() if path.is_dir()):
        count += 1
        manifest_path = profile_root / "manifest.yaml"
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("name") != profile_root.name:
                errors.append(f"profile {profile_root.name}: manifest name mismatch")
            if tuple(manifest.get("settings", ())) != SETTINGS:
                errors.append(f"profile {profile_root.name}: setting list mismatch")
            if set(manifest.get("methods", ())) != set(EXPECTED_METHODS):
                errors.append(f"profile {profile_root.name}: method list mismatch")
            expected_hashes = manifest.get("files_sha256", {})
            for relative, expected_hash in expected_hashes.items():
                path = profile_root / relative
                if not path.is_file():
                    errors.append(f"profile {profile_root.name}: missing {relative}")
                elif hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                    errors.append(f"profile {profile_root.name}: hash mismatch for {relative}")
            for setting in SETTINGS:
                plan_root = profile_root / "plans" / f"{setting}clients"
                system_path = plan_root / "system_profile.json"
                plan_path = plan_root / "async_train_plan.json"
                profile = load_system_profile(system_path)
                plan = load_train_plan(plan_path)
                configs = profile_root / "configs" / f"{setting}clients"
                paths = sorted(configs.glob("*.yaml"))
                if {path.stem for path in paths} != set(EXPECTED_METHODS):
                    errors.append(
                        f"profile {profile_root.name}/{setting}: config method set mismatch"
                    )
                    continue
                reference = load_config(configs / "fedavg.yaml")
                training = reference["training"]
                rounds = int(reference["federation"]["rounds"])
                expected_steps = {
                    item.id: optimizer_steps_for_epochs(
                        item.num_samples,
                        int(training["local_epochs"]),
                        int(training["batch_size"]),
                        int(training["gradient_accumulation"]),
                    )
                    for item in profile
                }
                if len(plan) != len(profile) * rounds:
                    errors.append(f"profile {profile_root.name}/{setting}: event count mismatch")
                if any(item.local_steps != expected_steps[item.client_id] for item in plan):
                    errors.append(f"profile {profile_root.name}/{setting}: local-step mismatch")
                for path in paths:
                    config = load_config(path)
                    validate_config(config, check_paths=False)
                    expected_plan = plan_path.relative_to(ROOT).as_posix()
                    expected_system = system_path.relative_to(ROOT).as_posix()
                    if config["federation"]["train_plan"] != expected_plan:
                        errors.append(f"profile config uses wrong plan: {path}")
                    if config["federation"]["system_profile"] != expected_system:
                        errors.append(f"profile config uses wrong system profile: {path}")
                    if config.get("experiment_profile", {}).get("name") != profile_root.name:
                        errors.append(f"profile identity missing from config: {path}")
        except Exception as exc:
            errors.append(f"profile {profile_root.name}: {exc}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow_missing_data",
        action="store_true",
        help="Validate a clean code checkout where local AFVLM-CM data is intentionally absent",
    )
    args = parser.parse_args()
    errors = []
    data_root = ROOT / "data" / "AFVLM_CM" / "partitioned"
    data_available = data_root.is_dir()
    if not data_available and not args.allow_missing_data:
        errors.append(f"immutable AFVLM-CM partition is missing: {data_root}")
    registered = set(method_names())
    if registered != set(EXPECTED_METHODS):
        errors.append(f"method registry mismatch: {sorted(registered)}")
    from afl_vlm.methods.registry import create_method

    try:
        create_method("ours", {}).validate_runtime()
        errors.append("ours must remain an unavailable placeholder")
    except NotImplementedError as exc:
        if str(exc) != "The proposed AFVLM method has not been implemented yet.":
            errors.append(f"ours placeholder message changed: {exc}")
    experiments = sorted(
        (ROOT / "configs" / "experiments" / "llava" / "afvlm_cm").glob("*clients/*.yaml")
    )
    if len(experiments) != 39:
        errors.append(f"expected 39 experiment configs, found {len(experiments)}")
    for path in experiments:
        try:
            config = load_config(path)
            validate_config(config, check_paths=data_available)
            if config["method"]["name"] != path.stem:
                errors.append(f"method/file mismatch: {path}")
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    for setting in SETTINGS:
        setting_root = ROOT / "configs" / "experiments" / "llava" / "afvlm_cm" / f"{setting}clients"
        resolved = {
            method: load_config(setting_root / f"{method}.yaml") for method in EXPECTED_METHODS
        }
        cfg = resolved["fedavg"]
        try:
            if data_available:
                _, clients = scan_partitions(cfg["dataset"])
                if len(clients) != 6 * setting:
                    errors.append(f"{setting} clients/task: detected total {len(clients)}")
            reference = resolved["fedavg"]
            for method, candidate in resolved.items():
                for section in (
                    "run",
                    "runtime",
                    "model",
                    "dataset",
                    "training",
                    "evaluation",
                ):
                    if candidate[section] != reference[section]:
                        errors.append(f"{method}/{setting} changes shared {section} settings")
                for key in (
                    "rounds",
                    "system_profile",
                    "train_plan",
                    "task_compute_factors",
                ):
                    if candidate["federation"][key] != reference["federation"][key]:
                        errors.append(f"{method}/{setting} changes shared federation.{key}")
                expected_output = f"runs/llava/afvlm_cm/{setting}clients/{method}/seed42"
                if candidate["output"]["directory"] != expected_output:
                    errors.append(
                        f"{method}/{setting} output is {candidate['output']['directory']}, "
                        f"expected {expected_output}"
                    )
            expected_plan = f"plans/afvlm_cm/{setting}clients/async_train_plan.json"
            async_methods = ("fedasync", "fedbuff", "fedasmu", "masfl", "adamasfl", "ours")
            for method in async_methods:
                method_cfg = resolved[method]
                if method_cfg["federation"]["train_plan"] != expected_plan:
                    errors.append(f"{method}/{setting} does not share the base TrainPlan")
            # The canonical plans are a legacy/default snapshot. Current configs are
            # intentionally mutable so a user can create a new immutable profile from
            # changed local-work parameters. Strict config/Plan compatibility therefore
            # belongs to validate_profiles(), not to this compatibility path.
            profile = load_system_profile(ROOT / reference["federation"]["system_profile"])
            plan = load_train_plan(ROOT / expected_plan)
            profile_ids = {item.id for item in profile}
            if any(item.client_id not in profile_ids for item in plan):
                errors.append(f"{setting} clients/task: TrainPlan contains an unknown client")
        except Exception as exc:
            errors.append(f"dataset {setting}: {exc}")
    actual: dict[str, object] | None = None
    if data_available:
        expected = json.loads(
            (ROOT / "configs" / "datasets" / "afvlm_cm_integrity.json").read_text(encoding="utf-8")
        )
        actual = integrity()
        for key in ("files", "bytes", "aggregate_sha256"):
            if actual[key] != expected[key]:
                errors.append(f"immutable data {key}: expected {expected[key]}, got {actual[key]}")
    profile_count = validate_profiles(errors)
    report = {
        "status": "ok" if not errors else "failed",
        "checks": {
            "registered_methods": list(EXPECTED_METHODS),
            "experiment_configs": len(experiments),
            "experiment_profiles": profile_count,
            "immutable_partition": actual or "not present (explicitly allowed)",
        },
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
