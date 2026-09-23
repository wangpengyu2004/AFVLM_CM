"""Evaluate a trainable-state checkpoint with the formal per-task pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from afl_vlm.config import load_config, validate_config  # noqa: E402
from afl_vlm.data.afvlm_cm import AFVLMDataModule  # noqa: E402
from afl_vlm.methods.registry import create_method  # noqa: E402
from afl_vlm.models.registry import create_model  # noqa: E402
from afl_vlm.runner import evaluate_method  # noqa: E402
from afl_vlm.scheduling.train_plan import load_system_profile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    config = load_config(args.config)
    validate_config(config)
    dataset_config = dict(config["dataset"])
    dataset_config["evaluation"] = dict(config["evaluation"])
    data_module = AFVLMDataModule(dataset_config)
    tasks = data_module.tasks
    method = create_method(config["method"]["name"], config["method"].get("params", {}))
    method.validate_runtime()
    model = create_model(config["model"]["adapter"])
    model_config = dict(config["model"])
    model_config["max_text_length"] = config["training"]["max_text_length"]
    model_config["max_new_tokens"] = config["evaluation"]["generation_max_new_tokens"]
    model.load(model_config)
    method.configure_model(model, load_system_profile(config["federation"]["system_profile"]))
    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_trainable(checkpoint["federated_state"])
    method.load_state_dict(checkpoint["method_state"])
    result = {
        "protocol": method.evaluation_scope,
        "split": config["evaluation"].get("final_split", "final"),
        "server_version": checkpoint["server_version"],
        "per_task": evaluate_method(
            model,
            method,
            checkpoint["federated_state"],
            tasks,
            config["evaluation"].get("final_split", "final"),
        ),
        "note": "No cross-task raw average is reported.",
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
