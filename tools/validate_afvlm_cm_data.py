"""Explicitly validate AFVLM-CM annotations and referenced images."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from afl_vlm.config import load_config  # noqa: E402
from afl_vlm.data.afvlm_cm import AFVLMDataModule  # noqa: E402


def _validate(setting: int) -> dict[str, object]:
    config_path = (
        ROOT
        / "configs"
        / "experiments"
        / "llava"
        / "afvlm_cm"
        / f"{setting}clients"
        / "fedavg.yaml"
    )
    config = load_config(config_path)
    report = AFVLMDataModule(config["dataset"]).preflight_validate(progress=True)
    return {"setting": setting, "config": str(config_path.relative_to(ROOT)), **report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--setting", type=int, choices=(2, 5, 10))
    choice.add_argument("--all", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    settings = (2, 5, 10) if args.all else (args.setting,)
    reports = [_validate(setting) for setting in settings]
    print(json.dumps({"status": "ok", "settings": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
