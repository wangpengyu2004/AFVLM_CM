"""Generate deterministic method-independent AFVLM-CM system metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from afl_vlm.scheduling.train_plan import (  # noqa: E402
    build_train_plan,
    event_records,
    load_system_profile,
)

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


def build(root: Path, setting: int, seed: int) -> dict:
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
    return {
        "schema_version": 1,
        "setting": setting,
        "seed": seed,
        "generation_scope": "method_independent",
        "clients": clients,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for setting in (2, 5, 10):
        target = args.root / "plans" / "afvlm_cm" / f"{setting}clients"
        target.mkdir(parents=True, exist_ok=True)
        (target / "system_profile.json").write_text(
            json.dumps(build(args.root, setting, args.seed), indent=2) + "\n", encoding="utf-8"
        )
        plan = build_train_plan(
            load_system_profile(target / "system_profile.json"),
            rounds=10,
            base_local_steps=10,
            mode="asynchronous",
        )
        (target / "async_train_plan.json").write_text(
            json.dumps(event_records(plan), indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
