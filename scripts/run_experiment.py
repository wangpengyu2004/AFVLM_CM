"""Run one formal AFVLM-CM experiment configuration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPOSITORY_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from afl_vlm.config import load_config  # noqa: E402
from afl_vlm.runner import execute  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--runtime-backend",
        choices=("serial", "client_parallel", "client_ddp"),
        help="Internal override selected by scripts/run_one.sh",
    )
    parser.add_argument("--local-rank", "--local_rank", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(REPOSITORY_ROOT)
    config = load_config(args.config)
    if args.runtime_backend:
        runtime = config.setdefault("runtime", {})
        runtime["backend"] = args.runtime_backend
        runtime["executor_policy"] = "fixed"
    if args.overwrite:
        config["output"]["overwrite"] = True
    try:
        result = execute(config)
    finally:
        # execute_serial handles successful teardown; this also covers errors.
        if args.runtime_backend == "client_ddp":
            import torch

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
