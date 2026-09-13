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
    args = parser.parse_args()
    os.chdir(REPOSITORY_ROOT)
    config = load_config(args.config)
    if args.overwrite:
        config["output"]["overwrite"] = True
    print(json.dumps(execute(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
