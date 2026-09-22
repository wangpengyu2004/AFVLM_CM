"""Print the capability-selected physical backend for one experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from afl_vlm.config import load_config  # noqa: E402
from afl_vlm.execution import resolve_runtime_backend  # noqa: E402
from afl_vlm.methods.registry import create_method  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    method = create_method(config["method"]["name"], config["method"].get("params", {}))
    print(resolve_runtime_backend(config, method))


if __name__ == "__main__":
    main()
