"""Validate the complete run configuration without loading data or a model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPOSITORY_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from afl_vlm.config import expand_runs, load_config, validate_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    validate_config(config)
    print(f"Configuration is valid and expands to {len(expand_runs(config))} run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
