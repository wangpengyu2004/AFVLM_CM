"""Run all enabled experiment × method × seed × scenario combinations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPOSITORY_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from afl_vlm.config import expand_runs, load_config  # noqa: E402
from afl_vlm.runner import dry_run, execute  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the single complete YAML config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print schedules without loading a model",
    )
    parser.add_argument("--output-root", help="Optional output-root override for this invocation")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacement of existing run artifacts",
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.output_root:
        config["run"]["output_root"] = arguments.output_root
    if arguments.overwrite:
        config["output"]["overwrite"] = True
    if arguments.dry_run:
        print(json.dumps(dry_run(config), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    rows = execute(config)
    print(
        json.dumps(
            {
                "completed_runs": len(expand_runs(config)),
                "summary_rows": len(rows),
                "output_root": config["run"]["output_root"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
