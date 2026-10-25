"""Combine completed rank sweeps in a new directory without modifying either source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from afl_vlm.experiments.rank_probe_history import merge_probe_runs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directories", type=Path, nargs="+")
    parser.add_argument("--output-directory", type=Path, required=True,
                        help="New combined result directory; existing paths are rejected")
    args = parser.parse_args()
    result = merge_probe_runs(args.run_directories, args.output_directory)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
