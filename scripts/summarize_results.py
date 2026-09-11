"""CLI wrapper for cross-seed summaries."""

# ruff: noqa: E402, I001

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPOSITORY_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from afl_vlm.analysis.summarize import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
