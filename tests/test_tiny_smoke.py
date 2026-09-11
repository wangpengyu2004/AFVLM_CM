from pathlib import Path

import pytest
from afl_vlm.analysis.summarize import summarize
from afl_vlm.config import load_config
from afl_vlm.runner import execute


def test_tiny_mock_completes_e1_e2_and_writes_artifacts(tmp_path: Path) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "run.yaml")
    config["run"]["output_root"] = str(tmp_path / "runs")
    rows = execute(config)
    assert {row["experiment"] for row in rows} == {"e1", "e2"}
    for experiment in ("e1", "e2"):
        directory = tmp_path / "runs" / experiment / "controlled" / "additive" / "seed42"
        for name in (
            "config.resolved.yaml",
            "schedule.json",
            "events.jsonl",
            "probes.jsonl",
            "updates.jsonl",
            "branches.jsonl",
            "summary.json",
            "summary.csv",
        ):
            assert (directory / name).is_file()
    aggregate = summarize(tmp_path / "runs")
    assert {row["experiment"] for row in aggregate} == {"e1", "e2"}
    with pytest.raises(FileExistsError, match="already exists"):
        execute(config)
