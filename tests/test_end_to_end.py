import json
from pathlib import Path

from afl_vlm.config import load_config
from afl_vlm.runner import execute


def test_equal_quota_trace_and_delay_control(tmp_path: Path) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "run.yaml")
    config["run"]["output_root"] = str(tmp_path / "trace")
    for experiment in config["experiments"]:
        experiment["enabled"] = experiment["id"] == "trace"
    rows = execute(config)
    assert {row["scenario"] for row in rows} == {"task_correlated", "task_shuffled"}
    assert all(row["received_updates"] == 8 for row in rows)
    assert all(set(row["upload_counts"].values()) == {2} for row in rows)
    for scenario in ("task_correlated", "task_shuffled"):
        path = tmp_path / "trace" / "trace" / scenario / "additive" / "seed42"
        events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
        assert len(events) == 8
        required = {
            "event_id",
            "virtual_time",
            "client_id",
            "task",
            "local_round",
            "download_version",
            "receive_version",
            "staleness",
            "virtual_duration",
            "applied_weight",
            "recent_window_task_counts",
        }
        assert required <= set(events[0])
