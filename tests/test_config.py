from copy import deepcopy
from pathlib import Path

import pytest
from afl_vlm.config import expand_runs, load_config, validate_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "run.yaml"


def test_default_config_is_complete_and_expands_to_e1_e2() -> None:
    config = load_config(CONFIG)
    validate_config(config)
    assert [run.experiment_id for run in expand_runs(config)] == ["e1", "e2"]


def test_unknown_field_fails_before_training() -> None:
    config = load_config(CONFIG)
    broken = deepcopy(config)
    broken["clients"]["surprise"] = True
    with pytest.raises(ValueError, match="Unknown configuration"):
        validate_config(broken)


def test_client_count_mismatch_fails() -> None:
    config = load_config(CONFIG)
    broken = deepcopy(config)
    broken["clients"]["count"] = 99
    with pytest.raises(ValueError, match="does not match"):
        validate_config(broken)
