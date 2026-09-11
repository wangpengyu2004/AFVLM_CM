import pytest
from afl_vlm.evaluation.paired_metrics import adaptation_gap, extra_harm


def test_extra_harm_uses_paired_within_task_differences() -> None:
    result = extra_harm(
        {"score": 0.8},
        {"score": 0.6},
        {"score": 0.75},
    )
    assert result["damage_stale"] == pytest.approx(0.2)
    assert result["damage_fresh"] == pytest.approx(0.05)
    assert result["extra_harm"] == pytest.approx(0.15)


def test_adaptation_gap_uses_loss_direction() -> None:
    result = adaptation_gap(
        {"loss": 1.4},
        {"loss": 1.0},
        {"loss": 0.8},
        {"loss": 0.7},
    )
    assert result == pytest.approx({"initial_loss_gap": 0.4, "post_train_loss_gap": 0.1})
