from dataclasses import dataclass

from afl_vlm.evaluation.branch import BranchEvaluator
from afl_vlm.methods.baselines.async_additive import AsyncAdditiveMethod
from afl_vlm.models.base import state_hash
from afl_vlm.models.tiny_mock import TinyMockAdapter


@dataclass
class MockTask:
    task_key: str


def test_branch_evaluation_restores_main_state() -> None:
    model = TinyMockAdapter()
    model.load({"mock_dimension": 3})
    method = AsyncAdditiveMethod({"server_lr": 0.5})
    receiving = model.snapshot_trainable()
    before = state_hash(receiving)
    evaluator = BranchEvaluator(model, method, 0.5)
    results, restored = evaluator.evaluate_branches(
        receiving,
        {
            "stale": {"adapter": [0.3, 0.2, -0.1]},
            "fresh": {"adapter": [0.1, 0.2, -0.1]},
        },
        {"fast": MockTask("fast"), "slow": MockTask("slow")},
        {"fast": ["f0"], "slow": ["s0"]},
    )
    assert set(results) == {"stale", "fresh"}
    assert restored == before == state_hash(model.snapshot_trainable())
