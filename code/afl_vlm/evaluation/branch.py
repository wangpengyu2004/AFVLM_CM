"""Isolated LoRA branch evaluator with model, method, and RNG restoration."""

from __future__ import annotations

import random
import sys
from collections.abc import Mapping
from typing import Any

from afl_vlm.evaluation.probes import evaluate_tasks
from afl_vlm.models.base import LoRAState, add_scaled, state_hash


class BranchEvaluator:
    def __init__(self, model: Any, method: Any, server_lr: float) -> None:
        self.model = model
        self.method = method
        self.server_lr = server_lr

    def evaluate_branches(
        self,
        receiving_state: LoRAState,
        deltas: Mapping[str, LoRAState],
        tasks: dict[str, Any],
        sample_ids: dict[str, list[str]],
        mode: str = "probe",
    ) -> tuple[dict[str, dict[str, dict[str, float]]], str]:
        original_model = self.model.snapshot_trainable()
        original_method = self.method.state_dict()
        original_rng = random.getstate()
        torch_rng = None
        cuda_rng = None
        if "torch" in sys.modules:
            torch = sys.modules["torch"]
            torch_rng = torch.random.get_rng_state()
            if torch.cuda.is_available():
                cuda_rng = torch.cuda.get_rng_state_all()
        original_hash = state_hash(original_model)
        results: dict[str, dict[str, dict[str, float]]] = {}
        try:
            for branch, delta in deltas.items():
                self.model.load_trainable(receiving_state)
                self.method.load_state_dict(original_method)
                random.setstate(original_rng)
                branch_state = add_scaled(receiving_state, delta, self.server_lr)
                self.model.load_trainable(branch_state)
                results[branch] = evaluate_tasks(self.model, tasks, sample_ids, mode=mode)
        finally:
            self.model.load_trainable(original_model)
            self.method.load_state_dict(original_method)
            random.setstate(original_rng)
            if torch_rng is not None:
                torch = sys.modules["torch"]
                torch.random.set_rng_state(torch_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
        restored_hash = state_hash(self.model.snapshot_trainable())
        if restored_hash != original_hash:
            raise RuntimeError("Diagnostic branch polluted the main model state")
        return results, restored_hash
